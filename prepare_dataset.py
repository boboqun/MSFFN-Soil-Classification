#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_dataset.py
==================
Image-level isolated patch generation with ruler-aware filtering.

Reads from a source directory containing a clean train/validation split at
the original-image level and generates 224x224 patches at stride 180.

Output directory structure:
    dataset/
        train/      <- patches from dataset_original/train  (183 images)
        val/        <- patches from ~50% of dataset_original/validation images
        test/       <- patches from ~50% of dataset_original/validation images

Key guarantees:
    - NO original image contributes patches to more than one subset.
    - A split_manifest.json records every original->subset mapping.

Usage:
    ORIG_ROOT=./dataset_original python prepare_dataset.py
"""

import os, sys, json, random, hashlib, time
from collections import defaultdict, Counter
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────────────
PATCH_SIZE = 224
STRIDE     = 180
RANDOM_SEED = 42
VAL_TEST_RATIO = 0.5  # 50% of held-out originals → val, 50% → test

# Source: the clean original-image split
# Set ORIG_ROOT via environment variable or default to ./dataset_original
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIG_ROOT  = os.environ.get('ORIG_ROOT',
                            os.path.join(SCRIPT_DIR, 'dataset_original'))
ORIG_TRAIN = os.path.join(ORIG_ROOT, 'train')
ORIG_VAL   = os.path.join(ORIG_ROOT, 'validation')

# Destination: patched dataset with image-level isolation
DEST_ROOT  = os.path.join(SCRIPT_DIR, 'dataset')

# Class name mapping (supports both Chinese and English folder names)
CLASS_MAP_ZH_EN = {'Loam': 'Loam', 'Sand': 'Sand', 'Clay': 'Clay'}
CLASS_MAP_EN_EN = {'Loam': 'Loam', 'Sand': 'Sand', 'Clay': 'Clay'}

def normalize_class(name):
    """Map any class directory name to canonical English."""
    if name in CLASS_MAP_ZH_EN:
        return CLASS_MAP_ZH_EN[name]
    if name in CLASS_MAP_EN_EN:
        return CLASS_MAP_EN_EN[name]
    return name

# ── Ruler (white label) detection ───────────────────────────────────────────
# Ported from experiment_v2/filter_ruler_patches.py.
# The ruler is a white card with black markings placed next to soil samples
# for scale reference. It appears as a bright rectangular region in the image.
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️  opencv-python not installed — ruler filtering disabled.")
    print("   Install with: pip install opencv-python")

def detect_ruler_mask(img_path, bright_thresh=200, min_ruler_area=2000):
    """
    Detect ruler (white scale-card) regions in the image.
    Returns (ruler_mask, has_ruler) where ruler_mask is a binary numpy array
    (255=ruler, 0=soil) and has_ruler is a boolean.
    """
    if not HAS_CV2:
        return None, False

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None, False

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Threshold to find bright-white regions
    _, bright_mask = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)

    # Morphological close: fill gaps from ruler markings
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    closed = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

    # Dilate to cover ruler edges
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    dilated = cv2.dilate(closed, dilate_kernel)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    import numpy as _np
    ruler_mask = _np.zeros(gray.shape, dtype=_np.uint8)
    found = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_ruler_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / (min(w, h) + 1e-6)
        # Ruler is elongated (aspect ratio 1.5-15)
        if aspect < 1.5 or aspect > 15:
            continue
        # Verify: white pixel ratio > 35% within bounding box
        roi = bright_mask[y:y+h, x:x+w]
        white_ratio = _np.sum(roi > 0) / (w * h + 1e-6)
        if white_ratio < 0.35:
            continue
        # Mark as ruler region with padding
        pad = 20
        y1 = max(0, y - pad)
        y2 = min(gray.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(gray.shape[1], x + w + pad)
        ruler_mask[y1:y2, x1:x2] = 255
        found = True

    return ruler_mask, found


def patch_overlaps_ruler(ruler_mask, px, py, patch_size, max_overlap_ratio=0.05):
    """Check if a patch overlaps with ruler region beyond threshold."""
    import numpy as _np
    roi = ruler_mask[py:py+patch_size, px:px+patch_size]
    overlap_ratio = _np.sum(roi > 0) / (patch_size * patch_size)
    return overlap_ratio > max_overlap_ratio


# ── Patching ────────────────────────────────────────────────────────────────
def extract_patches(img_path, patch_size=PATCH_SIZE, stride=STRIDE):
    """Yield (patch_image, patch_index) tuples from a single image.
    Patches overlapping with detected ruler regions are skipped."""
    # Detect ruler mask for this image
    ruler_mask, has_ruler = detect_ruler_mask(img_path)

    with Image.open(img_path) as img:
        img = img.convert('RGB')
        w, h = img.size
        idx = 0
        skipped = 0
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                # Skip patches overlapping with ruler
                if has_ruler and patch_overlaps_ruler(ruler_mask, x, y, patch_size):
                    skipped += 1
                    continue
                box = (x, y, x + patch_size, y + patch_size)
                yield img.crop(box), idx
                idx += 1
        if skipped > 0:
            print(f"    → Skipped {skipped} ruler-overlapping patches")


def process_directory(src_dir, dest_dir, manifest_entries, subset_name):
    """Process all images in src_dir, save patches to dest_dir/{class}/."""
    total_patches = 0
    for class_dir_raw in sorted(os.listdir(src_dir)):
        class_path = os.path.join(src_dir, class_dir_raw)
        if not os.path.isdir(class_path):
            continue
        class_name = normalize_class(class_dir_raw)
        dest_class = os.path.join(dest_dir, class_name)
        os.makedirs(dest_class, exist_ok=True)

        imgs = sorted([f for f in os.listdir(class_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {subset_name}/{class_name}: {len(imgs)} images")

        for img_name in imgs:
            img_path = os.path.join(class_path, img_name)
            # Create a unique prefix from filename hash (deterministic)
            stem = os.path.splitext(img_name)[0]
            prefix = hashlib.md5(f"{class_name}/{img_name}".encode('utf-8')).hexdigest()[:12]

            n_patches = 0
            for patch, idx in extract_patches(img_path):
                out_name = f"{prefix}_{idx:04d}.jpg"
                patch.save(os.path.join(dest_class, out_name), quality=95)
                n_patches += 1

            total_patches += n_patches
            manifest_entries.append({
                'original_file': img_name,
                'class': class_name,
                'subset': subset_name,
                'prefix': prefix,
                'n_patches': n_patches,
                'source_dir': os.path.basename(os.path.dirname(class_path)),
            })

    return total_patches


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    random.seed(RANDOM_SEED)
    t0 = time.time()

    print("=" * 60)
    print("Image-Level Isolated Patch Generation (v3)")
    print("=" * 60)
    print(f"Patch: {PATCH_SIZE}×{PATCH_SIZE}  Stride: {STRIDE}")
    print(f"Source: {ORIG_ROOT}")
    print(f"Dest:   {DEST_ROOT}")
    print()

    # Clean dest
    if os.path.exists(DEST_ROOT):
        print(f"⚠️  Removing existing {DEST_ROOT}")
        import shutil
        shutil.rmtree(DEST_ROOT)

    manifest = []

    # ── 1. Train set: ALL images from dataset_original/train ────────────
    print("─── Step 1: Train set ───")
    train_dest = os.path.join(DEST_ROOT, 'train')
    os.makedirs(train_dest, exist_ok=True)
    n_train = process_directory(ORIG_TRAIN, train_dest, manifest, 'train')
    print(f"  Total train patches: {n_train}\n")

    # ── 2. Val/Test split from dataset_original/validation ──────────────
    print("─── Step 2: Val/Test split (image-level) ───")
    # Collect all validation images grouped by class
    val_images_by_class = defaultdict(list)
    for class_dir_raw in sorted(os.listdir(ORIG_VAL)):
        class_path = os.path.join(ORIG_VAL, class_dir_raw)
        if not os.path.isdir(class_path):
            continue
        class_name = normalize_class(class_dir_raw)
        imgs = sorted([f for f in os.listdir(class_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for img in imgs:
            val_images_by_class[class_name].append((img, os.path.join(class_path, img)))

    # Stratified split: within each class, randomly assign images to val/test
    val_dest  = os.path.join(DEST_ROOT, 'val')
    test_dest = os.path.join(DEST_ROOT, 'test')
    os.makedirs(val_dest, exist_ok=True)
    os.makedirs(test_dest, exist_ok=True)

    n_val = 0
    n_test = 0
    for class_name in sorted(val_images_by_class.keys()):
        images = val_images_by_class[class_name]
        random.shuffle(images)
        split_point = int(len(images) * VAL_TEST_RATIO)
        val_imgs  = images[:split_point]
        test_imgs = images[split_point:]

        print(f"  {class_name}: {len(val_imgs)} val images, {len(test_imgs)} test images")

        # Process val images
        for img_name, img_path in val_imgs:
            stem = os.path.splitext(img_name)[0]
            prefix = hashlib.md5(f"{class_name}/{img_name}".encode('utf-8')).hexdigest()[:12]
            dest_class = os.path.join(val_dest, class_name)
            os.makedirs(dest_class, exist_ok=True)
            n_patches = 0
            for patch, idx in extract_patches(img_path):
                out_name = f"{prefix}_{idx:04d}.jpg"
                patch.save(os.path.join(dest_class, out_name), quality=95)
                n_patches += 1
            n_val += n_patches
            manifest.append({
                'original_file': img_name,
                'class': class_name,
                'subset': 'val',
                'prefix': prefix,
                'n_patches': n_patches,
                'source_dir': 'validation',
            })

        # Process test images
        for img_name, img_path in test_imgs:
            stem = os.path.splitext(img_name)[0]
            prefix = hashlib.md5(f"{class_name}/{img_name}".encode('utf-8')).hexdigest()[:12]
            dest_class = os.path.join(test_dest, class_name)
            os.makedirs(dest_class, exist_ok=True)
            n_patches = 0
            for patch, idx in extract_patches(img_path):
                out_name = f"{prefix}_{idx:04d}.jpg"
                patch.save(os.path.join(dest_class, out_name), quality=95)
                n_patches += 1
            n_test += n_patches
            manifest.append({
                'original_file': img_name,
                'class': class_name,
                'subset': 'test',
                'prefix': prefix,
                'n_patches': n_patches,
                'source_dir': 'validation',
            })

    print(f"  Total val patches:  {n_val}")
    print(f"  Total test patches: {n_test}\n")

    # ── 3. Save manifest ────────────────────────────────────────────────
    manifest_path = os.path.join(DEST_ROOT, 'split_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ── 4. Verification ─────────────────────────────────────────────────
    print("─── Step 3: Verification ───")
    # Check prefix isolation
    prefixes_by_subset = defaultdict(set)
    for entry in manifest:
        prefixes_by_subset[entry['subset']].add(entry['prefix'])

    train_p = prefixes_by_subset['train']
    val_p   = prefixes_by_subset['val']
    test_p  = prefixes_by_subset['test']

    leak_tv  = train_p & val_p
    leak_tt  = train_p & test_p
    leak_vt  = val_p & test_p

    if leak_tv or leak_tt or leak_vt:
        print(f"  ❌ DATA LEAKAGE DETECTED!")
        print(f"     train∩val:  {len(leak_tv)} prefixes")
        print(f"     train∩test: {len(leak_tt)} prefixes")
        print(f"     val∩test:   {len(leak_vt)} prefixes")
        sys.exit(1)
    else:
        print(f"  ✅ Zero prefix overlap across all three subsets")
        print(f"     train: {len(train_p)} unique original images")
        print(f"     val:   {len(val_p)} unique original images")
        print(f"     test:  {len(test_p)} unique original images")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Train: {n_train} patches from {len(train_p)} images")
    print(f"Val:   {n_val} patches from {len(val_p)} images")
    print(f"Test:  {n_test} patches from {len(test_p)} images")
    print(f"Total: {n_train + n_val + n_test} patches from {len(train_p) + len(val_p) + len(test_p)} images")
    print(f"Manifest: {manifest_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
