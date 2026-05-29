#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gradcam_visualize.py
====================
Generate Grad-CAM class-activation overlays for the four core models on a
common set of test patches.

Target layers (last spatial activation before global pooling):
    teacher              -> 'block6a_expand_activation'  (EfficientNet-B4)
    sota_mobilenetv3     -> 'multiply_17'                (MobileNetV3-Small)
    student_scratch      -> 'scratch_head_feat_gap'      (MSFFN w/o KD)
    student_distilled    -> 'student_head_feat_gap'      (MSFFN + KD)

Outputs:
    <out_dir>/gradcam_comparison.{png,pdf}      composite figure
    <out_dir>/gradcam_panels/<patch>__<model>.png   individual panels

Usage:
    python gradcam_visualize.py --run-tag seed42
    python gradcam_visualize.py --run-tag seed42 \\
        --patches path/to/img1.jpg path/to/img2.jpg
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# ──────────────────────────────────────────────────────────────────
# Paths and constants
# ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_ROOT  = SCRIPT_DIR / 'checkpoints'

IMG_SIZE = 224

DEFAULT_RUN_TAG = 'seed42'


def _build_teacher():
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB4
    from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
    base = EfficientNetB4(weights=None, include_top=False,
                          input_shape=(IMG_SIZE, IMG_SIZE, 3))
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = layers.Lambda(eff_preprocess)(inp)
    x = base(x)
    x = layers.GlobalAveragePooling2D(name='gap_teacher')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(3, activation='softmax')(x)
    return models.Model(inp, out, name='teacher'), 'block6a_expand_activation'


def _build_mobilenetv3():
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV3Small
    base = MobileNetV3Small(weights=None, include_top=False,
                            input_shape=(IMG_SIZE, IMG_SIZE, 3))
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = base(inp)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(3, activation='softmax')(x)
    return models.Model(inp, out, name='sota_mobilenetv3'), 'multiply_17'


def _build_msffn(name, head_name='ms_head'):
    """Reuse ``multiscale_backbone`` and ``multiscale_head`` from run_experiment.py.

    Loading the module triggers its initialisation code (which may call
    ``sys.exit`` if a dataset directory is missing); we swallow that
    SystemExit because we only need the model-building helpers.
    """
    # Provide harmless defaults so that module-level code does not blow up
    # when imported just to access the helper functions.
    os.environ.setdefault('SKIP_DISTILL', '1')
    os.environ.setdefault('EPOCHS', '0')
    os.environ.setdefault('SKIP_HEAVY_SOTA', '1')
    os.environ.setdefault('TRAIN_PATH', str(SCRIPT_DIR / 'dataset' / 'train'))
    os.environ.setdefault('VAL_PATH',   str(SCRIPT_DIR / 'dataset' / 'val'))
    os.environ.setdefault('TEST_PATH',  str(SCRIPT_DIR / 'dataset' / 'test'))

    spec = importlib.util.spec_from_file_location(
        'run_experiment', str(SCRIPT_DIR / 'run_experiment.py'))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    bb = mod.multiscale_backbone(name=f'{name}_bb')
    head = mod.multiscale_head(bb.output, 3, name=head_name)
    from tensorflow.keras import models
    return models.Model(bb.input, head, name=name), f'{head_name}_feat_gap'


MODEL_BUILDERS = {
    'teacher':           _build_teacher,
    'sota_mobilenetv3':  _build_mobilenetv3,
    'student_scratch':   lambda: _build_msffn('student_scratch',  head_name='scratch_head'),
    'student_distilled': lambda: _build_msffn('student_distilled', head_name='student_head'),
}

PRETTY = {
    'sota_mobilenetv3':  'MobileNetV3-Small\n(baseline)',
    'student_scratch':   'MSFFN w/o KD',
    'student_distilled': 'MSFFN + KD\n(ours)',
    'teacher':           'Teacher\nEfficientNet-B4',
}


# ──────────────────────────────────────────────────────────────────
# Grad-CAM core
# ──────────────────────────────────────────────────────────────────
def make_gradcam_heatmap(model, img_array, target_layer_name, pred_index=None):
    """Compute a Grad-CAM heatmap from the pre-softmax logit of ``pred_index``.

    Returns ``(heatmap, pred_index, softmax_probability)``.
    """
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            for sub in layer.layers:
                if sub.name == target_layer_name:
                    base_model = layer
                    break
    
    last_layer = model.layers[-1]
    
    with tf.GradientTape() as tape:
        if base_model is None:
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(target_layer_name).output, last_layer.input]
            )
            conv_out, dense_input = grad_model(img_array, training=False)
        else:
            x = img_array
            conv_out = None
            for layer in model.layers:
                if layer == base_model:
                    inner_grad_model = tf.keras.models.Model(
                        inputs=base_model.inputs,
                        outputs=[base_model.get_layer(target_layer_name).output, base_model.output]
                    )
                    conv_out, x = inner_grad_model(x, training=False)
                elif layer == last_layer:
                    dense_input = x
                else:
                    x = layer(x, training=False)
        
        weights = last_layer.get_weights()
        if len(weights) == 2:
            logits = tf.matmul(dense_input, weights[0]) + weights[1]
        else:
            logits = tf.matmul(dense_input, weights[0])
            
        if pred_index is None:
            pred_index = int(tf.argmax(logits[0]))
        loss = logits[:, pred_index]

    grads = tape.gradient(loss, conv_out)              # (1, h, w, C)
    if grads is None:
        grads = tf.zeros_like(conv_out)
        
    print(f"DEBUG {model.name}: loss={loss.numpy()}, logits={logits.numpy()}")
    print(f"DEBUG {model.name}: grads min={tf.reduce_min(grads).numpy()}, max={tf.reduce_max(grads).numpy()}")

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))     # (C,)
    conv_out = conv_out[0]                              # (h, w, C)
    heatmap = conv_out @ pooled[..., tf.newaxis]       # (h, w, 1)
    heatmap = tf.squeeze(heatmap)
    
    print(f"DEBUG {model.name}: raw heatmap min={tf.reduce_min(heatmap).numpy()}, max={tf.reduce_max(heatmap).numpy()}")
    
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    
    prob = float(tf.nn.softmax(logits)[0, pred_index])
    return heatmap.numpy(), int(pred_index), prob


def overlay_heatmap(img_uint8, heatmap, alpha=0.4):
    """Blend a jet-colormap heatmap on top of the input patch."""
    from matplotlib import cm
    h = np.array(Image.fromarray((heatmap * 255).astype(np.uint8))
                 .resize((img_uint8.shape[1], img_uint8.shape[0]),
                         Image.BILINEAR)) / 255.0
    cmap = cm.get_cmap('jet')
    color = (cmap(h)[..., :3] * 255).astype(np.uint8)
    out = ((1 - alpha) * img_uint8 + alpha * color).clip(0, 255).astype(np.uint8)
    return out


# ──────────────────────────────────────────────────────────────────
# Patch loading
# ──────────────────────────────────────────────────────────────────
def load_patches(patches_arg, n_per_class=2, val_root=None):
    """Resolve which patches to visualise.

    If ``patches_arg`` is provided, those paths are used directly;
    otherwise the first ``n_per_class`` images in each class folder of
    ``val_root`` are picked.
    """
    paths = []
    if patches_arg:
        for p in patches_arg:
            paths.append(Path(p).resolve())
    else:
        val_root = Path(val_root or './dataset/val')
        if not val_root.is_dir():
            raise FileNotFoundError(f'val directory does not exist: {val_root}\n'
                                    f'Please specify files using --patches.')
        for cls_dir in sorted(p for p in val_root.iterdir() if p.is_dir()):
            files = sorted(cls_dir.glob('*.jpg')) + sorted(cls_dir.glob('*.png'))
            paths.extend(files[:n_per_class])
    return paths


def read_image(path):
    img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE),
                                                 Image.BILINEAR)
    return np.array(img)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-tag', default=DEFAULT_RUN_TAG,
                    help="RUN_TAG sub-directory under ./checkpoints/ "
                         "(default: seed42)")
    ap.add_argument('--patches', nargs='*',
                    help="Explicit list of patch image paths to visualise.")
    ap.add_argument('--n-per-class', type=int, default=2,
                    help="When --patches is not given, take this many "
                         "patches per class from --val-root.")
    ap.add_argument('--val-root', default=None,
                    help="Validation directory to sample patches from "
                         "(defaults to ./dataset/val).")
    ap.add_argument('--out-dir', default=str(SCRIPT_DIR / 'plots'),
                    help="Directory for the composite figure and per-panel "
                         "overlays (default: ./plots).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = out_dir / 'gradcam_panels'
    panel_dir.mkdir(exist_ok=True)

    ckpt_dir = CKPT_ROOT / args.run_tag if (CKPT_ROOT / args.run_tag).exists() else CKPT_ROOT
    print(f'Loading checkpoints from: {ckpt_dir}')

    models_info = {}
    for name, builder in MODEL_BUILDERS.items():
        model, layer = builder()
        ckpt = ckpt_dir / f'{name}_best.weights.h5'
        if not ckpt.exists():
            print(f'  ⚠️  {name} missing weights {ckpt}, skipping')
            continue
        try:
            if name == 'student_distilled':
                class DummyDistiller(tf.keras.models.Model):
                    def __init__(self, student):
                        super().__init__()
                        self.student = student
                dummy = DummyDistiller(model)
                dummy.load_weights(str(ckpt))
            else:
                model.load_weights(str(ckpt))
        except Exception as e:
            print(f'  ⚠️  {name} failed to load weights: {e}, skipping')
            continue
        models_info[name] = (model, layer)
        print(f'  ✓ {name} ← {ckpt.name}, target layer = {layer}')

    if not models_info:
        print('No model checkpoints could be loaded; nothing to visualise.')
        sys.exit(1)

    patches = load_patches(args.patches, args.n_per_class, args.val_root)
    print(f'Total {len(patches)} patches to visualise')

    n_rows = len(patches)
    CLASSES = ['Clay', 'Loam', 'Sand']  # sorted alphabetically, matching tf.keras dataset order

    fig, axes = plt.subplots(n_rows, len(models_info) + 1, figsize=(3 * (len(models_info) + 1), 3 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for i, pth in enumerate(patches):
        img = read_image(pth)
        axes[i, 0].imshow(img); axes[i, 0].set_title('Input' if i == 0 else '')
        gt_label = Path(pth).parent.name
        axes[i, 0].set_ylabel(f'{gt_label}\n{Path(pth).stem}', fontsize=7)
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])

        for j, (name, (model, layer)) in enumerate(models_info.items(), 1):
            heatmap, pred_idx, prob = make_gradcam_heatmap(
                model, np.expand_dims(img.astype(np.float32), 0), layer)
            overlay = overlay_heatmap(img, heatmap)
            axes[i, j].imshow(overlay)
            if i == 0:
                axes[i, j].set_title(PRETTY.get(name, name), fontsize=9)

            pred_class = CLASSES[pred_idx] if pred_idx < len(CLASSES) else str(pred_idx)
            axes[i, j].set_xlabel(f'{pred_class} p={prob:.2f}', fontsize=8)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

            Image.fromarray(overlay).save(panel_dir /
                f'{Path(pth).stem}__{name}.png')

    plt.tight_layout()
    out_png = out_dir / 'gradcam_comparison.png'
    out_pdf = out_dir / 'gradcam_comparison.pdf'
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved -> {out_png}')
    print(f'Saved -> {out_pdf}')
    print(f'Per-panel overlays saved to: {panel_dir}')


if __name__ == '__main__':
    main()
