#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_tflite.py
================
Exports the trained MSFFN model (Student) to a quantized TFLite format
and injects the necessary Metadata (Input/Output tensors, normalization 
parameters, and label map) required by the MediaPipe Tasks API.

⚠️ NOTE ON APPLE SILICON (M1/M2/M3) COMPATIBILITY:
The `tflite-support` library, which is required to inject metadata, may
fail to install on newer Macs due to missing arm64 wheels. If you encounter
build errors during `pip install tflite-support`, it is highly recommended
to run this specific export script on a Windows or Linux x86 machine.

Usage:
    python export_tflite.py --run-tag seed42
"""

import os
import argparse
from pathlib import Path
import tensorflow as tf

# Try importing tflite_support for metadata injection
try:
    from tflite_support import flatbuffers
    from tflite_support import metadata as _metadata
    from tflite_support import metadata_schema_py_generated as _metadata_fb
    HAS_TFLITE_SUPPORT = True
except ImportError:
    HAS_TFLITE_SUPPORT = False

# Import model builders from the training script
import run_experiment as exp

def create_tflite_model(weights_path, out_tflite_path):
    print(f"[1/3] Building MSFFN model and loading weights from {weights_path}...")
    
    # We build the student architecture directly
    # Note: Using the exact multiscale definitions from run_experiment.py
    bb = exp.multiscale_backbone(name='student_bb')
    head = exp.multiscale_head(bb.output, 3, name='student_head')
    model = tf.keras.models.Model(bb.input, head, name='student_distilled')
    
    # Load the best distillation weights
    model.load_weights(weights_path)
    print("      Weights loaded successfully.")

    print("[2/3] Converting to TFLite (with default optimizations)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    with open(out_tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"      Raw TFLite model saved to {out_tflite_path}.")
    return out_tflite_path


def inject_metadata(tflite_path, labels_txt_path):
    if not HAS_TFLITE_SUPPORT:
        print("\n[3/3] ⚠️ SKIPPED METADATA INJECTION!")
        print("      'tflite-support' is not installed. The model was converted to .tflite")
        print("      but it lacks the Metadata required for MediaPipe Android deployment.")
        print("      Please run this on a compatible machine (Windows/Linux x86) with:")
        print("      pip install tflite-support")
        return

    print("[3/3] Injecting MediaPipe-compatible Metadata...")
    
    # Create model info
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "MSFFN Soil Texture Classifier"
    model_meta.description = (
        "Identify soil texture (Clay, Loam, Sand) from smartphone images. "
        "Distilled lightweight MSFFN architecture."
    )
    model_meta.version = "v1.0"
    model_meta.author = "Boqun Li et al."
    model_meta.license = "MIT"

    # Input info
    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "image"
    input_meta.description = (
        "Input image to be classified. The expected image is 224 x 224, with "
        "three channels (red, green, and blue) per pixel. Each value in the "
        "tensor is a single byte between 0 and 255."
    )
    input_meta.content = _metadata_fb.ContentT()
    input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
    input_meta.content.contentProperties.colorSpace = (
        _metadata_fb.ColorSpaceType.RGB
    )
    input_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.ImageProperties
    )
    input_normalization = _metadata_fb.ProcessUnitT()
    input_normalization.optionsType = (
        _metadata_fb.ProcessUnitOptions.NormalizationOptions
    )
    input_normalization.options = _metadata_fb.NormalizationOptionsT()
    # In run_experiment.py, images are simply divided by 255: x / 255.0
    # MediaPipe Formula: (input - mean) / std
    input_normalization.options.mean = [0.0, 0.0, 0.0]
    input_normalization.options.std = [255.0, 255.0, 255.0]
    input_meta.processUnits = [input_normalization]
    input_stats = _metadata_fb.StatsT()
    input_stats.max = [255.0]
    input_stats.min = [0.0]
    input_meta.stats = input_stats

    # Output info
    output_meta = _metadata_fb.TensorMetadataT()
    output_meta.name = "probability"
    output_meta.description = "Probabilities of the 3 soil texture classes."
    output_meta.content = _metadata_fb.ContentT()
    output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
    output_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties
    )
    output_stats = _metadata_fb.StatsT()
    output_stats.max = [1.0]
    output_stats.min = [0.0]
    output_meta.stats = output_stats

    # Associate label file
    label_file = _metadata_fb.AssociatedFileT()
    label_file.name = os.path.basename(labels_txt_path)
    label_file.description = "Labels for objects that the model can recognize."
    label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
    output_meta.associatedFiles = [label_file]

    # Combine subgraph
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [output_meta]
    model_meta.subgraphMetadata = [subgraph]

    # Write metadata into model
    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()

    populator = _metadata.MetadataPopulator.with_model_file(tflite_path)
    populator.load_metadata_buffer(metadata_buf)
    populator.load_associated_files([labels_txt_path])
    populator.populate()
    print(f"      Successfully injected Metadata and {label_file.name} into the TFLite model!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-tag', default='seed42', help='Which run to export')
    args = parser.parse_args()

    # Paths
    ckpt_dir = Path(__file__).parent / 'checkpoints' / args.run_tag
    weights_path = ckpt_dir / 'student_distilled_best.weights.h5'
    
    if not weights_path.exists():
        print(f"❌ Error: Weights not found at {weights_path}")
        print("Please train the model first by running `python run.py`")
        return

    out_dir = Path(__file__).parent / 'exported_models' / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tflite_path = out_dir / 'msffn_soil_texture.tflite'
    labels_txt_path = out_dir / 'labels.txt'

    # Create standard labels file
    # Ensure they match the alphabetical order outputted by tf.data (Clay, Loam, Sand)
    with open(labels_txt_path, 'w') as f:
        f.write("Clay\nLoam\nSand\n")

    # Disable GPU to avoid crashes during conversion on some platforms
    tf.config.set_visible_devices([], 'GPU')

    create_tflite_model(str(weights_path), str(out_tflite_path))
    inject_metadata(str(out_tflite_path), str(labels_txt_path))
    
    print("\n✅ Export Process Finished!")
    print(f"The final TFLite model is available at: {out_tflite_path}")

if __name__ == '__main__':
    main()
