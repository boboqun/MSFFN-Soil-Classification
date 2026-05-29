# Multi-Scale Feature Fusion Network (MSFFN) for Mobile Soil Texture Classification

Official implementation of the paper
**"A Lightweight Multi-Scale Feature Fusion Network with Knowledge Distillation for Mobile Soil Texture Classification"** (under review).

![Model Architecture](assets/model_architecture.png)

## Overview
Accurate soil-texture classification is a key building block of precision agriculture, but most state-of-the-art models are too heavy to run on a phone. **MSFFN** is a lightweight CNN that taps shallow, mid and deep features from a MobileNetV3-Small backbone in parallel, then refines them through a parameter-efficient fusion head. With a **mixed knowledge-distillation** strategy (soft-label KL + intermediate-feature MSE) from an EfficientNet-B4 teacher, the distilled student reaches 98.94% test accuracy with only 1.02 M parameters and runs at 45.3 FPS on a commercial Snapdragon-based smartphone.

![Grad-CAM Comparison](assets/gradcam_comparison.png)
*Grad-CAM overlays show that MSFFN+KD attends to intrinsic soil-particle textures rather than background or ruler artefacts.*

## Repository layout
```
open_source_repo/
├── prepare_dataset.py    # Sliding-window patch extraction with ruler filtering
├── run_experiment.py     # Models, knowledge-distillation training, evaluation
├── run.py                # Thin launcher that wires up environment variables
├── run_ablations.sh      # Ablation orchestration (branch, loss, hyperparameter)
├── watchdog.py           # Restart-on-failure wrapper around run.py
├── plot_results.py       # Generates training / ablation / stability plots
├── gradcam_visualize.py  # Generates Grad-CAM interpretability maps
├── export_tflite.py      # TFLite conversion + MediaPipe metadata injection
├── requirements.txt      # Python dependencies
├── LICENSE               # MIT
└── assets/               # README images and benchmark protocol PDF
```

## Setup
Python 3.9+ is required. Install the dependencies once:
```bash
pip install -r requirements.txt
```

## Dataset
There are two ways to obtain the patch-level dataset.

**Option A — use the released patches (recommended).**
The 92,050 ruler-filtered 224x224 RGB patches with the original-image-level
`train`/`val`/`test` split manifest are archived on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20404766.svg)](https://doi.org/10.5281/zenodo.20404766)

1. Download the dataset archive and extract it.
2. Place the resulting directory under the repository root so that the layout is:
   ```
   open_source_repo/dataset/{train,val,test}/{Loam,Sand,Clay}/*.jpg
   ```
3. Skip `prepare_dataset.py` and jump straight to training.

**Option B — start from your own raw photographs.**
1. Organise your high-resolution images under:
   ```
   open_source_repo/dataset_original/train/<class_name>/*.jpg
   open_source_repo/dataset_original/validation/<class_name>/*.jpg
   ```
   Use `Loam`, `Sand`, `Clay` as the class folder names, or extend
   `CANONICAL_CLASSES` in `prepare_dataset.py` to add your own.
2. Run the patch extractor (224×224 windows, stride 180, ruler filter):
   ```bash
   ORIG_ROOT=./dataset_original python prepare_dataset.py
   ```
   This populates `./dataset/{train,val,test}/` and writes a
   `split_manifest.json` for auditing the original-to-subset mapping.

## Training
Once `./dataset/` exists, launch the full pipeline (teacher → SOTA baselines → MSFFN-no-KD → MSFFN+KD → evaluation):
```bash
python run.py
```
The trainer is fully resumable. If a run crashes, simply re-launch — the
config-hash mechanism keeps stale checkpoints out, and partial-epoch
weights are restored automatically.

Common environment overrides:
```bash
EPOCHS=60 python run.py                         # longer schedule
SEED=43 RUN_TAG=seed43 python run.py            # second seed
SKIP_HEAVY_SOTA=1 python run.py                 # skip ResNet50 / EffNetV2-B0 / MobileNetV2
ABLATE_BRANCH=no_low python run.py              # branch-removal ablation
DISTILL_MODE=kl_only python run.py              # loss-decomposition ablation
```

To restart automatically on crashes:
```bash
python watchdog.py
```

## Ablation studies
After the main experiment completes, run the ablation studies reported in the
paper. The script reuses the trained teacher checkpoint and only trains the
student model for each variant.

```bash
bash run_ablations.sh AB    # (A) branch-removal + (B) loss-decomposition
bash run_ablations.sh C     # (C) hyperparameter sensitivity (OFAT)
bash run_ablations.sh ALL   # all three groups
```

Results are saved to `results/ablation_*/metrics.json`.

## Plots and Grad-CAM
After training finishes, generate the comparison plots:
```bash
python plot_results.py --run-tag seed42
```
Outputs land in `./plots/seed42/`.

To regenerate the Grad-CAM panels used in the paper:
```bash
python gradcam_visualize.py --run-tag seed42
```

## Mobile deployment (TFLite export)
After training, convert the best distilled student to a quantised TFLite
model with embedded MediaPipe metadata:
```bash
python export_tflite.py --run-tag seed42
```
The script writes `exported_models/seed42/msffn_soil_texture.tflite` which
can be loaded directly by the MediaPipe Tasks Vision SDK on Android.

> **Note on Apple Silicon:** The metadata-injection step requires the
> `tflite-support` library, which may fail to build on ARM64 Macs.
> If you encounter build errors, run this script on a Windows or Linux
> x86 machine instead. The TFLite conversion itself works everywhere.

## Latency benchmark protocol
The exact on-device benchmark protocol used to obtain the 22.10 ms / 45.3 FPS
numbers reported in the paper is provided as
[`assets/supplementary.pdf`](assets/supplementary.pdf).

## License
Released under the [MIT License](LICENSE).

## Citation
If you find this code or dataset useful, please cite our paper (the entry
below will be updated once the paper is accepted):
```bibtex
@article{msffn_soil_2026,
  title   = {A Lightweight Multi-Scale Feature Fusion Network with Knowledge Distillation for Mobile Soil Texture Classification},
  author  = {Boqun Li and Yuejiao Ji and Man Jiao and Xiaoyang Zhao and Fuming Xie and Xiaoqiang Zhang and Fangyan Xue and Hao Yang and Yueming Hu and Wei Chen},
  year    = {2026},
  note    = {Under review; details will be updated upon acceptance.}
}
```

