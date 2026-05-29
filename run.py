#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py
======
Thin launcher that points ``run_experiment.py`` at the image-level
isolated dataset under ``./dataset/`` and forwards a small set of
environment variables.

Usage:
    python run.py                       # use default config
    EPOCHS=60 SEED=42 python run.py     # override hyper-parameters
    RUN_TAG=my_exp python run.py        # write outputs under
                                        # results/my_exp/, plots/my_exp/, ...

All checkpoints, metrics, plots and logs are written next to this script,
inside ``checkpoints/<RUN_TAG>/``, ``results/<RUN_TAG>/``,
``plots/<RUN_TAG>/`` and ``logs/``.
"""

import os, sys, subprocess

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT  = os.path.join(SCRIPT_DIR, 'run_experiment.py')
DATASET_DIR   = os.path.join(SCRIPT_DIR, 'dataset')

# Verify dataset exists
for subset in ['train', 'val', 'test']:
    p = os.path.join(DATASET_DIR, subset)
    if not os.path.isdir(p):
        print(f"Missing dataset directory: {p}")
        print("Either download the released dataset from Zenodo and extract")
        print("it as ./dataset/, or run prepare_dataset.py on your own raw")
        print("images placed under ./dataset_original/.")
        sys.exit(1)

env = os.environ.copy()
env.update({
    'TRAIN_PATH': os.path.join(DATASET_DIR, 'train'),
    'VAL_PATH':   os.path.join(DATASET_DIR, 'val'),
    'TEST_PATH':  os.path.join(DATASET_DIR, 'test'),
    'EPOCHS':     os.environ.get('EPOCHS', '30'),
    'SEED':       os.environ.get('SEED', '42'),
    'RUN_TAG':    os.environ.get('RUN_TAG', 'seed42'),
    'BATCH_SIZE': os.environ.get('BATCH_SIZE', '128'),
    # Allow resuming so the watchdog can restart without losing checkpoints.
    # The config-hash mechanism in run_experiment.py already drops stale
    # weights automatically whenever hyper-parameters change.
    'RESUME_OK':  os.environ.get('RESUME_OK', '1'),
})

print("=" * 60)
print("MSFFN training launcher")
print("=" * 60)
print(f"Dataset:  {DATASET_DIR}")
print(f"Script:   {TRAIN_SCRIPT}")
print(f"RUN_TAG:  {env['RUN_TAG']}")
print(f"EPOCHS:   {env['EPOCHS']}")
print(f"SEED:     {env['SEED']}")
print(f"TRAIN:    {env['TRAIN_PATH']}")
print(f"VAL:      {env['VAL_PATH']}")
print(f"TEST:     {env['TEST_PATH']}")
print("=" * 60)

result = subprocess.run(
    [sys.executable, TRAIN_SCRIPT],
    env=env,
    cwd=SCRIPT_DIR,
)
sys.exit(result.returncode)
