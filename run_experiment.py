#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_experiment.py
=================
Trains the teacher, the lightweight SOTA baselines, MSFFN (no distillation),
and the distilled MSFFN student in a single pipeline, with checkpoint resume
and an image-level-isolated train/val/test split.

Usage:
    python run.py                    # via thin wrapper (recommended)
    python run_experiment.py         # direct invocation; requires TRAIN_PATH,
                                     # VAL_PATH (and optionally TEST_PATH)
                                     # environment variables.
Environment variables:
    EPOCHS, SEED, BATCH_SIZE, RUN_TAG          training schedule
    TRAIN_PATH, VAL_PATH, TEST_PATH            data directories
    SKIP_DISTILL=1                             skip the distilled student
    SKIP_HEAVY_SOTA=1                          skip ResNet50 / EffNetV2-B0 / MobileNetV2
    ABLATE_BRANCH=no_low|no_mid|no_high        branch-removal ablation
    DISTILL_MODE=full|kl_only|feat_only        loss-decomposition ablation
    ALPHA, BETA, TEMPERATURE                   KD hyper-parameters
    RESUME_OK=0                                force fresh training
    ONLY_TRAIN=<comma-separated names>         restrict the run to a subset
"""

import os, sys, json, time, shutil
import numpy as np

class TeeLogger:
    """Duplicate stdout/stderr to both the terminal and a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'a', encoding='utf-8', buffering=1)
        self.line_buffer = ""

    def write(self, message):
        self.terminal.write(message)
        for char in message:
            if char == '\r':
                self.line_buffer = ""
            elif char == '\n':
                if self.line_buffer.strip():
                    self.log_file.write(self.line_buffer + '\n')
                self.line_buffer = ""
            else:
                self.line_buffer += char

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def fileno(self):
        return self.terminal.fileno()
import tensorflow as tf

assert tf.__version__.startswith('2'), f"TensorFlow 2.x is required, got: {tf.__version__}"

from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.applications import EfficientNetB4, MobileNetV3Small, ResNet50, MobileNetV2
from tensorflow.keras.applications import efficientnet_v2
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.resnet   import preprocess_input as res_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_preprocess
from tensorflow.keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau,
                                        EarlyStopping, Callback)

EPOCH_STATE_PATH = None   # filled in after CKPT_DIR is created

def load_epoch_state():
    if EPOCH_STATE_PATH and os.path.exists(EPOCH_STATE_PATH):
        with open(EPOCH_STATE_PATH) as f:
            return json.load(f)
    return {}

def save_epoch_state(state):
    with open(EPOCH_STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)

class EpochSaverCallback(Callback):
    """Save per-epoch weights and update epoch_state.json so that training
    can be resumed from the latest completed epoch."""
    def __init__(self, model_name, ckpt_dir):
        super().__init__()
        self.model_name = model_name
        self.ckpt_dir   = ckpt_dir

    def on_epoch_end(self, epoch, logs=None):
        epoch_1based = epoch + 1
        path = os.path.join(self.ckpt_dir,
                            f'{self.model_name}_epoch_{epoch_1based}.weights.h5')
        self.model.save_weights(path)
        try:
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        except Exception:
            current_lr = None
        state = load_epoch_state()
        state[self.model_name] = {
            'last_epoch': epoch_1based,
            'val_acc':    float(logs.get('val_accuracy', 0)) if logs else 0,
            'lr':         current_lr,
        }
        save_epoch_state(state)
        prev = os.path.join(self.ckpt_dir,
                            f'{self.model_name}_epoch_{epoch_1based - 1}.weights.h5')
        if os.path.exists(prev):
            os.remove(prev)
        print(f"  💾 Epoch {epoch_1based} weights saved (lr={current_lr})")

import time
class TimestampProgressCallback(Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.current_epoch = 0
        self.start_time = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.start_time = time.time()
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] >>> Epoch {self.current_epoch}/{self.epochs} Started")

    def on_batch_end(self, batch, logs=None):
        # 3813 batches per epoch -> output every 500 batches (~13% progress) to prevent log explosion in IDE
        if batch > 0 and batch % 500 == 0:
            loss = logs.get('loss', logs.get('total_loss', 0)) if logs else 0
            acc = logs.get('accuracy', 0) if logs else 0
            sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] Epoch {self.current_epoch}/{self.epochs} | Batch {batch}/3813 | loss: {loss:.4f} - acc: {acc:.4f}\n")
            sys.stdout.flush()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        val_loss = logs.get('val_loss', 0) if logs else 0
        val_acc = logs.get('val_accuracy', 0) if logs else 0
        loss = logs.get('loss', logs.get('total_loss', 0)) if logs else 0
        acc = logs.get('accuracy', 0) if logs else 0
        sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] Epoch {self.current_epoch}/{self.epochs} DONE | loss: {loss:.4f} - acc: {acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} | {elapsed:.1f}s\n")
        sys.stdout.flush()

def resume_fit(model, name, train_ds, val_ds, epochs, callbacks_list, ckpt_dir):
    """model.fit() with checkpoint resume.

    If epoch_state.json records a last_epoch for ``name``, training resumes
    from that point; otherwise it starts from epoch 0.
    """
    state       = load_epoch_state()
    model_state = state.get(name, {})
    last_epoch  = model_state.get('last_epoch', 0)   # epochs already completed

    if last_epoch > 0:
        tmp_ckpt  = os.path.join(ckpt_dir, f'{name}_epoch_{last_epoch}.weights.h5')
        best_ckpt = os.path.join(ckpt_dir, f'{name}_best.weights.h5')
        ckpt_to_load = tmp_ckpt if os.path.exists(tmp_ckpt) else (
                       best_ckpt if os.path.exists(best_ckpt) else None)
        if ckpt_to_load:
            try:
                model.load_weights(ckpt_to_load)
                print(f"  ↩️  Resuming from Epoch {last_epoch} checkpoint ({os.path.basename(ckpt_to_load)})")
            except Exception as e:
                print(f"  ⚠️  Failed to load checkpoint, training from scratch: {e}")
                last_epoch = 0
        else:
            print(f"  ⚠️  Cannot find Epoch {last_epoch} checkpoint, training from scratch")
            last_epoch = 0

        saved_lr = model_state.get('lr')
        if saved_lr is not None and last_epoch > 0:
            try:
                tf.keras.backend.set_value(model.optimizer.lr, saved_lr)
                print(f"  🔧 Learning rate restored to {saved_lr:.2e}")
            except Exception as e:
                print(f"  ⚠️  Failed to restore learning rate: {e}")

    if last_epoch >= epochs:
        print(f"  ✅ {name} has completed {epochs} epochs, no need to continue")
        class DummyHist: history = {}
        return DummyHist()

    callbacks_list = list(callbacks_list) + [EpochSaverCallback(name, ckpt_dir), TimestampProgressCallback(epochs)]

    return model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, initial_epoch=last_epoch,
        callbacks=callbacks_list, verbose=0)
from sklearn.metrics import confusion_matrix

EPOCHS       = int(os.environ.get('EPOCHS', 30))
SKIP_DISTILL = os.environ.get('SKIP_DISTILL', '0') == '1'
SEED         = int(os.environ.get('SEED', 42))
RUN_TAG      = os.environ.get('RUN_TAG', f'seed{SEED}')
BATCH_SIZE   = int(os.environ.get('BATCH_SIZE', 128))
IMG_SIZE     = 224

import random as _random_mod
_random_mod.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

ABLATE_BRANCH = os.environ.get('ABLATE_BRANCH', 'none')   # none | no_low | no_mid | no_high
DISTILL_MODE  = os.environ.get('DISTILL_MODE', 'full')    # full | kl_only | feat_only
ALPHA       = float(os.environ.get('ALPHA',  0.5))
BETA        = float(os.environ.get('BETA',   0.2))
TEMPERATURE = float(os.environ.get('TEMPERATURE', 5))

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR    = os.path.join(SCRIPT_DIR, 'checkpoints', RUN_TAG)
RESULT_DIR  = os.path.join(SCRIPT_DIR, 'results',     RUN_TAG)
PLOT_DIR    = os.path.join(SCRIPT_DIR, 'plots',       RUN_TAG)
LOG_DIR     = os.path.join(SCRIPT_DIR, 'logs')             # shared log root
for d in [CKPT_DIR, RESULT_DIR, PLOT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

EPOCH_STATE_PATH = os.path.join(CKPT_DIR, 'epoch_state.json')


import hashlib as _hashlib
def _compute_config_hash():
    """Hash every hyper-parameter that affects training.

    The hash is stored in ``epoch_state.json`` so that a resumed run
    automatically discards stale checkpoints whenever the configuration
    changes.
    """
    payload = {
        'seed':           SEED,
        'epochs':         EPOCHS,
        'batch_size':     BATCH_SIZE,
        'img_size':       IMG_SIZE,
        'ablate_branch':  ABLATE_BRANCH,
        'distill_mode':   DISTILL_MODE,
        'alpha':          ALPHA,
        'beta':           BETA,
        'temperature':    TEMPERATURE,
        'train_path':     TRAIN_PATH if 'TRAIN_PATH' in dir() else os.environ.get('TRAIN_PATH', ''),
        'val_path':       VAL_PATH   if 'VAL_PATH'   in dir() else os.environ.get('VAL_PATH', ''),
        'test_path':      TEST_PATH  if 'TEST_PATH'  in dir() else os.environ.get('TEST_PATH', ''),
    }
    blob = json.dumps(payload, sort_keys=True).encode('utf-8')
    return _hashlib.sha256(blob).hexdigest()[:16], payload


def _validate_or_clean_resume(force_fresh: bool = False):
    """Compare the saved config hash with the current one.

    - missing  : write the new hash (first run)
    - match    : keep checkpoints, resume normally
    - mismatch or ``force_fresh``: warn and wipe CKPT_DIR (weights and
      stale metrics) so that an incompatible old run does not pollute
      the new one.
    """
    cur_hash, cur_payload = _compute_config_hash()
    saved_hash = None
    if os.path.exists(EPOCH_STATE_PATH):
        try:
            with open(EPOCH_STATE_PATH) as f:
                _st = json.load(f)
            saved_hash = _st.get('config_hash')
        except Exception:
            saved_hash = None

    need_clean = force_fresh or (saved_hash is not None and saved_hash != cur_hash)
    if need_clean:
        print("\n" + "=" * 60)
        if force_fresh:
            print("⚠️  RESUME_OK=0: Forcing discard of all existing checkpoints, starting from scratch.")
        else:
            print("⚠️  Config hash mismatch (saved=%s, current=%s)" % (saved_hash, cur_hash))
            print("    Hyperparameters differ from old checkpoint, cleaning CKPT_DIR to prevent pollution.")
        print(f"    Cleaning directory: {CKPT_DIR}")
        for fn in os.listdir(CKPT_DIR):
            fp = os.path.join(CKPT_DIR, fn)
            if os.path.isfile(fp):
                os.remove(fp)
        _stale_metrics = os.path.join(RESULT_DIR, 'metrics.json')
        if os.path.exists(_stale_metrics):
            os.remove(_stale_metrics)
            print(f"    Cleaned: {_stale_metrics}")
        print("    Cleanup complete." )
        print("=" * 60 + "\n")

    state = {}
    if os.path.exists(EPOCH_STATE_PATH):
        try:
            with open(EPOCH_STATE_PATH) as f:
                state = json.load(f)
        except Exception:
            state = {}
    state['config_hash']    = cur_hash
    state['config_payload'] = cur_payload
    with open(EPOCH_STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)
    return cur_hash, saved_hash


_RESUME_OK = os.environ.get('RESUME_OK', '1') == '1'

_only_raw = os.environ.get('ONLY_TRAIN', '').strip()
ONLY_TRAIN = set(s.strip() for s in _only_raw.split(',') if s.strip()) if _only_raw else None
def _should_train(model_name: str) -> bool:
    """Return True if ``model_name`` should be trained in this run."""
    if ONLY_TRAIN is None:
        return True
    return model_name in ONLY_TRAIN

_log_filename = time.strftime(f"run_{RUN_TAG}_%Y%m%d_%H%M%S.log")
_log_path     = os.path.join(LOG_DIR, _log_filename)
sys.stdout    = TeeLogger(_log_path)
sys.stderr    = sys.stdout
print(f"Log file: {_log_path}")
print(f"Live view: tail -f {_log_path}\n")

JSON_PATH = os.path.join(RESULT_DIR, 'metrics.json')

DEFAULT_DATA_ROOT = os.environ.get('DEFAULT_DATA_ROOT', './dataset')
TRAIN_PATH = os.environ.get('TRAIN_PATH', os.path.join(DEFAULT_DATA_ROOT, 'train'))
VAL_PATH   = os.environ.get('VAL_PATH',   os.path.join(DEFAULT_DATA_ROOT, 'validation'))
TEST_PATH  = os.environ.get('TEST_PATH',  '')
HAS_INDEPENDENT_TEST = bool(TEST_PATH) and os.path.isdir(TEST_PATH)

_cur_hash, _saved_hash = _validate_or_clean_resume(force_fresh=not _RESUME_OK)
print(f"[resume] config hash = {_cur_hash}  (RESUME_OK={int(_RESUME_OK)}, "
      f"prev_hash={_saved_hash})")

print("=" * 60)
print(f"experiment  |  EPOCHS={EPOCHS}  BATCH={BATCH_SIZE}  SEED={SEED}  RUN_TAG={RUN_TAG}")
print(f"Train set: {TRAIN_PATH}")
print(f"Val set: {VAL_PATH}")
if HAS_INDEPENDENT_TEST:
    print(f"Test set: {TEST_PATH}  (Independent dir, image-level isolation)")
else:
    print(f"⚠️  No independent test dir detected, using val->val+test 50/50 split")
print(f"Ablation: branch={ABLATE_BRANCH}  distill={DISTILL_MODE}  α={ALPHA} β={BETA} T={TEMPERATURE}")
print("=" * 60)

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH, seed=SEED, image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, label_mode='categorical')

if HAS_INDEPENDENT_TEST:
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_PATH, seed=SEED, image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, label_mode='categorical')
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH, seed=SEED, image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, label_mode='categorical', shuffle=False)
else:
    val_ds_full = tf.keras.utils.image_dataset_from_directory(
        VAL_PATH, seed=SEED, image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, label_mode='categorical')
    half = len(val_ds_full) // 2
    val_ds  = val_ds_full.take(half)
    test_ds = val_ds_full.skip(half)

class_names = train_ds_raw.class_names
NUM_CLASSES = len(class_names)
print(f"Classes: {class_names}")

print("\nCounting patches (may take a few seconds on the first run)...")
n_val  = sum(1 for _ in val_ds.unbatch())
n_test = sum(1 for _ in test_ds.unbatch())
print(f"Val set: {n_val} patches  |  Test set: {n_test} patches")
for cls in sorted(os.listdir(TRAIN_PATH)):
    p = os.path.join(TRAIN_PATH, cls)
    if not os.path.isdir(p): continue
    n = len([f for f in os.listdir(p) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    print(f"  Train/{cls}: {n} patches")

AUTOTUNE = tf.data.AUTOTUNE
aug = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="aug")

train_ds = (train_ds_raw
            .cache()
            .map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
            .prefetch(AUTOTUNE))
val_ds  = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

def multiscale_backbone(input_shape=(IMG_SIZE, IMG_SIZE, 3), name="ms_backbone"):
    """MobileNetV3-Small with three feature taps (shallow / mid / deep)."""
    bb = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=input_shape)
    bb.trainable = True
    # On a 224x224 input the taps fall at strides 8 / 16 / 32 and correspond
    # to spatial resolutions 28x28x24 / 14x14x48 / 7x7x576 respectively.
    low  = bb.get_layer('expanded_conv_2/project').output
    mid  = bb.get_layer('expanded_conv_6/project').output
    high = bb.get_layer('Conv_1').output
    return models.Model(inputs=bb.input, outputs=[low, mid, high], name=name)

def multiscale_head(backbone_outputs, num_classes, name="ms_head",
                    use_low=True, use_mid=True, use_high=True):
    low, mid, high = backbone_outputs
    feats = []
    if use_low:
        feats.append(layers.AveragePooling2D((4, 4), strides=(4, 4), padding='same')(low))
    if use_mid:
        feats.append(layers.AveragePooling2D((2, 2), strides=(2, 2), padding='same')(mid))
    if use_high:
        feats.append(high)
    fused = layers.Concatenate(axis=-1)(feats) if len(feats) > 1 else feats[0]
    x = layers.Conv2D(128, 1, padding='same', use_bias=False)(fused)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.DepthwiseConv2D(3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(name=f"{name}_feat_gap")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    return layers.Dense(num_classes, activation='softmax', name=f"{name}_out")(x)

class Distiller(models.Model):
    """Wrap a (frozen) teacher and a trainable student.

    ``distill_mode`` selects the loss configuration:
        'full'      -> CE + KL + FeatMSE  (default, matches the paper)
        'kl_only'   -> CE + KL            (remove feature MSE)
        'feat_only' -> CE + FeatMSE       (remove KL soft-label)
    """

    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher

    def compile(self, optimizer, metrics, student_loss_fn,
                distillation_loss_fn, feature_loss_fn,
                alpha=0.5, beta=0.2, temperature=5, distill_mode='full'):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn      = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.feature_loss_fn      = feature_loss_fn
        self.alpha = alpha; self.beta = beta; self.temperature = temperature
        self.distill_mode = distill_mode

    def call(self, x, training=False):
        out = self.student(x, training=training)
        return out['prediction'] if isinstance(out, dict) else out

    def train_step(self, data):
        x, y = data
        t_out   = self.teacher(x, training=False)
        t_pred  = t_out['prediction']
        t_feat  = t_out['feature']
        with tf.GradientTape() as tape:
            s_out   = self.student(x, training=True)
            s_pred  = s_out['prediction']
            s_feat  = s_out['feature']
            ce_loss = self.student_loss_fn(y, s_pred)
            soft_t  = tf.nn.softmax(t_pred / self.temperature, axis=1)
            soft_s  = tf.nn.log_softmax(s_pred / self.temperature, axis=1)
            kd_loss = self.distillation_loss_fn(soft_t, soft_s) * self.temperature ** 2
            ft_loss = self.feature_loss_fn(t_feat, s_feat)
            if self.distill_mode == 'kl_only':
                loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
            elif self.distill_mode == 'feat_only':
                loss = ce_loss + self.beta * ft_loss
            else:
                loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss + self.beta * ft_loss
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.compiled_metrics.update_state(y, s_pred)
        res = {m.name: m.result() for m in self.metrics}
        res.update({'total_loss': loss, 'ce_loss': ce_loss,
                    'kd_loss': kd_loss, 'ft_loss': ft_loss})
        return res

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss   = self.student_loss_fn(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        res = {m.name: m.result() for m in self.metrics}
        res['loss'] = loss
        return res

def get_callbacks(model_name, monitor='val_accuracy', patience_es=15, patience_lr=5):
    """Return [ModelCheckpoint (best), ReduceLROnPlateau, EarlyStopping]."""
    ckpt = os.path.join(CKPT_DIR, f'{model_name}_best.weights.h5')
    return [
        ModelCheckpoint(ckpt, monitor=monitor, save_weights_only=True,
                        save_best_only=True, mode='max', verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                         patience=patience_lr, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor=monitor, patience=patience_es,
                      restore_best_weights=True, mode='max', verbose=1),
    ]

def calc_metrics(cm, names):
    acc = np.sum(np.diag(cm)) / np.sum(cm)
    precs, recs, f1s = [], [], []
    res = {}
    for i, n in enumerate(names):
        tp = cm[i,i]; fp = cm[:,i].sum()-tp; fn = cm[i,:].sum()-tp
        p = tp/(tp+fp) if tp+fp else 0
        r = tp/(tp+fn) if tp+fn else 0
        f = 2*p*r/(p+r) if p+r else 0
        precs.append(p); recs.append(r); f1s.append(f)
        res[n] = {'precision': p, 'recall': r, 'f1': f}
    res['accuracy']   = acc
    res['macro_avg']  = {'precision': np.mean(precs),
                         'recall':    np.mean(recs),
                         'f1':        np.mean(f1s)}
    res['confusion_matrix'] = cm.tolist()
    return res

def evaluate_model(model, test_ds, names):
    y_true, y_pred = [], []
    for imgs, labels in test_ds:
        preds = model.predict(imgs, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), 1))
        y_pred.extend(np.argmax(preds, 1))
    cm = confusion_matrix(y_true, y_pred)
    return calc_metrics(cm, names)

if os.path.exists(JSON_PATH):
    with open(JSON_PATH) as f:
        all_results = json.load(f)
else:
    all_results = {}

def save_result(name, metrics, history=None):
    entry = {'metrics': metrics}
    if history:
        entry['history'] = history
    all_results[name] = entry
    with open(JSON_PATH, 'w') as f:
        class Enc(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, (np.integer,)): return int(o)
                if isinstance(o, (np.floating,)): return float(o)
                if isinstance(o, np.ndarray): return o.tolist()
                return super().default(o)
        json.dump(all_results, f, indent=2, cls=Enc)
    print(f"  ✅ Saved {name}: acc={metrics['accuracy']:.4f}")


TEACHER_NAME = 'teacher'
if not _should_train(TEACHER_NAME):
    print(f"\n[1/7] {TEACHER_NAME} not in ONLY_TRAIN whitelist, skipping.")
elif TEACHER_NAME in all_results:
    print(f"\n[1/7] {TEACHER_NAME} already exists, skipping training.")
else:
    print(f"\n[1/7] Training teacher model ({TEACHER_NAME})...")
    base = EfficientNetB4(weights='imagenet', include_top=False,
                          input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = True
    teacher_train = models.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 3)),
        layers.Lambda(eff_preprocess),
        base,
        layers.GlobalAveragePooling2D(name='gap_teacher'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ], name=TEACHER_NAME)
    teacher_train.compile(
        optimizer=optimizers.legacy.Adam(1e-4),
        loss='categorical_crossentropy', metrics=['accuracy'])
    hist = resume_fit(teacher_train, TEACHER_NAME, train_ds, val_ds, EPOCHS,
                      get_callbacks(TEACHER_NAME), CKPT_DIR)
    teacher_train.load_weights(
        os.path.join(CKPT_DIR, f'{TEACHER_NAME}_best.weights.h5'))
    m = evaluate_model(teacher_train, test_ds, class_names)
    save_result(TEACHER_NAME, m, hist.history)

SKIP_HEAVY_SOTA = os.environ.get('SKIP_HEAVY_SOTA', '0') == '1'
sota_configs_all = [
    ('sota_mobilenetv3', MobileNetV3Small,   None,         1e-3),
    ('sota_resnet50',    ResNet50,            res_preprocess, 1e-4),
    ('sota_effnetv2_b0', efficientnet_v2.EfficientNetV2B0,
                         efficientnet_v2.preprocess_input, 1e-4),
    ('sota_mobilenetv2', MobileNetV2,         mv2_preprocess, 1e-3),
]
if SKIP_HEAVY_SOTA:
    sota_configs = [c for c in sota_configs_all if c[0] == 'sota_mobilenetv3']
    print(f"[SKIP_HEAVY_SOTA=1] Only training MobileNetV3-Small, skipping others")
else:
    sota_configs = sota_configs_all

for idx, (name, BaseClass, preprocess_fn, lr) in enumerate(sota_configs, 2):
    if not _should_train(name):
        print(f"\n[{idx}/7] {name} not in ONLY_TRAIN whitelist, skipping.")
        continue
    if name in all_results:
        print(f"\n[{idx}/7] {name} already exists, skipping.")
        continue
    print(f"\n[{idx}/7] Training {name}...")
    base = BaseClass(weights='imagenet', include_top=False,
                     input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = True
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x   = layers.Lambda(preprocess_fn)(inp) if preprocess_fn else inp
    x   = base(x)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    m_sota = models.Model(inp, out, name=name)
    m_sota.compile(optimizer=optimizers.legacy.Adam(lr),
                   loss='categorical_crossentropy', metrics=['accuracy'])
    hist = resume_fit(m_sota, name, train_ds, val_ds, EPOCHS,
                      get_callbacks(name), CKPT_DIR)
    m_sota.load_weights(os.path.join(CKPT_DIR, f'{name}_best.weights.h5'))
    metrics = evaluate_model(m_sota, test_ds, class_names)
    save_result(name, metrics, hist.history)

SCRATCH_NAME = 'student_scratch'
if not _should_train(SCRATCH_NAME):
    print(f"\n[6/7] {SCRATCH_NAME} not in ONLY_TRAIN whitelist, skipping.")
elif SCRATCH_NAME in all_results:
    print(f"\n[6/7] {SCRATCH_NAME} already exists, skipping.")
else:
    print(f"\n[6/7] Training {SCRATCH_NAME} (MSFFN, without distillation)...")
    _use_low  = (ABLATE_BRANCH != 'no_low')
    _use_mid  = (ABLATE_BRANCH != 'no_mid')
    _use_high = (ABLATE_BRANCH != 'no_high')
    print(f"  branches: use_low={_use_low} use_mid={_use_mid} use_high={_use_high}")
    bb   = multiscale_backbone()
    head = multiscale_head(bb.output, NUM_CLASSES, name='scratch_head',
                           use_low=_use_low, use_mid=_use_mid, use_high=_use_high)
    scratch = models.Model(bb.input, head, name=SCRATCH_NAME)
    scratch.compile(
        optimizer=optimizers.legacy.Adam(5e-4, clipnorm=1.0),
        loss='categorical_crossentropy', metrics=['accuracy'])
    hist = resume_fit(scratch, SCRATCH_NAME, train_ds, val_ds, EPOCHS,
                      get_callbacks(SCRATCH_NAME, patience_es=20, patience_lr=7), CKPT_DIR)
    scratch.load_weights(os.path.join(CKPT_DIR, f'{SCRATCH_NAME}_best.weights.h5'))
    metrics = evaluate_model(scratch, test_ds, class_names)
    save_result(SCRATCH_NAME, metrics, hist.history)

DISTILL_NAME = 'student_distilled'
if not _should_train(DISTILL_NAME):
    print(f"\n[7/7] {DISTILL_NAME} not in ONLY_TRAIN whitelist, skipping.")
elif DISTILL_NAME in all_results:
    print(f"\n[7/7] {DISTILL_NAME} already exists, skipping.")
elif SKIP_DISTILL:
    print(f"\n[7/7] SKIP_DISTILL=1, skipping distilled student.")
else:
    print(f"\n[7/7] Training {DISTILL_NAME} (MSFFN + KD)...")

    t_base = EfficientNetB4(weights=None, include_top=False,
                            input_shape=(IMG_SIZE, IMG_SIZE, 3))
    feat_layer = 'block6a_expand_activation'
    t_base_multi = models.Model(t_base.input,
                                [t_base.output,
                                 t_base.get_layer(feat_layer).output])
    inp_t = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x_t   = layers.Lambda(eff_preprocess)(inp_t)
    base_out, feat_out = t_base_multi(x_t)
    pred_t = layers.Dense(NUM_CLASSES, activation='softmax',
                          name='prediction')(layers.GlobalAveragePooling2D()(base_out))
    teacher_distill = models.Model(inp_t, {'prediction': pred_t, 'feature': feat_out},
                                   name='teacher_distill')
    teacher_distill.load_weights(
        os.path.join(CKPT_DIR, f'{TEACHER_NAME}_best.weights.h5'), by_name=True)
    teacher_distill.trainable = False

    _use_low  = (ABLATE_BRANCH != 'no_low')
    _use_mid  = (ABLATE_BRANCH != 'no_mid')
    _use_high = (ABLATE_BRANCH != 'no_high')
    print(f"  branches: use_low={_use_low} use_mid={_use_mid} use_high={_use_high}")
    s_bb  = multiscale_backbone(name='student_bb')
    t_ch  = teacher_distill.output['feature'].shape[-1]
    s_feat = layers.Conv2D(t_ch, 1, padding='same', name='feat_adapter')(s_bb.output[1])
    s_head = multiscale_head(s_bb.output, NUM_CLASSES, name='student_head',
                             use_low=_use_low, use_mid=_use_mid, use_high=_use_high)
    student = models.Model(s_bb.input,
                           {'prediction': s_head, 'feature': s_feat},
                           name='student_distill')

    distiller = Distiller(student=student, teacher=teacher_distill)
    distiller.compile(
        optimizer=optimizers.legacy.Adam(5e-4, clipnorm=1.0),
        metrics=['accuracy'],
        student_loss_fn=losses.CategoricalCrossentropy(),
        distillation_loss_fn=losses.KLDivergence(),
        feature_loss_fn=losses.MeanSquaredError(),
        alpha=ALPHA, beta=BETA, temperature=TEMPERATURE,
        distill_mode=DISTILL_MODE)

    hist = resume_fit(distiller, DISTILL_NAME, train_ds, val_ds, EPOCHS,
                      get_callbacks(DISTILL_NAME, patience_es=20, patience_lr=7), CKPT_DIR)

    ckpt_path = os.path.join(CKPT_DIR, f'{DISTILL_NAME}_best.weights.h5')
    distiller.load_weights(ckpt_path)

    infer_model = models.Model(distiller.student.input, distiller.student.output['prediction'], name='student_infer')

    metrics = evaluate_model(infer_model, test_ds, class_names)
    save_result(DISTILL_NAME, metrics, hist.history)


print("\n" + "=" * 60)
print("Experiment Results Summary")
print("=" * 60)
for name, entry in all_results.items():
    m = entry.get('metrics', {})
    ep = len(entry.get('history', {}).get('val_accuracy', []))
    print(f"  {name:30s}  acc={m.get('accuracy', 0):.4f}  "
          f"macro_f1={m.get('macro_avg',{}).get('f1',0):.4f}  "
          f"epochs_run={ep}")
print("Next step: run experiment/plot_results.py to generate plots")
