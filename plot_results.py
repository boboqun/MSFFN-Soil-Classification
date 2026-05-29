#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_results.py
===============
Read ``results/<RUN_TAG>/metrics.json`` produced by ``run_experiment.py``
and render the following figures into ``plots/<RUN_TAG>/``:

    training_curves.png      validation accuracy & loss per epoch
    ablation_bar.png         baseline -> MSFFN -> MSFFN+KD -> Teacher
    stability_compare.png    distilled vs. from-scratch convergence

Usage:
    python plot_results.py                       # default RUN_TAG=seed42
    python plot_results.py --run-tag seed42
    RUN_TAG=seed42 python plot_results.py
"""

import argparse
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--run-tag',
                default=os.environ.get('RUN_TAG', 'seed42'),
                help='RUN_TAG sub-directory under ./results/ and ./plots/ '
                     '(default: seed42).')
args = ap.parse_args()
RUN_TAG = args.run_tag

JSON_PATH = os.path.join(SCRIPT_DIR, 'results', RUN_TAG, 'metrics.json')
PLOT_DIR  = os.path.join(SCRIPT_DIR, 'plots',   RUN_TAG)
os.makedirs(PLOT_DIR, exist_ok=True)

if not os.path.exists(JSON_PATH):
    print(f"Cannot find {JSON_PATH}, please run run_experiment.py first.")
    raise SystemExit(1)

with open(JSON_PATH) as f:
    data = json.load(f)

COLORS = {
    'teacher':          '#E07B39',
    'student_distilled':'#2E86AB',
    'student_scratch':  '#27AE60',
    'sota_mobilenetv3': '#C0392B',
    'sota_resnet50':    '#8E44AD',
    'sota_effnetv2_b0': '#2C3E50',
    'sota_mobilenetv2': '#7F8C8D',
}
LABELS = {
    'teacher':          'Teacher (EfficientNetB4)',
    'student_distilled':'MSFFN + Distillation (Proposed)',
    'student_scratch':  'MSFFN w/o Distillation',
    'sota_mobilenetv3': 'MobileNetV3-Small',
    'sota_resnet50':    'ResNet50',
    'sota_effnetv2_b0': 'EfficientNetV2-B0',
    'sota_mobilenetv2': 'MobileNetV2',
}

def hist(name):
    return data.get(name, {}).get('history', {})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training Dynamics on Soil Texture Dataset', fontsize=14, fontweight='bold')

ax_acc, ax_loss = axes

for name in LABELS:
    h = hist(name)
    va = h.get('val_accuracy', [])
    vl = h.get('val_loss', [])
    if not va:
        continue
    ep = range(1, len(va)+1)
    c  = COLORS.get(name, '#999')
    lb = LABELS.get(name, name)
    ax_acc.plot(ep, [v*100 for v in va], color=c, lw=2, marker='o', ms=3.5, label=lb)
    vl_clipped = [min(v, 5.0) for v in vl]
    ax_loss.plot(ep, vl_clipped, color=c, lw=2, marker='o', ms=3.5, label=lb)

for ax, title, ylabel in [
    (ax_acc,  'Validation Accuracy per Epoch', 'Accuracy (%)'),
    (ax_loss, 'Validation Loss (clipped at 5)', 'Loss'),
]:
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right' if 'Acc' in title else 'upper right')
    ax.grid(True, alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax_acc.set_ylim(20, 105)
fig.tight_layout()
out = os.path.join(PLOT_DIR, 'training_curves.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"✅ {out}")
plt.close(fig)

ablation_models = [
    ('sota_mobilenetv3', 'SOTA MobileNetV3\n(No MS)'),
    ('student_scratch',  'MSFFN w/o KD\n(MS Arch)'),
    ('student_distilled','MSFFN + KD\n(Full Method)'),
    ('teacher',          'Teacher\nEfficientNetB4'),
]
names_avail = [(k, lb) for k, lb in ablation_models if k in data]
accs = [data[k]['metrics'].get('accuracy', 0) * 100 for k, _ in names_avail]
labels = [lb for _, lb in names_avail]
bar_colors = ['#AED6F1', '#27AE60', '#2E86AB', '#E07B39'][:len(names_avail)]

fig2, ax2 = plt.subplots(figsize=(9, 5))
bars = ax2.bar(labels, accs, color=bar_colors[:len(accs)], width=0.5, edgecolor='white', lw=1.5)
for bar, acc in zip(bars, accs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

keys = [k for k, _ in names_avail]
if 'sota_mobilenetv3' in keys and 'student_scratch' in keys:
    i1 = keys.index('sota_mobilenetv3')
    i2 = keys.index('student_scratch')
    delta = accs[i2] - accs[i1]
    mid_x = (i1 + i2) / 2
    ax2.annotate('', xy=(i2, accs[i2]), xytext=(i1, accs[i1]),
                 arrowprops=dict(arrowstyle='->', color='#1A5276', lw=1.5))
    ax2.text(mid_x, (accs[i1]+accs[i2])/2 + 1,
             f'Arch.\n+{delta:.1f}%', ha='center', fontsize=9, color='#1A5276')

if 'student_scratch' in keys and 'student_distilled' in keys:
    i1 = keys.index('student_scratch')
    i2 = keys.index('student_distilled')
    delta = accs[i2] - accs[i1]
    mid_x = (i1 + i2) / 2
    ax2.annotate('', xy=(i2, accs[i2]), xytext=(i1, accs[i1]),
                 arrowprops=dict(arrowstyle='->', color='#C0392B', lw=2))
    ax2.text(mid_x, (accs[i1]+accs[i2])/2 + 1,
             f'Distillation\n+{delta:.1f}%', ha='center', fontsize=9,
             color='#C0392B', fontweight='bold')

ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
ax2.set_title('Ablation: Architecture vs. Distillation Contribution', fontsize=13, fontweight='bold')
ax2.set_ylim(max(0, min(accs) - 15), 108)
ax2.grid(axis='y', alpha=0.3, ls='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
fig2.tight_layout()
out2 = os.path.join(PLOT_DIR, 'ablation_bar.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"✅ {out2}")
plt.close(fig2)

pairs = [('teacher', 'student_scratch', 'student_distilled')]
h_t  = hist('teacher')
h_sc = hist('student_scratch')
h_di = hist('student_distilled')

if h_t.get('val_accuracy') or h_sc.get('val_accuracy') or h_di.get('val_accuracy'):
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig3.suptitle('Training Stability: Impact of Knowledge Distillation',
                  fontsize=13, fontweight='bold')

    for h, name, col in [
        (h_t,  'teacher',          '#E07B39'),
        (h_sc, 'student_scratch',  '#27AE60'),
        (h_di, 'student_distilled','#2E86AB'),
    ]:
        va = h.get('val_accuracy', [])
        vl = h.get('val_loss', [])
        if not va: continue
        ep = range(1, len(va)+1)
        ax3a.plot(ep, [v*100 for v in va], color=col, lw=2, marker='o', ms=3.5,
                  label=LABELS.get(name, name))
        ax3b.plot(ep, [min(v, 10.0) for v in vl], color=col, lw=2, marker='o', ms=3.5,
                  label=LABELS.get(name, name))

    for ax, title, ylabel, ylim in [
        (ax3a, 'Validation Accuracy',  'Accuracy (%)', (20, 105)),
        (ax3b, 'Validation Loss (clipped at 10)', 'Loss', (0, 10.5)),
    ]:
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(*ylim)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig3.tight_layout()
    out3 = os.path.join(PLOT_DIR, 'stability_compare.png')
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    print(f"✅ {out3}")
    plt.close(fig3)

print("\n── Final Results Summary ──────────────────────────")
print(f"{'Model':<30} {'Acc':>7} {'Macro F1':>9} {'Epochs':>7}")
print("-" * 58)
for name, entry in data.items():
    m  = entry.get('metrics', {})
    ep = len(entry.get('history', {}).get('val_accuracy', []))
    print(f"{name:<30} {m.get('accuracy',0)*100:>6.2f}%"
          f" {m.get('macro_avg',{}).get('f1',0)*100:>8.2f}%"
          f" {ep:>7}")
print(f"\nPlots saved to: {PLOT_DIR}/")
