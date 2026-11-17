import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('results/1000_epochs_stress_experiment_metrics.json') as f:
    d = json.load(f)

fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=300)
cohorts = ['ACRIMA', 'RIM-ONE DL', 'EyePACS', 'Moondream']
classes = ['Sano', 'Glaucoma']

for idx, c in enumerate(cohorts):
    # Epoch 15
    cm15 = np.array(d['eval_comparison']['epoch_15']['cohorts'][c]['cm'])
    acc15 = d['eval_comparison']['epoch_15']['cohorts'][c]['acc'] * 100
    auc15 = d['eval_comparison']['epoch_15']['cohorts'][c]['auc']
    ax15 = axes[0, idx]
    ax15.imshow(cm15, cmap='Blues', interpolation='nearest')
    ax15.set_title(f"{c} - Época 15 (Convencional)\nAcc: {acc15:.1f}% | AUC: {auc15:.3f}", fontsize=9.5, fontweight='bold')
    ax15.set_xticks([0, 1])
    ax15.set_xticklabels(classes, fontsize=8.5, fontweight='bold')
    ax15.set_yticks([0, 1])
    ax15.set_yticklabels(classes, fontsize=8.5, fontweight='bold')
    if idx == 0:
        ax15.set_ylabel('Etiqueta Real (Época 15)', fontweight='bold')
    thresh15 = cm15.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm15[i, j]
            ax15.text(j, i, str(val), ha='center', va='center', color='white' if val > thresh15 else 'black', fontweight='bold', fontsize=11)

    # Epoch 1000
    cm1000 = np.array(d['eval_comparison']['epoch_1000']['cohorts'][c]['cm'])
    acc1000 = d['eval_comparison']['epoch_1000']['cohorts'][c]['acc'] * 100
    auc1000 = d['eval_comparison']['epoch_1000']['cohorts'][c]['auc']
    ax1000 = axes[1, idx]
    ax1000.imshow(cm1000, cmap='Greens', interpolation='nearest')
    ax1000.set_title(f"{c} - Época 1000 (Régimen Asintótico)\nAcc: {acc1000:.1f}% | AUC: {auc1000:.3f}", fontsize=9.5, fontweight='bold')
    ax1000.set_xticks([0, 1])
    ax1000.set_xticklabels(classes, fontsize=8.5, fontweight='bold')
    ax1000.set_yticks([0, 1])
    ax1000.set_yticklabels(classes, fontsize=8.5, fontweight='bold')
    ax1000.set_xlabel('Predicción del Modelo', fontweight='bold')
    if idx == 0:
        ax1000.set_ylabel('Etiqueta Real (Época 1000)', fontweight='bold')
    thresh1000 = cm1000.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm1000[i, j]
            ax1000.text(j, i, str(val), ha='center', va='center', color='white' if val > thresh1000 else 'black', fontweight='bold', fontsize=11)

plt.suptitle('Comparativa Diagnóstica de Matrices de Confusión por Sensor Clínico: Época 15 vs. Época 1000', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figures/fig28_1000_epochs_confusion_matrices.png', bbox_inches='tight')
plt.close()
print('Figure 28 generated successfully!')
