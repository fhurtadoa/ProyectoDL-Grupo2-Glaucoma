#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de Clasificación y Estadificación Multietapa de Glaucoma (3 Clases):
- Clase 0: Sano / Control (Normal)
- Clase 1: Glaucoma Prematuro / Incipiente / Sospechoso (Early Stage)
- Clase 2: Glaucoma Avanzado / Severo (Advanced Stage)

Este pipeline:
1. Carga la cohorte graduada (data/MOONDREAM_STAGED).
2. Entrena una ResNet-18 con cabeza de clasificación de 3 etapas y ponderación de clases.
3. Evalúa en el conjunto de prueba independiente (n=200) computando:
   - Matriz de confusión 3x3.
   - Sensibilidad, Especificidad y F1-score por estadío.
   - Tasa de error adyacente (Sano <-> Temprano o Temprano <-> Avanzado)
     vs. Tasa de error catastrófico (Sano <-> Avanzado).
   - ROC-AUC binarizado proyectado (P(Glaucoma) = P(Early) + P(Advanced)).
4. Perfila cohortes externas binarias (ACRIMA, RIM-ONE, EyePACS) para desvelar
   qué fracción de sus diagnósticos corresponden a daño incipiente vs. avanzado.
5. Exporta figuras diagnósticas y resultados cuantitativos.
"""

import os, glob, random, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Configuración y Semilla
# ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"[INFO] Dispositivo de cómputo seleccionado: {DEVICE}")

BASE_DIR = Path("/Users/fabian/GlaucomaDL")
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "modelos"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["Sano (Normal)", "Prematuro (Incipiente)", "Avanzado (Severo)"]
CLASS_LABELS = [0, 1, 2]

# ---------------------------------------------------------
# Carga de Datos de Entrenamiento y Prueba
# ---------------------------------------------------------
print("\n[INFO] 1. Indexando datos de MOONDREAM_STAGED...")

train_records = []
for p in sorted(glob.glob(str(BASE_DIR / "data/MOONDREAM_STAGED/train/normal/*.jpg"))):
    train_records.append({"filepath": p, "stage": "normal", "label": 0})
for p in sorted(glob.glob(str(BASE_DIR / "data/MOONDREAM_STAGED/train/early/*.jpg"))):
    train_records.append({"filepath": p, "stage": "early", "label": 1})
for p in sorted(glob.glob(str(BASE_DIR / "data/MOONDREAM_STAGED/train/advanced/*.jpg"))):
    train_records.append({"filepath": p, "stage": "advanced", "label": 2})

test_records = []
for p in sorted(glob.glob(str(BASE_DIR / "data/MOONDREAM_STAGED/test/normal/*.jpg"))):
    test_records.append({"filepath": p, "stage": "normal", "label": 0})
for p in sorted(glob.glob(str(BASE_DIR / "data/MOONDREAM_STAGED/test/early/*.jpg"))):
    test_records.append({"filepath": p, "stage": "early", "label": 1})
for p in sorted(glob.glob(str(BASE_DIR / "data/MOONDREAM_STAGED/test/advanced/*.jpg"))):
    test_records.append({"filepath": p, "stage": "advanced", "label": 2})

df_train_full = pd.DataFrame(train_records)
df_test = pd.DataFrame(test_records)

# Split estratificado para validación interna (15%)
df_train, df_val = train_test_split(
    df_train_full, test_size=0.15, stratify=df_train_full["label"], random_state=SEED
)

print(f"  - Total Entrenamiento: {len(df_train)} (Sano: {sum(df_train['label']==0)}, Incipiente: {sum(df_train['label']==1)}, Avanzado: {sum(df_train['label']==2)})")
print(f"  - Total Validación:    {len(df_val)} (Sano: {sum(df_val['label']==0)}, Incipiente: {sum(df_val['label']==1)}, Avanzado: {sum(df_val['label']==2)})")
print(f"  - Total Prueba Indep.: {len(df_test)} (Sano: {sum(df_test['label']==0)}, Incipiente: {sum(df_test['label']==1)}, Avanzado: {sum(df_test['label']==2)})")

# ---------------------------------------------------------
# Dataset y Transformaciones
# ---------------------------------------------------------
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

class StagedFundusDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = row["label"]
        return img, label, row["filepath"]

train_loader = DataLoader(StagedFundusDataset(df_train, train_transform), batch_size=32, shuffle=True)
val_loader = DataLoader(StagedFundusDataset(df_val, eval_transform), batch_size=32, shuffle=False)
test_loader = DataLoader(StagedFundusDataset(df_test, eval_transform), batch_size=32, shuffle=False)

# ---------------------------------------------------------
# Arquitectura y Función de Pérdida Ponderada
# ---------------------------------------------------------
print("\n[INFO] 2. Configurando arquitectura ResNet-18 para 3 clases...")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3) # 3 clases de salida: [z0, z1, z2]
model.to(DEVICE)

# Ponderación inversa de frecuencia para balancear clases
counts = [sum(df_train['label']==0), sum(df_train['label']==1), sum(df_train['label']==2)]
total_samples = len(df_train)
weights = [total_samples / (3.0 * c) for c in counts]
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
print(f"  - Ponderación de pérdida (Class Weights): Sano={weights[0]:.2f}, Incipiente={weights[1]:.2f}, Avanzado={weights[2]:.2f}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# ---------------------------------------------------------
# Ciclo de Entrenamiento
# ---------------------------------------------------------
print("\n[INFO] 3. Iniciando entrenamiento multietapa (15 épocas)...")
best_macro_f1 = 0.0
best_model_path = MODELS_DIR / "best_resnet18_multistage.pth"

history = []

for epoch in range(1, 16):
    model.train()
    running_loss = 0.0
    for imgs, labels, _ in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(labels)
    scheduler.step()
    train_loss = running_loss / len(df_train)

    # Validación
    model.eval()
    val_loss = 0.0
    val_preds, val_targets = [], []
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * len(labels)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_loss = val_loss / len(df_val)
    val_acc = accuracy_score(val_targets, val_preds)
    val_macro_f1 = f1_score(val_targets, val_preds, average="macro")

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_macro_f1": val_macro_f1
    })

    print(f"  Época {epoch:02d}/15 | Loss Train: {train_loss:.4f} | Loss Val: {val_loss:.4f} | Acc Val: {val_acc*100:.2f}% | Macro F1: {val_macro_f1:.4f}")

    if val_macro_f1 > best_macro_f1:
        best_macro_f1 = val_macro_f1
        torch.save(model.state_dict(), best_model_path)
        print(f"   --> Nuevo mejor modelo guardado (Macro F1 = {best_macro_f1:.4f})")

print(f"\n[INFO] Entrenamiento finalizado. Mejor checkpoint en: {best_model_path}")

# Cargar mejor modelo
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
model.eval()

# ---------------------------------------------------------
# Evaluación en Cohorte de Prueba Independiente (n=200)
# ---------------------------------------------------------
print("\n[INFO] 4. Evaluando en cohorte de prueba independiente (n=200)...")

test_preds = []
test_probs = []
test_targets = []
test_paths = []

with torch.no_grad():
    for imgs, labels, paths in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        test_preds.extend(preds.cpu().numpy())
        test_probs.extend(probs.cpu().numpy())
        test_targets.extend(labels.numpy())
        test_paths.extend(paths)

test_preds = np.array(test_preds)
test_probs = np.array(test_probs)
test_targets = np.array(test_targets)

# Matriz de confusión 3x3
cm = confusion_matrix(test_targets, test_preds, labels=[0, 1, 2])
print("\n--- MATRIZ DE CONFUSIÓN 3x3 ---")
print("                Pred: Sano | Pred: Prematuro | Pred: Avanzado")
print(f"Real: Sano       {cm[0,0]:9d} | {cm[0,1]:15d} | {cm[0,2]:14d}")
print(f"Real: Prematuro  {cm[1,0]:9d} | {cm[1,1]:15d} | {cm[1,2]:14d}")
print(f"Real: Avanzado   {cm[2,0]:9d} | {cm[2,1]:15d} | {cm[2,2]:14d}")

# Métricas por clase
acc_overall = accuracy_score(test_targets, test_preds)
macro_f1_test = f1_score(test_targets, test_preds, average="macro")
weighted_f1_test = f1_score(test_targets, test_preds, average="weighted")

# Sensibilidad y Especificidad por clase (One-vs-Rest)
class_metrics = {}
for c in range(3):
    y_true_c = (test_targets == c).astype(int)
    y_pred_c = (test_preds == c).astype(int)
    tp = np.sum((y_true_c == 1) & (y_pred_c == 1))
    fp = np.sum((y_true_c == 0) & (y_pred_c == 1))
    fn = np.sum((y_true_c == 1) & (y_pred_c == 0))
    tn = np.sum((y_true_c == 0) & (y_pred_c == 0))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_c = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
    roc_c = roc_auc_score(y_true_c, test_probs[:, c])
    class_metrics[CLASS_NAMES[c]] = {
        "sensibilidad": float(sens),
        "especificidad": float(spec),
        "precision": float(prec),
        "f1": float(f1_c),
        "roc_auc": float(roc_c),
        "total": int(np.sum(y_true_c))
    }

print("\n--- DESGLOSE MÉTRICO POR ESTADÍO ---")
for k, v in class_metrics.items():
    print(f"[{k}] (N={v['total']}) | Sens: {v['sensibilidad']*100:.2f}% | Espec: {v['especificidad']*100:.2f}% | F1: {v['f1']:.4f} | ROC-AUC: {v['roc_auc']:.4f}")

# Error adyacente vs. error catastrófico
# Errores adyacentes: |real - pred| == 1 (Sano <-> Temprano o Temprano <-> Avanzado)
# Errores catastróficos: |real - pred| == 2 (Sano <-> Avanzado)
total_errors = np.sum(test_targets != test_preds)
adjacent_errors = np.sum(np.abs(test_targets - test_preds) == 1)
catastrophic_errors = np.sum(np.abs(test_targets - test_preds) == 2)

catastrophic_rate_total = (catastrophic_errors / len(test_targets)) * 100.0
adjacent_rate_total = (adjacent_errors / len(test_targets)) * 100.0
catastrophic_pct_of_errors = (catastrophic_errors / total_errors * 100.0) if total_errors > 0 else 0.0
adjacent_pct_of_errors = (adjacent_errors / total_errors * 100.0) if total_errors > 0 else 0.0

print(f"\n--- AUDITORÍA DE ERRORES CLÍNICOS ---")
print(f"Total Muestras Test:         {len(test_targets)}")
print(f"Aciertos Exactos:            {len(test_targets) - total_errors} ({(1 - total_errors/len(test_targets))*100:.2f}%)")
print(f"Errores Adyacentes (|Δ|=1):  {adjacent_errors} ({adjacent_rate_total:.2f}% del total, {adjacent_pct_of_errors:.2f}% de los fallos)")
print(f"Errores Catastróficos (|Δ|=2):{catastrophic_errors} ({catastrophic_rate_total:.2f}% del total, {catastrophic_pct_of_errors:.2f}% de los fallos)")

# Proyección a Tamizaje Binario (Cualquier Glaucoma = Clase 1 + Clase 2)
p_any_glaucoma = test_probs[:, 1] + test_probs[:, 2]
y_any_glaucoma = (test_targets > 0).astype(int)
binary_auc = roc_auc_score(y_any_glaucoma, p_any_glaucoma)
binary_pred = (p_any_glaucoma >= 0.5).astype(int)
binary_acc = accuracy_score(y_any_glaucoma, binary_pred)
binary_sens = recall_score(y_any_glaucoma, binary_pred)
binary_spec = recall_score(1 - y_any_glaucoma, 1 - binary_pred)

print(f"\n--- PROYECCIÓN A TAMIZAJE BINARIO (Sano vs. Cualquier Glaucoma) ---")
print(f"ROC-AUC Binario: {binary_auc:.4f} | Exactitud: {binary_acc*100:.2f}% | Sens: {binary_sens*100:.2f}% | Espec: {binary_spec*100:.2f}%")

# ---------------------------------------------------------
# Perfilado Clínico en Cohortes Externas Binarias
# ---------------------------------------------------------
print("\n[INFO] 5. Perfilando estadificación en datasets externos (ACRIMA, RIM-ONE, EyePACS)...")

def profile_external_dataset(df_records, dataset_name):
    ds = StagedFundusDataset(pd.DataFrame(df_records), eval_transform)
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    preds, probs, true_lbls = [], [], []
    with torch.no_grad():
        for imgs, lbls, _ in dl:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            pr = F.softmax(out, dim=1)
            preds.extend(torch.argmax(pr, dim=1).cpu().numpy())
            probs.extend(pr.cpu().numpy())
            true_lbls.extend(lbls.numpy())
    
    preds = np.array(preds)
    true_lbls = np.array(true_lbls)
    
    # Análisis para los casos que eran etiquetados como Glaucoma (true_lbl == 1)
    glauc_idx = (true_lbls == 1)
    glauc_preds = preds[glauc_idx]
    glauc_total = np.sum(glauc_idx)
    
    # Desglose de qué predijo el modelo de 3 etapas para los casos con glaucoma
    norm_count = np.sum(glauc_preds == 0)
    early_count = np.sum(glauc_preds == 1)
    adv_count = np.sum(glauc_preds == 2)
    
    # Análisis para los casos sanos (true_lbl == 0)
    norm_idx = (true_lbls == 0)
    norm_preds = preds[norm_idx]
    norm_total = np.sum(norm_idx)
    sano_as_sano = np.sum(norm_preds == 0)
    sano_as_early = np.sum(norm_preds == 1)
    sano_as_adv = np.sum(norm_preds == 2)
    
    res = {
        "dataset": dataset_name,
        "total_glaucoma": int(glauc_total),
        "glauc_pred_sano": int(norm_count),
        "glauc_pred_early": int(early_count),
        "glauc_pred_advanced": int(adv_count),
        "glauc_pct_early": float(early_count / glauc_total * 100.0) if glauc_total > 0 else 0.0,
        "glauc_pct_advanced": float(adv_count / glauc_total * 100.0) if glauc_total > 0 else 0.0,
        "glauc_pct_missed": float(norm_count / glauc_total * 100.0) if glauc_total > 0 else 0.0,
        "total_sano": int(norm_total),
        "sano_pred_sano": int(sano_as_sano),
        "sano_pred_early": int(sano_as_early),
        "sano_pred_advanced": int(sano_as_adv),
        "sano_specificity": float(sano_as_sano / norm_total * 100.0) if norm_total > 0 else 0.0
    }
    return res

# 1. ACRIMA
acrima_files = sorted(glob.glob(str(BASE_DIR / "data/Database/Images/*.jpg")))
acr_recs = []
for f in acrima_files:
    lbl = 1 if "_g_" in os.path.basename(f).lower() else 0
    acr_recs.append({"filepath": f, "label": lbl})
acr_profile = profile_external_dataset(acr_recs, "ACRIMA")

# 2. RIM-ONE DL
rim_df = pd.read_csv(BASE_DIR / "data/RIM_ONE_DL/metadata.csv")
rim_recs = []
for _, r in rim_df.iterrows():
    rim_recs.append({"filepath": r["filepath"], "label": int(r["label"])})
rim_profile = profile_external_dataset(rim_recs, "RIM-ONE DL")

# 3. EyePACS Real
eyepacs_recs = []
for p in sorted(glob.glob(str(BASE_DIR / "data/EYEPACS_REAL/normal/*.png"))):
    eyepacs_recs.append({"filepath": p, "label": 0})
for p in sorted(glob.glob(str(BASE_DIR / "data/EYEPACS_REAL/glaucoma/*.png"))):
    eyepacs_recs.append({"filepath": p, "label": 1})
eyepacs_profile = profile_external_dataset(eyepacs_recs, "EyePACS Real")

print(f"\n--- PERFILADO EXTERNO DE GLAUCOMA ---")
for prof in [acr_profile, rim_profile, eyepacs_profile]:
    print(f"[{prof['dataset']}] Total Glaucoma={prof['total_glaucoma']} -> Avanzado: {prof['glauc_pred_advanced']} ({prof['glauc_pct_advanced']:.1f}%), Prematuro: {prof['glauc_pred_early']} ({prof['glauc_pct_early']:.1f}%), Omitido (Sano): {prof['glauc_pred_sano']} ({prof['glauc_pct_missed']:.1f}%) | Espec Sano: {prof['sano_specificity']:.1f}%")

# ---------------------------------------------------------
# Generación de Figuras de Calidad Publicación
# ---------------------------------------------------------
print("\n[INFO] 6. Generando figuras de diagnóstico...")

# FIGURA 16: Matriz de Confusión 3x3 y Estadísticas de Error
fig16, ax = plt.subplots(figsize=(8, 7), dpi=300)
cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

for i in range(3):
    for j in range(3):
        val = cm[i, j]
        pct = (val / np.sum(cm[i, :])) * 100.0
        color = "white" if val > np.max(cm)/2 else "black"
        ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                color=color, fontsize=12, fontweight="bold")

fig16.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(["Sano\n(Normal)", "Prematuro\n(Incipiente)", "Avanzado\n(Severo)"], fontsize=11, fontweight="semibold")
ax.set_yticklabels(["Sano\n(Normal)", "Prematuro\n(Incipiente)", "Avanzado\n(Severo)"], fontsize=11, fontweight="semibold")
ax.set_xlabel("Predicción del Modelo", fontsize=12, labelpad=10, fontweight="bold")
ax.set_ylabel("Condición Clínica Real (Gold Standard)", fontsize=12, labelpad=10, fontweight="bold")
ax.set_title("Matriz de Confusión Multietapa 3x3 (N = 200)\nEvaluación en Cohorte de Prueba Independiente", fontsize=13, pad=20, fontweight="bold")

# Añadir recuadro con resumen de auditoría
summary_text = (
    f"Exactitud Global: {acc_overall*100:.1f}%  |  Macro F1: {macro_f1_test:.3f}\n"
    f"Errores Adyacentes (|Δ|=1): {adjacent_errors} ({adjacent_rate_total:.1f}% total, {adjacent_pct_of_errors:.1f}% fallos)\n"
    f"Errores Catastróficos (|Δ|=2): {catastrophic_errors} ({catastrophic_rate_total:.1f}% total, {catastrophic_pct_of_errors:.1f}% fallos)"
)
plt.figtext(0.5, 0.02, summary_text, wrap=True, horizontalalignment='center',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#D0D7DE', linewidth=1.2))

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
fig16_path = FIGURES_DIR / "fig16_multistage_confusion_matrix_3x3.png"
fig16.savefig(fig16_path)
plt.close(fig16)
print(f"  - Guardado: {fig16_path}")

# FIGURA 18: Perfilado de Cohortes Externas Binarias
fig18, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

cohort_labels = ["ACRIMA\n(Canon CR-DGi)", "RIM-ONE DL\n(Kowa 3D)", "EyePACS Real\n(Tamizaje Abierto)"]
pct_advanced = [acr_profile["glauc_pct_advanced"], rim_profile["glauc_pct_advanced"], eyepacs_profile["glauc_pct_advanced"]]
pct_early = [acr_profile["glauc_pct_early"], rim_profile["glauc_pct_early"], eyepacs_profile["glauc_pct_early"]]
pct_missed = [acr_profile["glauc_pct_missed"], rim_profile["glauc_pct_missed"], eyepacs_profile["glauc_pct_missed"]]

y_pos = np.arange(len(cohort_labels))
bar_height = 0.45

p1 = ax.barh(y_pos, pct_advanced, bar_height, label="Glaucoma Avanzado (Severo)", color="#CF222E", alpha=0.9)
p2 = ax.barh(y_pos, pct_early, bar_height, left=pct_advanced, label="Glaucoma Prematuro (Incipiente)", color="#D29922", alpha=0.9)
p3 = ax.barh(y_pos, pct_missed, bar_height, left=np.array(pct_advanced)+np.array(pct_early), label="Omitido como Sano (Falso Negativo)", color="#8C959F", alpha=0.85)

for i in range(len(cohort_labels)):
    # Texto avanzado
    if pct_advanced[i] > 8:
        ax.text(pct_advanced[i]/2, y_pos[i], f"{pct_advanced[i]:.1f}%\n(n={int(pct_advanced[i]*[acr_profile, rim_profile, eyepacs_profile][i]['total_glaucoma']/100)})",
                ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    # Texto temprano
    if pct_early[i] > 8:
        ax.text(pct_advanced[i] + pct_early[i]/2, y_pos[i], f"{pct_early[i]:.1f}%\n(n={int(pct_early[i]*[acr_profile, rim_profile, eyepacs_profile][i]['total_glaucoma']/100)})",
                ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    # Texto omitido
    if pct_missed[i] > 5:
        ax.text(pct_advanced[i] + pct_early[i] + pct_missed[i]/2, y_pos[i], f"{pct_missed[i]:.1f}%",
                ha="center", va="center", color="black", fontweight="bold", fontsize=8)

ax.set_yticks(y_pos)
ax.set_yticklabels(cohort_labels, fontsize=11, fontweight="semibold")
ax.set_xlabel("Porcentaje de Casos Etiquetados con Glaucoma (%)", fontsize=11, fontweight="bold")
ax.set_xlim(0, 105)
ax.set_title("Perfilado de Estadío Clínico en Datasets Binarios Externos\n¿Qué tipo de glaucoma contenían realmente las bases de datos públicas?", fontsize=12, fontweight="bold", pad=15)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True, fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig18_path = FIGURES_DIR / "fig18_multistage_external_cohort_profiling.png"
fig18.savefig(fig18_path)
plt.close(fig18)
print(f"  - Guardado: {fig18_path}")

# ---------------------------------------------------------
# Exportación de Resultados Cuantitativos a JSON
# ---------------------------------------------------------
metrics_output = {
    "test_set_size": len(test_targets),
    "accuracy_global": float(acc_overall),
    "macro_f1": float(macro_f1_test),
    "weighted_f1": float(weighted_f1_test),
    "confusion_matrix": cm.tolist(),
    "class_metrics": class_metrics,
    "adjacent_errors": int(adjacent_errors),
    "adjacent_error_rate_pct": float(adjacent_rate_total),
    "adjacent_pct_of_all_errors": float(adjacent_pct_of_errors),
    "catastrophic_errors": int(catastrophic_errors),
    "catastrophic_error_rate_pct": float(catastrophic_rate_total),
    "catastrophic_pct_of_all_errors": float(catastrophic_pct_of_errors),
    "binary_projection": {
        "roc_auc": float(binary_auc),
        "accuracy": float(binary_acc),
        "sensibilidad": float(binary_sens),
        "especificidad": float(binary_spec)
    },
    "external_profiling": {
        "ACRIMA": acr_profile,
        "RIM_ONE": rim_profile,
        "EyePACS": eyepacs_profile
    }
}

metrics_json_path = RESULTS_DIR / "multistage_metrics.json"
with open(metrics_json_path, "w", encoding="utf-8") as f:
    json.dump(metrics_output, f, indent=4)
print(f"\n[INFO] Métricas consolidadas exportadas a: {metrics_json_path}")
print("[INFO] Pipeline multietapa ejecutado exitosamente.")

