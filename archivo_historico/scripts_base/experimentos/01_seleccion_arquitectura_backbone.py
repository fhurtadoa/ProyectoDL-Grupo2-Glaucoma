#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estudio Comparativo y Justificación de Arquitecturas Backbone para Transfer Learning:
Compara de manera empírica y cuantitativa:
1. ResNet-18
2. ResNet-50
3. EfficientNet-B0
4. MobileNet-V2

Todas inicializadas con pesos preentrenados en ImageNet-1K y adaptadas
para la clasificación binaria de glaucoma sobre imágenes de fondo de ojo.
Mide:
- Número de parámetros y tamaño del modelo.
- Latencia de inferencia (ms/batch).
- Convergencia y pérdida (Train & Val Loss).
- Desempeño métrico: Exactitud, Sensibilidad, Especificidad, F1-Score y ROC-AUC.
- Tendencia al sobreajuste (Generalization Gap).
"""

import os, sys, time, json, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"[INFO] Dispositivo seleccionado: {DEVICE}", flush=True)

BASE_DIR = Path("/Users/fabian/GlaucomaDL")
DATA_DIR = BASE_DIR / "data" / "Database" / "Images"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# Preparación de Datos (ACRIMA Cohort N=705)
# -----------------------------------------------------------------------------
import glob
files = sorted(glob.glob(str(DATA_DIR / "*.jpg")))
records = []
for f in files:
    fname = os.path.basename(f).lower()
    lbl = 1 if "_g_" in fname else 0
    records.append({"filepath": f, "label": lbl})

df = pd.DataFrame(records)
df_train, df_test = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=SEED)
df_train, df_val = train_test_split(df_train, test_size=0.15, stratify=df_train["label"], random_state=SEED)

print(f"[INFO] Dataset ACRIMA dividido: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}", flush=True)

transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

transform_eval = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

class FundusDataset(Dataset):
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
        return img, row["label"]

train_loader = DataLoader(FundusDataset(df_train, transform_train), batch_size=32, shuffle=True)
val_loader = DataLoader(FundusDataset(df_val, transform_eval), batch_size=32, shuffle=False)
test_loader = DataLoader(FundusDataset(df_test, transform_eval), batch_size=32, shuffle=False)

# -----------------------------------------------------------------------------
# Definición de Arquitecturas
# -----------------------------------------------------------------------------
def get_backbone(name):
    if name == "ResNet-18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif name == "ResNet-50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif name == "EfficientNet-B0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    elif name == "MobileNet-V2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    else:
        raise ValueError(f"Arquitectura no reconocida: {name}")
    return m

candidate_models = ["ResNet-18", "ResNet-50", "EfficientNet-B0", "MobileNet-V2"]
benchmark_results = {}

EPOCHS = 3

print("\n================ BENCHMARK DE ARQUITECTURAS BACKBONE ================", flush=True)

for name in candidate_models:
    print(f"\n---> Evaluando arquitectura: {name}", flush=True)
    model = get_backbone(name).to(DEVICE)

    # Conteo de parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)

    # Medición de tiempo de entrenamiento
    start_train_time = time.time()
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_epoch_loss = 0.0
        n_batches = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.float().unsqueeze(1).to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.detach()
            n_batches += 1
        epoch_train_loss = float(total_epoch_loss.cpu().item() / n_batches)
        train_loss_history.append(epoch_train_loss)

        # Validación
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.float().unsqueeze(1).to(DEVICE)
                out = model(imgs)
                loss = criterion(out, lbls)
                total_val_loss += loss.detach()
                n_val_batches += 1
        epoch_val_loss = float(total_val_loss.cpu().item() / n_val_batches)
        val_loss_history.append(epoch_val_loss)
        print(f"     Época {epoch}/{EPOCHS} -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}", flush=True)

    train_duration = time.time() - start_train_time

    # Latencia de inferencia (10 repeticiones de batch=32)
    model.eval()
    dummy_input = torch.randn(32, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    # Warmup
    for _ in range(3):
        _ = model(dummy_input)
    if DEVICE.type == "mps":
        torch.mps.synchronize()

    latencies = []
    for _ in range(10):
        t0 = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        if DEVICE.type == "mps":
            torch.mps.synchronize()
        latencies.append((time.time() - t0) * 1000.0) # ms
    avg_latency = float(np.mean(latencies))

    # Evaluación en Test Set
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            probs = torch.sigmoid(out).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(lbls.numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = float(accuracy_score(all_targets, all_preds))
    auc = float(roc_auc_score(all_targets, all_probs))
    sens = float(recall_score(all_targets, all_preds))
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    spec = float(tn / (tn + fp))
    f1 = float(f1_score(all_targets, all_preds))
    gen_gap = float(abs(train_loss_history[-1] - val_loss_history[-1]))

    print(f"  Parámetros: {total_params:,} ({model_size_mb:.2f} MB)", flush=True)
    print(f"  Latencia Inferencia: {avg_latency:.2f} ms/batch (32 imágenes)", flush=True)
    print(f"  Tiempo Entrenamiento ({EPOCHS} épocas): {train_duration:.2f} s", flush=True)
    print(f"  Test ROC-AUC: {auc:.4f} | Exactitud: {acc*100:.2f}% | Sens: {sens*100:.2f}% | Espec: {spec*100:.2f}% | F1: {f1:.4f}", flush=True)
    print(f"  Gap de Generalización (Loss Train vs Val): {gen_gap:.4f}", flush=True)

    benchmark_results[name] = {
        "params": int(total_params),
        "size_mb": float(model_size_mb),
        "latency_ms": float(avg_latency),
        "train_time_sec": float(train_duration),
        "test_auc": float(auc),
        "test_acc": float(acc),
        "test_sens": float(sens),
        "test_spec": float(spec),
        "test_f1": float(f1),
        "train_loss_final": float(train_loss_history[-1]),
        "val_loss_final": float(val_loss_history[-1]),
        "generalization_gap": float(gen_gap)
    }

# Guardar resultados JSON
with open(RESULTS_DIR / "backbone_selection_benchmark.json", "w", encoding="utf-8") as f:
    json.dump(benchmark_results, f, indent=4)
print(f"\n[OK] Resultados guardados en: {RESULTS_DIR / 'backbone_selection_benchmark.json'}", flush=True)

# -----------------------------------------------------------------------------
# Generación de Figura Comparativa Formal
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=300)

names = list(benchmark_results.keys())
aucs = [benchmark_results[n]["test_auc"] for n in names]
accs = [benchmark_results[n]["test_acc"] * 100 for n in names]
latencies = [benchmark_results[n]["latency_ms"] for n in names]
params = [benchmark_results[n]["params"] / 1e6 for n in names]
gaps = [benchmark_results[n]["generalization_gap"] for n in names]

# Panel 1: ROC-AUC y Exactitud
x = np.arange(len(names))
width = 0.35
axes[0, 0].bar(x - width/2, aucs, width, label="ROC-AUC", color="#1F77B4", alpha=0.9)
axes[0, 0].bar(x + width/2, [a/100 for a in accs], width, label="Exactitud (Ratio)", color="#2CA02C", alpha=0.9)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(names, fontsize=10, fontweight="bold")
axes[0, 0].set_ylim(0.85, 1.02)
axes[0, 0].set_title("Rendimiento Diagnóstico en Test Set (ACRIMA)", fontsize=11, fontweight="bold")
axes[0, 0].legend(loc="lower right")
axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)
for i in range(len(names)):
    axes[0, 0].text(x[i] - width/2, aucs[i] + 0.005, f"{aucs[i]:.3f}", ha="center", fontsize=8, fontweight="bold")
    axes[0, 0].text(x[i] + width/2, accs[i]/100 + 0.005, f"{accs[i]:.1f}%", ha="center", fontsize=8)

# Panel 2: Latencia de Inferencia (Eficiencia Computacional)
bars = axes[0, 1].bar(names, latencies, color="#FF7F0E", alpha=0.85, edgecolor="#CC6600")
axes[0, 1].set_ylabel("Latencia (ms / batch 32)", fontsize=10, fontweight="bold")
axes[0, 1].set_title("Velocidad de Inferencia por Lote", fontsize=11, fontweight="bold")
axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)
for bar in bars:
    yval = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f} ms", ha="center", fontsize=9, fontweight="bold")

# Panel 3: Tamaño del Modelo vs. Parámetros
bars = axes[1, 0].bar(names, params, color="#9467BD", alpha=0.85, edgecolor="#663399")
axes[1, 0].set_ylabel("Millones de Parámetros (M)", fontsize=10, fontweight="bold")
axes[1, 0].set_title("Complejidad Paramétrica y Huella de Memoria", fontsize=11, fontweight="bold")
axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)
for bar in bars:
    yval = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f} M", ha="center", fontsize=9, fontweight="bold")

# Panel 4: Gap de Generalización (Sobreajuste |Loss Train - Loss Val|)
bars = axes[1, 1].bar(names, gaps, color="#D62728", alpha=0.85, edgecolor="#990000")
axes[1, 1].set_ylabel("Diferencia Absoluta de Pérdida (|Train - Val|)", fontsize=10, fontweight="bold")
axes[1, 1].set_title("Riesgo de Sobreajuste (Generalization Gap)", fontsize=11, fontweight="bold")
axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)
for bar in bars:
    yval = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", ha="center", fontsize=9, fontweight="bold")

plt.suptitle("Benchmark Comparativo de Redes Convolucionales Base para Detección de Glaucoma\nEvaluación Multi-Criterio para Elección de la Arquitectura", fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_fig = FIGURES_DIR / "fig19_backbone_comparison_benchmark.png"
plt.savefig(out_fig)
plt.close()
print(f"[OK] Gráfico comparativo generado: {out_fig}", flush=True)
