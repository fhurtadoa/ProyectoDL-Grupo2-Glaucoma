#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 10: Benchmark Sistemático de 10 Arquitecturas Candidatas para Transfer Learning
y Fase de Fine-Tuning Profundo de la Red Ganadora en Detección de Glaucoma.

Modelos evaluados (10 candidatos de 5 familias):
1. resnet18 (Residual clásica ligera)
2. resnet50 (Residual clásica bottleneck)
3. resnext50_32x4d (Convoluciones agrupadas / cardinalidad)
4. densenet121 (Conexiones densas / reutilización de features)
5. efficientnet_b0 (Escalado compuesto clásico)
6. efficientnet_v2_s (Escalado Fused-MBConv moderno)
7. convnext_tiny (Convolucional moderno Post-ViT)
8. swin_t (Transformer jerárquico con ventanas desplazadas)
9. vit_b_16 (Vision Transformer global puro)
10. mobilenet_v3_large (Eficiencia móvil optimizada por NAS)

Protocolo:
- Mega-cohorte médica armonizada (data/HARMONIZED_MEGA_COHORT, ROI 35°)
- Warmup (backbone congelado) + Fine-tuning comparativo
- Evaluación multicriterio: Diagnóstico, Generalización, Eficiencia y Brecha
- Fase de Fine-Tuning profundo con data augmentation maligno para la red ganadora
- Generación de Figuras 30, 31 y 32 a 300 DPI
- Métricas consolidadas en JSON para el informe final
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
import torchvision.transforms as T
import torchvision.models as models
from torchvision.transforms import functional as TF

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 0. Configuración Global y Reproducibilidad
# -------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Dispositivo de cómputo: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_INDEX_PATH = BASE_DIR / "results" / "harmonized_mega_cohort_index.csv"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
MODELS_DIR = BASE_DIR / "modelos"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. Transformaciones y Malignant Scale Augmentation
# -------------------------------------------------------------
class MalignantScaleAugmentation(object):
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size
        self.p_zoom_out = 0.45
        self.p_zoom_in = 0.35
        self.p_photometric = 0.65

    def __call__(self, img):
        mode_roll = random.random()
        w, h = img.size

        if mode_roll < self.p_zoom_out:
            scale = random.uniform(0.25, 0.65)
            nw, nh = max(16, int(w * scale)), max(16, int(h * scale))
            img_scaled = img.resize((nw, nh), Image.BILINEAR)

            bg_arr = np.array(img)
            mu_bg = np.mean(bg_arr, axis=(0, 1))
            canvas_np = np.clip(
                np.random.normal(loc=mu_bg, scale=7.0, size=(h, w, 3)), 0, 255
            ).astype(np.uint8)

            cx, cy = w // 2, h // 2
            max_r = int(min(w, h) * 0.48)
            y_idx, x_idx = np.ogrid[:h, :w]
            dist = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
            canvas_np[dist > max_r] = [0, 0, 0]

            canvas = Image.fromarray(canvas_np)
            max_dx = max(0, w - nw)
            max_dy = max(0, h - nh)
            off_x = random.randint(0, max_dx) if max_dx > 0 else 0
            off_y = random.randint(0, max_dy) if max_dy > 0 else 0
            canvas.paste(img_scaled, (off_x, off_y))
            img = canvas

        elif mode_roll < (self.p_zoom_out + self.p_zoom_in):
            scale = random.uniform(1.25, 1.85)
            crop_w, crop_h = int(w / scale), int(h / scale)
            max_x = max(0, w - crop_w)
            max_y = max(0, h - crop_h)
            cx, cy = max_x // 2, max_y // 2
            jitter_x = int(cx + random.uniform(-0.15, 0.15) * max_x)
            jitter_y = int(cy + random.uniform(-0.15, 0.15) * max_y)
            jitter_x = max(0, min(max_x, jitter_x))
            jitter_y = max(0, min(max_y, jitter_y))
            img = img.crop((jitter_x, jitter_y, jitter_x + crop_w, jitter_y + crop_h))

        if random.random() < self.p_photometric:
            gamma = random.uniform(0.75, 1.35)
            img = TF.adjust_gamma(img, gamma)
            green_gain = random.uniform(0.90, 1.15)
            arr = np.array(img, dtype=np.float32)
            arr[:, :, 1] = np.clip(arr[:, :, 1] * green_gain, 0, 255)
            img = Image.fromarray(arr.astype(np.uint8))

        img = img.resize(self.output_size, Image.BILINEAR)
        return img

train_transform_standard = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

malignant_transform = T.Compose([
    MalignantScaleAugmentation(output_size=(224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------------------------------------------
# 2. Carga Rápida en Memoria RAM
# -------------------------------------------------------------
print("[*] Cargando índice de la mega-cohorte armonizada...")
df_all = pd.read_csv(DATA_INDEX_PATH)

df_train_full = df_all[df_all["split"] == "train"].copy().reset_index(drop=True)
df_test_full = df_all[df_all["split"] == "test"].copy().reset_index(drop=True)

# Split 85% train, 15% val estratificado
val_parts = []
train_parts = []
for (src, lbl), group in df_train_full.groupby(["source", "label"]):
    n_val = max(1, int(len(group) * 0.15))
    shuffled = group.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val_parts.append(shuffled.iloc[:n_val])
    train_parts.append(shuffled.iloc[n_val:])

train_df = pd.concat(train_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_df = pd.concat(val_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

print(f"[*] Tamaño Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(df_test_full)}")

# Pre-cargar imágenes de entrenamiento en RAM como PIL
print("[*] Pre-cargando imágenes de entrenamiento en RAM para máxima velocidad...")
t0_load = time.time()
train_pil_cache = [Image.open(p).convert("RGB") for p in train_df["path"]]
train_labels_cache = torch.tensor(train_df["label"].values, dtype=torch.float32)
print(f"[OK] {len(train_pil_cache)} imágenes cargadas en {time.time()-t0_load:.2f}s")

# Pre-cargar tensores de validación en GPU
print("[*] Pre-procesando tensores de validación y test...")
val_tensors = torch.stack([eval_transform(Image.open(p).convert("RGB")) for p in val_df["path"]]).to(DEVICE)
val_targets_tensor = torch.tensor(val_df["label"].values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
val_targets_np = val_df["label"].values

test_tensors = torch.stack([eval_transform(Image.open(p).convert("RGB")) for p in df_test_full["path"]]).to(DEVICE)
test_targets_tensor = torch.tensor(df_test_full["label"].values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
test_targets_np = df_test_full["label"].values
print(f"[OK] Tensores residentes en {DEVICE}")

# Dataset para DataLoader de entrenamiento
class RAMTrainDataset(Dataset):
    def __init__(self, pil_images, labels, transform):
        self.pil_images = pil_images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.pil_images)

    def __getitem__(self, idx):
        img = self.transform(self.pil_images[idx])
        label = self.labels[idx]
        return img, label

# -------------------------------------------------------------
# 3. Diccionario de 10 Arquitecturas Candidatas
# -------------------------------------------------------------
CANDIDATE_CONFIGS = [
    {
        "id": "resnet18",
        "name": "ResNet-18",
        "family": "Residual Clásica",
        "fn": models.resnet18,
        "weights": "DEFAULT"
    },
    {
        "id": "resnet50",
        "name": "ResNet-50",
        "family": "Residual Bottleneck",
        "fn": models.resnet50,
        "weights": "DEFAULT"
    },
    {
        "id": "resnext50_32x4d",
        "name": "ResNeXt-50 (32x4d)",
        "family": "Convolución Agrupada",
        "fn": models.resnext50_32x4d,
        "weights": "DEFAULT"
    },
    {
        "id": "densenet121",
        "name": "DenseNet-121",
        "family": "Conexiones Densas",
        "fn": models.densenet121,
        "weights": "DEFAULT"
    },
    {
        "id": "efficientnet_b0",
        "name": "EfficientNet-B0",
        "family": "Escalado Compuesto",
        "fn": models.efficientnet_b0,
        "weights": "DEFAULT"
    },
    {
        "id": "efficientnet_v2_s",
        "name": "EfficientNet-V2-S",
        "family": "Escalado Moderno",
        "fn": models.efficientnet_v2_s,
        "weights": "DEFAULT"
    },
    {
        "id": "convnext_tiny",
        "name": "ConvNeXt-Tiny",
        "family": "Convolucional Post-ViT",
        "fn": models.convnext_tiny,
        "weights": "DEFAULT"
    },
    {
        "id": "swin_t",
        "name": "Swin-T",
        "family": "Hierarchical Transformer",
        "fn": models.swin_t,
        "weights": "DEFAULT"
    },
    {
        "id": "vit_b_16",
        "name": "ViT-B/16",
        "family": "Global Transformer",
        "fn": models.vit_b_16,
        "weights": "DEFAULT"
    },
    {
        "id": "mobilenet_v3_large",
        "name": "MobileNet-V3-Large",
        "family": "Eficiencia Móvil / Edge",
        "fn": models.mobilenet_v3_large,
        "weights": "DEFAULT"
    }
]

def build_model(cfg):
    model = cfg["fn"](weights=cfg["weights"])
    # Reemplazo uniforme del clasificador
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, 1)
        elif isinstance(model.classifier, nn.Sequential):
            last_idx = max(i for i, layer in enumerate(model.classifier) if isinstance(layer, nn.Linear))
            in_f = model.classifier[last_idx].in_features
            model.classifier[last_idx] = nn.Linear(in_f, 1)
    elif hasattr(model, 'heads'):
        model.heads.head = nn.Linear(model.heads.head.in_features, 1)
    elif hasattr(model, 'head'):
        model.head = nn.Linear(model.head.in_features, 1)
    return model

def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, 'fc'):
        for p in model.fc.parameters(): p.requires_grad = True
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Linear):
            for p in model.classifier.parameters(): p.requires_grad = True
        elif isinstance(model.classifier, nn.Sequential):
            last_idx = max(i for i, layer in enumerate(model.classifier) if isinstance(layer, nn.Linear))
            for p in model.classifier[last_idx].parameters(): p.requires_grad = True
    elif hasattr(model, 'heads'):
        for p in model.heads.head.parameters(): p.requires_grad = True
    elif hasattr(model, 'head'):
        for p in model.head.parameters(): p.requires_grad = True

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

# -------------------------------------------------------------
# 4. Benchmarking de Inferencia (Latencia y VRAM)
# -------------------------------------------------------------
def benchmark_latency_and_memory(model, sample_input, device, n_warmup=15, n_runs=50):
    model.eval()
    sample_input = sample_input.to(device)
    
    # Warmup GPU
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(sample_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Medir latencia GPU
    t_start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
    total_time = time.time() - t_start
    lat_ms = (total_time / n_runs) * 1000.0 / sample_input.size(0)

    # Memoria VRAM pico
    vram_mb = 0.0
    if device.type == "cuda":
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return lat_ms, vram_mb

# -------------------------------------------------------------
# 5. Ejecución del Benchmark Comparativo (10 Candidatos)
# -------------------------------------------------------------
BENCHMARK_EPOCHS_WARMUP = 1
BENCHMARK_EPOCHS_FINETUNE = 7
TOTAL_BENCH_EPOCHS = BENCHMARK_EPOCHS_WARMUP + BENCHMARK_EPOCHS_FINETUNE
BATCH_SIZE = 32

benchmark_results = []
trained_models = {}

print("\n" + "="*80)
print(f"[*] INICIANDO BENCHMARK SISTEMÁTICO: 10 ARQUITECTURAS CANDIDATAS")
print(f"[*] Configuración: {BENCHMARK_EPOCHS_WARMUP} Ep Warmup + {BENCHMARK_EPOCHS_FINETUNE} Ep Fine-Tuning | Batch: {BATCH_SIZE}")
print("="*80 + "\n")

criterion = nn.BCEWithLogitsLoss()
train_dataset = RAMTrainDataset(train_pil_cache, train_labels_cache, transform=train_transform_standard)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

for idx, cfg in enumerate(CANDIDATE_CONFIGS):
    model_id = cfg["id"]
    model_name = cfg["name"]
    family = cfg["family"]
    
    print(f"\n---> [{idx+1}/10] Evaluando: {model_name} ({family})")
    torch.cuda.empty_cache()
    
    # 1. Construcción
    model = build_model(cfg).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    param_mb = (total_params * 4) / (1024 * 1024)
    
    # 2. Medir Latencia en Batch=1 y Batch=32
    sample_b1 = torch.randn(1, 3, 224, 224)
    sample_b32 = torch.randn(32, 3, 224, 224)
    lat_b1_gpu, vram_peak = benchmark_latency_and_memory(model, sample_b1, DEVICE)
    lat_b32_gpu, _ = benchmark_latency_and_memory(model, sample_b32, DEVICE)

    # 3. Fase Warmup
    freeze_backbone(model)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    t_train_start = time.time()
    for ep in range(BENCHMARK_EPOCHS_WARMUP):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(x_b), y_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(x_b), y_b)
                loss.backward()
                optimizer.step()

    # 4. Fase Fine-Tuning
    unfreeze_all(model)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=BENCHMARK_EPOCHS_FINETUNE, eta_min=1e-5)

    best_val_auc = 0.0
    best_weights = None
    train_loss_hist = []

    for ep in range(BENCHMARK_EPOCHS_FINETUNE):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(x_b)
                    loss = criterion(logits, y_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * x_b.size(0)
            n_samples += x_b.size(0)

        scheduler.step()
        epoch_loss = running_loss / n_samples
        train_loss_hist.append(epoch_loss)

        # Validación
        model.eval()
        with torch.no_grad():
            if scaler:
                with torch.amp.autocast('cuda'):
                    val_logits = model(val_tensors)
            else:
                val_logits = model(val_tensors)
            val_probs = torch.sigmoid(val_logits).cpu().squeeze(1).numpy()
        val_auc = float(roc_auc_score(val_targets_np, val_probs))
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_duration = time.time() - t_train_start
    final_train_loss = train_loss_hist[-1]

    # Restaurar mejor modelo
    if best_weights:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})

    # 5. Evaluación Exhaustiva en Test Set ($N=431$)
    model.eval()
    with torch.no_grad():
        if scaler:
            with torch.amp.autocast('cuda'):
                test_logits = model(test_tensors)
                test_loss = criterion(test_logits, test_targets_tensor).item()
        else:
            test_logits = model(test_tensors)
            test_loss = criterion(test_logits, test_targets_tensor).item()
        test_probs = torch.sigmoid(test_logits).cpu().squeeze(1).numpy()

    test_preds = (test_probs >= 0.5).astype(int)
    global_auc = float(roc_auc_score(test_targets_np, test_probs))
    global_acc = float(accuracy_score(test_targets_np, test_preds))
    global_sens = float(recall_score(test_targets_np, test_preds))
    global_spec = float(confusion_matrix(test_targets_np, test_preds)[0, 0] / (confusion_matrix(test_targets_np, test_preds)[0, 0] + confusion_matrix(test_targets_np, test_preds)[0, 1]))
    global_f1 = float(f1_score(test_targets_np, test_preds))
    cm = confusion_matrix(test_targets_np, test_preds).tolist()
    gen_gap = abs(test_loss - final_train_loss)

    # 6. Desglose por Cohortes Externas
    cohort_results = {}
    for c_name in ["ACRIMA", "RIM-ONE DL", "EyePACS", "Moondream"]:
        c_mask = (df_test_full["source"] == c_name).values
        c_y_true = test_targets_np[c_mask]
        c_y_probs = test_probs[c_mask]
        c_y_preds = test_preds[c_mask]
        c_auc = float(roc_auc_score(c_y_true, c_y_probs)) if len(np.unique(c_y_true)) > 1 else 1.0
        c_acc = float(accuracy_score(c_y_true, c_y_preds))
        c_spec = float(confusion_matrix(c_y_true, c_y_preds)[0, 0] / (confusion_matrix(c_y_true, c_y_preds)[0, 0] + confusion_matrix(c_y_true, c_y_preds)[0, 1])) if len(confusion_matrix(c_y_true, c_y_preds)) > 1 else 1.0
        cohort_results[c_name] = {
            "auc": c_auc,
            "acc": c_acc,
            "spec": c_spec
        }

    # Cross-domain promedio entre sensores retadores (RIM-ONE 3D y EyePACS 45°)
    cross_domain_auc = (cohort_results["RIM-ONE DL"]["auc"] + cohort_results["EyePACS"]["auc"]) / 2.0

    print(f"    -> Test ROC-AUC: {global_auc:.4f} | Acc: {global_acc*100:.2f}% | Spec: {global_spec*100:.2f}% | F1: {global_f1:.4f}")
    print(f"    -> Cross-Domain (RIM-ONE + EyePACS): {cross_domain_auc:.4f} | Latencia GPU (Batch 1): {lat_b1_gpu:.2f} ms")

    # Guardar checkpoint base
    ckpt_path = MODELS_DIR / f"backbone_screen_{model_id}.pth"
    torch.save(model.state_dict(), ckpt_path)

    res_item = {
        "id": model_id,
        "name": model_name,
        "family": family,
        "total_params_M": round(total_params / 1e6, 2),
        "param_mb": round(param_mb, 2),
        "lat_gpu_b1_ms": round(lat_b1_gpu, 2),
        "lat_gpu_b32_ms": round(lat_b32_gpu, 2),
        "vram_peak_mb": round(vram_peak, 2),
        "train_time_sec": round(train_duration, 2),
        "train_loss": round(final_train_loss, 4),
        "test_loss": round(test_loss, 4),
        "generalization_gap": round(gen_gap, 4),
        "roc_auc": round(global_auc, 4),
        "accuracy": round(global_acc, 4),
        "sensitivity": round(global_sens, 4),
        "specificity": round(global_spec, 4),
        "f1_score": round(global_f1, 4),
        "cross_domain_auc": round(cross_domain_auc, 4),
        "cohorts": cohort_results,
        "confusion_matrix": cm,
        "test_probs": test_probs.tolist() # Para curvas ROC
    }
    benchmark_results.append(res_item)

# -------------------------------------------------------------
# 6. Cálculo del Índice Multicriterio de Decisión y Selección
# -------------------------------------------------------------
print("\n" + "="*80)
print("[*] EVALUACIÓN MULTICRITERIO Y SELECCIÓN DEL MODELO GANADOR")
print("="*80)

# Normalizar factores de eficiencia para el score
aucs = np.array([r["roc_auc"] for r in benchmark_results])
accs = np.array([r["accuracy"] for r in benchmark_results])
specs = np.array([r["specificity"] for r in benchmark_results])
cross_aucs = np.array([r["cross_domain_auc"] for r in benchmark_results])
lats = np.array([r["lat_gpu_b1_ms"] for r in benchmark_results])
params = np.array([r["total_params_M"] for r in benchmark_results])

# Eficiencia normalizada (mayor es mejor: inversa de latencia y parámetros)
lat_norm = 1.0 - (lats - lats.min()) / (lats.max() - lats.min() + 1e-6)
param_norm = 1.0 - (params - params.min()) / (params.max() - params.min() + 1e-6)
efficiency_score = 0.5 * lat_norm + 0.5 * param_norm

# Score compuesto clínico-técnico ponderado:
# 35% ROC-AUC, 20% Exactitud, 20% Especificidad (crítica en cribado), 15% Cross-Domain, 10% Eficiencia
composite_scores = (
    0.35 * aucs +
    0.20 * accs +
    0.20 * specs +
    0.15 * cross_aucs +
    0.10 * efficiency_score
)

for i, r in enumerate(benchmark_results):
    r["composite_score"] = round(float(composite_scores[i]), 4)
    r["efficiency_score"] = round(float(efficiency_score[i]), 4)

# Ordenar por score compuesto
benchmark_results = sorted(benchmark_results, key=lambda x: x["composite_score"], reverse=True)
winner = benchmark_results[0]

print(f"\n🏆 MODELO GANADOR DEL BENCHMARK: {winner['name']} ({winner['family']})")
print(f"   -> Score Compuesto: {winner['composite_score']:.4f}")
print(f"   -> ROC-AUC Test: {winner['roc_auc']:.4f}")
print(f"   -> Exactitud: {winner['accuracy']*100:.2f}% | Especificidad: {winner['specificity']*100:.2f}%")
print(f"   -> Cross-Domain AUC: {winner['cross_domain_auc']:.4f}")
print(f"   -> Latencia GPU: {winner['lat_gpu_b1_ms']:.2f} ms | Parámetros: {winner['total_params_M']} M")

# Guardar resultados del benchmark inicial
with open(RESULTS_DIR / "benchmark_10_backbones_metrics.json", "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_models": len(benchmark_results),
        "winner_id": winner["id"],
        "winner_name": winner["name"],
        "benchmark_summary": benchmark_results
    }, f, indent=4)
print(f"[OK] Resultados de benchmark guardados en results/benchmark_10_backbones_metrics.json")

# -------------------------------------------------------------
# 7. Fase de Fine-Tuning Profundo para la Red Ganadora
# -------------------------------------------------------------
print("\n" + "="*80)
print(f"[*] FASE DE FINE-TUNING PROFUNDO PARA EL MODELO GANADOR: {winner['name']}")
print(f"[*] Protocolo: Malignant Scale Augmentation + Cosine Annealing (25 Épocas)")
print("="*80 + "\n")

winner_cfg = next(c for c in CANDIDATE_CONFIGS if c["id"] == winner["id"])
winning_model = build_model(winner_cfg).to(DEVICE)

# Cargar pesos del benchmark como base
screen_ckpt = MODELS_DIR / f"backbone_screen_{winner['id']}.pth"
winning_model.load_state_dict(torch.load(screen_ckpt, map_location=DEVICE))

# Dataset de entrenamiento con Data Augmentation Maligno
malignant_train_dataset = RAMTrainDataset(train_pil_cache, train_labels_cache, transform=malignant_transform)
malignant_train_loader = DataLoader(malignant_train_dataset, batch_size=32, shuffle=True, drop_last=False)

# Optimizador diferencial: backbone con LR menor, cabeza con LR mayor
if hasattr(winning_model, 'fc'):
    head_params = list(winning_model.fc.parameters())
    backbone_params = [p for n, p in winning_model.named_parameters() if not n.startswith('fc')]
elif hasattr(winning_model, 'classifier'):
    head_params = list(winning_model.classifier.parameters())
    backbone_params = [p for n, p in winning_model.named_parameters() if not n.startswith('classifier')]
elif hasattr(winning_model, 'heads'):
    head_params = list(winning_model.heads.parameters())
    backbone_params = [p for n, p in winning_model.named_parameters() if not n.startswith('heads')]
elif hasattr(winning_model, 'head'):
    head_params = list(winning_model.head.parameters())
    backbone_params = [p for n, p in winning_model.named_parameters() if not n.startswith('head')]

optimizer_ft = optim.AdamW([
    {"params": backbone_params, "lr": 2e-5},
    {"params": head_params, "lr": 1e-4}
], weight_decay=1e-4)

FT_EPOCHS = 25
scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=FT_EPOCHS, eta_min=1e-6)
scaler_ft = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

ft_history = []
best_ft_auc = 0.0
best_ft_weights = None
winner_best_ckpt_path = MODELS_DIR / f"best_fine_tuned_{winner['id']}_glaucoma.pth"

t_ft_start = time.time()
for epoch in range(1, FT_EPOCHS + 1):
    winning_model.train()
    running_loss = 0.0
    n_samples = 0
    for x_b, y_b in malignant_train_loader:
        x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE).unsqueeze(1)
        optimizer_ft.zero_grad()
        if scaler_ft:
            with torch.amp.autocast('cuda'):
                logits = winning_model(x_b)
                loss = criterion(logits, y_b)
            scaler_ft.scale(loss).backward()
            scaler_ft.step(optimizer_ft)
            scaler_ft.update()
        else:
            logits = winning_model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer_ft.step()
        running_loss += loss.item() * x_b.size(0)
        n_samples += x_b.size(0)

    scheduler_ft.step()
    ep_train_loss = running_loss / n_samples

    # Validación
    winning_model.eval()
    with torch.no_grad():
        if scaler_ft:
            with torch.amp.autocast('cuda'):
                val_logits = winning_model(val_tensors)
                val_loss = criterion(val_logits, val_targets_tensor).item()
        else:
            val_logits = winning_model(val_tensors)
            val_loss = criterion(val_logits, val_targets_tensor).item()
        val_probs = torch.sigmoid(val_logits).cpu().squeeze(1).numpy()

    val_auc = float(roc_auc_score(val_targets_np, val_probs))
    val_acc = float(accuracy_score(val_targets_np, (val_probs >= 0.5).astype(int)))

    is_best = False
    if val_auc > best_ft_auc:
        best_ft_auc = val_auc
        is_best = True
        torch.save(winning_model.state_dict(), winner_best_ckpt_path)

    ft_history.append({
        "epoch": epoch,
        "train_loss": round(ep_train_loss, 5),
        "val_loss": round(val_loss, 5),
        "val_auc": round(val_auc, 4),
        "val_acc": round(val_acc, 4),
        "is_best": is_best
    })

    if epoch % 5 == 0 or is_best or epoch == FT_EPOCHS:
        print(f"  [Época {epoch:02d}/{FT_EPOCHS}] Train Loss: {ep_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Acc: {val_acc*100:.2f}% {'★ (MEJOR)' if is_best else ''}")

ft_total_time = time.time() - t_ft_start
print(f"[OK] Fine-tuning completado en {ft_total_time:.2f}s ({ft_total_time/60:.2f} min)")

# Cargar mejores pesos para evaluación final en Test
winning_model.load_state_dict(torch.load(winner_best_ckpt_path, map_location=DEVICE))
winning_model.eval()

with torch.no_grad():
    if scaler_ft:
        with torch.amp.autocast('cuda'):
            test_logits = winning_model(test_tensors)
    else:
        test_logits = winning_model(test_tensors)
    final_test_probs = torch.sigmoid(test_logits).cpu().squeeze(1).numpy()

final_preds = (final_test_probs >= 0.5).astype(int)
final_auc = float(roc_auc_score(test_targets_np, final_test_probs))
final_acc = float(accuracy_score(test_targets_np, final_preds))
final_sens = float(recall_score(test_targets_np, final_preds))
final_cm = confusion_matrix(test_targets_np, final_preds)
final_spec = float(final_cm[0, 0] / (final_cm[0, 0] + final_cm[0, 1]))
final_f1 = float(f1_score(test_targets_np, final_preds))

final_cohort_metrics = {}
for c_name in ["ACRIMA", "RIM-ONE DL", "EyePACS", "Moondream"]:
    c_mask = (df_test_full["source"] == c_name).values
    c_y = test_targets_np[c_mask]
    c_p = final_test_probs[c_mask]
    c_pred = final_preds[c_mask]
    c_cm = confusion_matrix(c_y, c_pred)
    final_cohort_metrics[c_name] = {
        "auc": float(roc_auc_score(c_y, c_p)) if len(np.unique(c_y)) > 1 else 1.0,
        "acc": float(accuracy_score(c_y, c_pred)),
        "sens": float(recall_score(c_y, c_pred)),
        "spec": float(c_cm[0, 0] / (c_cm[0, 0] + c_cm[0, 1])) if len(c_cm) > 1 else 1.0,
        "cm": c_cm.tolist()
    }

print("\n" + "="*80)
print(f"[*] RESULTADOS FINALES DE LA RED GANADORA TRAS FINE-TUNING ({winner['name']}):")
print(f"    - ROC-AUC Global: {final_auc:.4f} (Ganancia vs. Pre: {final_auc - winner['roc_auc']:+.4f})")
print(f"    - Exactitud Global: {final_acc*100:.2f}% | Sensibilidad: {final_sens*100:.2f}% | Especificidad: {final_spec*100:.2f}%")
print(f"    - Falsos Positivos Totales: {final_cm[0, 1]} (en {final_cm[0, 0]+final_cm[0, 1]} normales)")
for c_name, m in final_cohort_metrics.items():
    print(f"    * {c_name:<12}: AUC = {m['auc']:.4f} | Acc = {m['acc']*100:.2f}% | Spec = {m['spec']*100:.2f}%")
print("="*80)

# Guardar registro de fine-tuning
with open(RESULTS_DIR / "winning_backbone_finetuning_metrics.json", "w", encoding="utf-8") as f:
    json.dump({
        "winner_id": winner["id"],
        "winner_name": winner["name"],
        "checkpoint_path": str(winner_best_ckpt_path),
        "fine_tuning_epochs": FT_EPOCHS,
        "fine_tuning_time_sec": round(ft_total_time, 2),
        "history": ft_history,
        "final_metrics": {
            "roc_auc": round(final_auc, 4),
            "accuracy": round(final_acc, 4),
            "sensitivity": round(final_sens, 4),
            "specificity": round(final_spec, 4),
            "f1_score": round(final_f1, 4),
            "confusion_matrix": final_cm.tolist(),
            "cohorts": final_cohort_metrics
        }
    }, f, indent=4)

# -------------------------------------------------------------
# 8. Generación de Figuras de Publicación Científica (Fig 30, 31, 32)
# -------------------------------------------------------------
print("\n[*] Generando figuras de publicación a 300 DPI...")

# -------------------------------------------------------------
# Figura 30: Trade-Off Latencia vs ROC-AUC vs Parámetros
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)

family_colors = {
    "Residual Clásica": "#1e40af",
    "Residual Bottleneck": "#2563eb",
    "Convolución Agrupada": "#0284c7",
    "Conexiones Densas": "#0d9488",
    "Escalado Compuesto": "#059669",
    "Escalado Moderno": "#10b981",
    "Convolucional Post-ViT": "#7c3aed",
    "Hierarchical Transformer": "#d97706",
    "Global Transformer": "#dc2626",
    "Eficiencia Móvil / Edge": "#e11d48"
}

for r in benchmark_results:
    c = family_colors.get(r["family"], "#475569")
    # Tamaño de burbuja proporcional a los parámetros
    sz = np.clip(r["total_params_M"] * 18, 90, 950)
    ax.scatter(r["lat_gpu_b1_ms"], r["roc_auc"], s=sz, color=c, alpha=0.75, edgecolors="#0f172a", lw=1.2)
    
    # Etiqueta de texto con desplazamiento inteligente
    offset_y = 0.0035 if r["id"] not in ["convnext_tiny", "swin_t"] else -0.0055
    ax.annotate(
        f"{r['name']}\n({r['total_params_M']}M | Score: {r['composite_score']:.3f})",
        (r["lat_gpu_b1_ms"], r["roc_auc"] + offset_y),
        fontsize=8.0, fontweight="bold", ha="center", color="#0f172a"
    )

# Resaltar ganador con corona / aureola
ax.scatter(
    winner["lat_gpu_b1_ms"], winner["roc_auc"],
    s=winner["total_params_M"] * 18 + 160,
    facecolors='none', edgecolors='#b45309', lw=2.5, linestyle='--'
)

ax.set_title("Figura 30: Compromiso Multidimensional en Transfer Learning (10 Arquitecturas)\nLatencia de Inferencia GPU vs. Capacidad Diagnóstica (ROC-AUC) vs. Número de Parámetros", fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("Latencia Media de Inferencia en GPU (Batch = 1, ms/muestra)", fontsize=9.5, fontweight="bold")
ax.set_ylabel("ROC-AUC en Conjunto de Prueba Multicéntrico ($N=431$)", fontsize=9.5, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.45)
ax.set_ylim([min(r["roc_auc"] for r in benchmark_results) - 0.015, max(r["roc_auc"] for r in benchmark_results) + 0.015])

# Leyenda de familias
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=col, markersize=8, label=fam) for fam, col in family_colors.items()]
ax.legend(handles=legend_handles, title="Familia Arquitectónica", loc="lower left", fontsize=7.5, title_fontsize=8.5, frameon=True)

plt.tight_layout()
fig30_path = FIGURES_DIR / "fig30_backbone_benchmark_multicriteria_tradeoff.png"
plt.savefig(fig30_path, bbox_inches="tight")
plt.close()
print(f"[OK] Figura 30 guardada en: {fig30_path}")

# -------------------------------------------------------------
# Figura 31: Curvas ROC y Radar Multidimensional
# -------------------------------------------------------------
fig, (ax_roc, ax_radar) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# 1. Curvas ROC Superpuestas
for r in benchmark_results:
    fpr, tpr, _ = roc_curve(test_targets_np, r["test_probs"])
    c = family_colors.get(r["family"], "#475569")
    is_w = (r["id"] == winner["id"])
    ax_roc.plot(fpr, tpr, lw=2.5 if is_w else 1.2, color=c, alpha=0.9 if is_w else 0.6, label=f"{r['name']} (AUC = {r['roc_auc']:.3f})")

ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.4)
ax_roc.set_title("A. Curvas ROC Superpuestas (10 Candidatos en Test Held-Out)", fontsize=10.5, fontweight="bold")
ax_roc.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontsize=9, fontweight="bold")
ax_roc.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)", fontsize=9, fontweight="bold")
ax_roc.legend(loc="lower right", fontsize=7.5, frameon=True)
ax_roc.grid(True, linestyle="--", alpha=0.4)

# 2. Gráfico Radar Multidimensional (Top 5 Familias)
# Dimensiones: ROC-AUC, Exactitud, Especificidad, Generalización Cruzada, Eficiencia
categories = ["ROC-AUC", "Exactitud", "Especificidad", "Cross-Domain", "Eficiencia"]
N_cat = len(categories)
angles = [n / float(N_cat) * 2 * np.pi for n in range(N_cat)]
angles += angles[:1]

# Convertir ax_radar a polar
plt.delaxes(ax_radar)
ax_polar = fig.add_subplot(1, 2, 2, polar=True)

# Seleccionar 5 modelos clave representativos
radar_models = [benchmark_results[0], benchmark_results[1], benchmark_results[2], benchmark_results[-2], benchmark_results[-1]]
for m_rad in radar_models:
    vals = [
        m_rad["roc_auc"],
        m_rad["accuracy"],
        m_rad["specificity"],
        m_rad["cross_domain_auc"],
        m_rad["efficiency_score"]
    ]
    vals += vals[:1]
    c = family_colors.get(m_rad["family"], "#2563eb")
    ax_polar.plot(angles, vals, lw=2.0 if m_rad["id"] == winner["id"] else 1.2, color=c, label=f"{m_rad['name']}")
    ax_polar.fill(angles, vals, color=c, alpha=0.15 if m_rad["id"] == winner["id"] else 0.05)

ax_polar.set_xticks(angles[:-1])
ax_polar.set_xticklabels(categories, fontsize=8.5, fontweight="bold")
ax_polar.set_ylim(0.5, 1.0)
ax_polar.set_title("B. Radar Comparativo Multicriterio (Modelos Clave)", fontsize=10.5, fontweight="bold", pad=15)
ax_polar.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8, frameon=True)

plt.suptitle("Figura 31: Evaluación Comparativa de Discriminación Diagnóstica y Radar Multicriterio", fontsize=12, fontweight="bold", y=0.98)
plt.tight_layout()
fig31_path = FIGURES_DIR / "fig31_backbone_roc_curves_and_radar.png"
plt.savefig(fig31_path, bbox_inches="tight")
plt.close()
print(f"[OK] Figura 31 guardada en: {fig31_path}")

# -------------------------------------------------------------
# Figura 32: Curvas de Fine-Tuning y Matrices de Confusión del Ganador
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

# Curvas de Fine-Tuning
df_ft = pd.DataFrame(ft_history)
ax_ft = axes[0]
ax_ft.plot(df_ft["epoch"], df_ft["train_loss"], color="#1e40af", lw=2.0, label="Train Loss (Malignant Aug)")
ax_ft.plot(df_ft["epoch"], df_ft["val_loss"], color="#dc2626", lw=2.0, label="Val Loss")
ax_ft_r = ax_ft.twinx()
ax_ft_r.plot(df_ft["epoch"], df_ft["val_auc"], color="#059669", lw=2.2, linestyle="--", label="Val ROC-AUC")
ax_ft_r.plot(df_ft["epoch"], df_ft["val_acc"]*100, color="#7c3aed", lw=1.8, linestyle=":", label="Val Accuracy (%)")

ax_ft.set_title(f"A. Trayectoria de Fine-Tuning Profundo: {winner['name']}", fontsize=10.5, fontweight="bold")
ax_ft.set_xlabel("Época de Fine-Tuning", fontweight="bold")
ax_ft.set_ylabel("Pérdida Binaria (BCE Loss)", fontweight="bold", color="#1e40af")
ax_ft_r.set_ylabel("ROC-AUC / Exactitud (%)", fontweight="bold", color="#059669")
lines1, labels1 = ax_ft.get_legend_handles_labels()
lines2, labels2 = ax_ft_r.get_legend_handles_labels()
ax_ft.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8, frameon=True)
ax_ft.grid(True, linestyle="--", alpha=0.4)

# Matrices de Confusión por Cohorte
ax_cm = axes[1]
# Dibujar 4 sub-matrices de confusión en un canvas de 2x2
cohort_list = ["ACRIMA", "RIM-ONE DL", "EyePACS", "Moondream"]
grid_cm = np.zeros((2, 2))
# Mostramos la matriz global en detalle
cm_global = np.array(final_cm)
im = ax_cm.imshow(cm_global, cmap="Blues", interpolation="nearest")
ax_cm.set_title(f"B. Matriz de Confusión Global en Prueba ($N=431$)\nExactitud: {final_acc*100:.2f}% | Especificidad: {final_spec*100:.2f}%", fontsize=10.5, fontweight="bold")
ax_cm.set_xticks([0, 1])
ax_cm.set_xticklabels(["Sano (Control)", "Glaucoma"], fontweight="bold")
ax_cm.set_yticks([0, 1])
ax_cm.set_yticklabels(["Sano (Control)", "Glaucoma"], fontweight="bold")
ax_cm.set_xlabel("Predicción del Modelo", fontweight="bold")
ax_cm.set_ylabel("Etiqueta Clínica Real", fontweight="bold")

thresh = cm_global.max() / 2.0
for i in range(2):
    for j in range(2):
        v = cm_global[i, j]
        ax_cm.text(j, i, f"{v}\n({v/cm_global.sum()*100:.1f}%)", ha="center", va="center", color="white" if v > thresh else "black", fontweight="bold", fontsize=11)

plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

plt.suptitle(f"Figura 32: Optimización y Desempeño Diagnóstico de la Arquitectura Ganadora ({winner['name']})", fontsize=12, fontweight="bold", y=0.98)
plt.tight_layout()
fig32_path = FIGURES_DIR / "fig32_winning_model_finetuning_curves_and_confusion_matrix.png"
plt.savefig(fig32_path, bbox_inches="tight")
plt.close()
print(f"[OK] Figura 32 guardada en: {fig32_path}")

print("\n[EXITO COMPLETO] Benchmark de 10 modelos y Fine-Tuning de la red ganadora finalizados satisfactoriamente.")
