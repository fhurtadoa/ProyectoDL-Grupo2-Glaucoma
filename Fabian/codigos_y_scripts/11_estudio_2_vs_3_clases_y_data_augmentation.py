#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 11: Estudio Riguroso Factorial:
1. Predicción sobre 2 Categorías (Sano vs. Glaucoma) vs. 3 Categorías (Sano vs. Prematuro vs. Avanzado).
2. Beneficio del Data Augmentation Maligno (Vanilla sin aumento vs. MalignantScaleAugmentation)
   sobre ambos esquemas de formulación clínica.
3. Evaluación cruzada y perfilado de severidad sobre cohortes externas (ACRIMA, RIM-ONE DL, EyePACS).

Genera:
- Checkpoints en modelos/
- Métricas en results/study_2_vs_3_classes_metrics.json y data_augmentation_benefit_metrics.json
- Figuras 33, 34, 35 y 36 a 300 DPI en figures/
"""

import os, sys, glob, time, json, random
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
    confusion_matrix, classification_report, roc_auc_score, accuracy_score,
    recall_score, precision_score, f1_score, cohen_kappa_score
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 0. Configuración Global
# ---------------------------------------------------------
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
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "modelos"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 1. Transformaciones y Malignant Scale Augmentation
# ---------------------------------------------------------
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

vanilla_eval_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

malignant_train_transform = T.Compose([
    MalignantScaleAugmentation(output_size=(224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# 2. Carga y Estructuración de Datos
# ---------------------------------------------------------
print("\n[*] 1. Indexando cohorte MOONDREAM_STAGED (600 Train, 200 Test)...")

def load_staged_dataset(split):
    records = []
    base_split = BASE_DIR / "data" / "MOONDREAM_STAGED" / split
    for f in sorted((base_split / "normal").glob("*.jpg")):
        records.append({"path": str(f), "label_3": 0, "label_2": 0, "stage": "normal"})
    for f in sorted((base_split / "early").glob("*.jpg")):
        records.append({"path": str(f), "label_3": 1, "label_2": 1, "stage": "early"})
    for f in sorted((base_split / "advanced").glob("*.jpg")):
        records.append({"path": str(f), "label_3": 2, "label_2": 1, "stage": "advanced"})
    return pd.DataFrame(records)

df_staged_train = load_staged_dataset("train").sample(frac=1.0, random_state=SEED).reset_index(drop=True)
df_staged_test = load_staged_dataset("test").reset_index(drop=True)

print(f"    - Train: {len(df_staged_train)} imágenes (Normal: 300, Early: 150, Advanced: 150)")
print(f"    - Test : {len(df_staged_test)} imágenes (Normal: 100, Early: 50, Advanced: 50)")

# Pre-cargar imágenes en RAM
print("[*] Pre-cargando imágenes en memoria RAM...")
train_pil = [Image.open(p).convert("RGB") for p in df_staged_train["path"]]
test_pil = [Image.open(p).convert("RGB") for p in df_staged_test["path"]]

test_tensors = torch.stack([vanilla_eval_transform(img) for img in test_pil]).to(DEVICE)
y_test_3 = df_staged_test["label_3"].values
y_test_2 = df_staged_test["label_2"].values

# Cargar cohortes externas independientes (ACRIMA, RIM-ONE DL, EyePACS)
print("[*] Cargando cohortes externas independientes para validación cruzada...")
df_index = pd.read_csv(RESULTS_DIR / "harmonized_mega_cohort_index.csv")
df_ext_test = df_index[df_index["split"] == "test"].copy()

external_cohorts = {}
for c_name in ["ACRIMA", "RIM-ONE DL", "EyePACS"]:
    sub = df_ext_test[df_ext_test["source"] == c_name].copy().reset_index(drop=True)
    tensors = torch.stack([vanilla_eval_transform(Image.open(p).convert("RGB")) for p in sub["path"]]).to(DEVICE)
    labels = sub["label"].values
    external_cohorts[c_name] = {
        "tensors": tensors,
        "labels": labels,
        "n": len(sub)
    }
    print(f"    - {c_name:<12}: {len(sub)} imágenes")

# Dataset para DataLoader
class StagedDataset(Dataset):
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

# ---------------------------------------------------------
# 3. Función de Entrenamiento y Evaluación
# ---------------------------------------------------------
EPOCHS = 20
BATCH_SIZE = 32

def train_and_eval(model_mode, use_augmentation, exp_name):
    """
    model_mode: 'binary' (2 clases) o 'multiclass' (3 clases)
    use_augmentation: True (Malignant) o False (Vanilla)
    """
    print(f"\n================================================================================")
    print(f"[*] EJECUTANDO: {exp_name.upper()}")
    print(f"[*] Modo: {model_mode} | Augmentation: {'MalignantScale' if use_augmentation else 'Vanilla (None)'}")
    print(f"================================================================================")

    # 1. Configurar red
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if model_mode == "binary":
        model.fc = nn.Linear(512, 1)
        criterion = nn.BCEWithLogitsLoss()
        y_train = torch.tensor(df_staged_train["label_2"].values, dtype=torch.float32)
    else:
        model.fc = nn.Linear(512, 3)
        # Ponderación inversa para balancear las 3 clases (300 normal, 150 early, 150 advanced)
        class_weights = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        y_train = torch.tensor(df_staged_train["label_3"].values, dtype=torch.long)

    model = model.to(DEVICE)

    # 2. Transformación
    tform = malignant_train_transform if use_augmentation else vanilla_eval_transform
    ds_train = StagedDataset(train_pil, y_train, transform=tform)
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    history = []
    t_start = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for x_b, y_b in loader_train:
            x_b = x_b.to(DEVICE)
            if model_mode == "binary":
                y_b = y_b.to(DEVICE).unsqueeze(1)
            else:
                y_b = y_b.to(DEVICE)

            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    out = model(x_b)
                    loss = criterion(out, y_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * x_b.size(0)
            n_samples += x_b.size(0)

        scheduler.step()
        ep_loss = running_loss / n_samples
        history.append({"epoch": ep, "train_loss": round(ep_loss, 5)})
        if ep % 5 == 0 or ep == EPOCHS:
            print(f"  [Época {ep:02d}/{EPOCHS}] Train Loss: {ep_loss:.4f}")

    train_time = time.time() - t_start
    print(f"[OK] Entrenamiento completado en {train_time:.2f}s")

    # Guardar modelo
    ckpt_name = f"model_staged_{model_mode}_{'malignant' if use_augmentation else 'vanilla'}.pth"
    torch.save(model.state_dict(), MODELS_DIR / ckpt_name)

    # 3. Evaluación en Test Set Moondream (N=200)
    model.eval()
    with torch.no_grad():
        if scaler:
            with torch.amp.autocast('cuda'):
                raw_logits = model(test_tensors)
        else:
            raw_logits = model(test_tensors)

    results_dict = {
        "exp_name": exp_name,
        "mode": model_mode,
        "use_augmentation": use_augmentation,
        "train_time_sec": round(train_time, 2),
        "history": history
    }

    if model_mode == "binary":
        probs = torch.sigmoid(raw_logits).cpu().squeeze(1).numpy()
        preds = (probs >= 0.5).astype(int)

        auc = float(roc_auc_score(y_test_2, probs))
        acc = float(accuracy_score(y_test_2, preds))
        sens = float(recall_score(y_test_2, preds))
        spec = float(confusion_matrix(y_test_2, preds)[0, 0] / 100.0)
        f1 = float(f1_score(y_test_2, preds))
        cm = confusion_matrix(y_test_2, preds).tolist()

        # Sensibilidad en Early vs Advanced
        is_early_mask = (y_test_3 == 1)
        is_adv_mask = (y_test_3 == 2)
        sens_early = float(np.mean(preds[is_early_mask] == 1))
        sens_adv = float(np.mean(preds[is_adv_mask] == 1))

        results_dict["metrics"] = {
            "roc_auc": round(auc, 4),
            "accuracy": round(acc, 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "sensitivity_early": round(sens_early, 4),
            "sensitivity_advanced": round(sens_adv, 4),
            "confusion_matrix": cm,
            "probs": probs.tolist(),
            "preds": preds.tolist()
        }
        print(f"  -> Binario Test: AUC={auc:.4f} | Acc={acc*100:.2f}% | Spec={spec*100:.2f}% | Sens Early={sens_early*100:.2f}% | Sens Adv={sens_adv*100:.2f}%")

    else: # multiclass (3 clases)
        probs_3 = torch.softmax(raw_logits, dim=1).cpu().numpy()
        preds_3 = np.argmax(probs_3, axis=1)

        acc_3 = float(accuracy_score(y_test_3, preds_3))
        macro_f1 = float(f1_score(y_test_3, preds_3, average="macro"))
        cm_3 = confusion_matrix(y_test_3, preds_3)

        # Métricas por clase
        class_metrics = {}
        for c_idx, c_name in enumerate(["Normal", "Early", "Advanced"]):
            tp = cm_3[c_idx, c_idx]
            fn = cm_3[c_idx, :].sum() - tp
            fp = cm_3[:, c_idx].sum() - tp
            tn = cm_3.sum() - (tp + fn + fp)
            class_metrics[c_name] = {
                "sensitivity": round(float(tp / (tp + fn + 1e-6)), 4),
                "specificity": round(float(tn / (tn + fp + 1e-6)), 4),
                "f1": round(float(2 * tp / (2 * tp + fp + fn + 1e-6)), 4)
            }

        # Tipología de Error Clínico
        # Error adyacente: Normal <-> Early (cm[0,1] + cm[1,0]) o Early <-> Advanced (cm[1,2] + cm[2,1])
        # Error catastrófico: Normal <-> Advanced (cm[0,2] + cm[2,0])
        total_errors = np.sum(cm_3) - np.trace(cm_3)
        err_adjacent = int(cm_3[0, 1] + cm_3[1, 0] + cm_3[1, 2] + cm_3[2, 1])
        err_catastrophic = int(cm_3[0, 2] + cm_3[2, 0])
        err_catastrophic_pct = round(float(err_catastrophic / (total_errors + 1e-6) * 100), 2)

        # Quadratic Weighted Kappa
        qwk = float(cohen_kappa_score(y_test_3, preds_3, weights="quadratic"))

        # Binarización Proyectada: P(Glaucoma) = P(Early) + P(Advanced) = 1 - P(Normal)
        proj_binary_probs = 1.0 - probs_3[:, 0]
        proj_binary_preds = (proj_binary_probs >= 0.5).astype(int)
        proj_auc = float(roc_auc_score(y_test_2, proj_binary_probs))
        proj_acc = float(accuracy_score(y_test_2, proj_binary_preds))
        proj_sens = float(recall_score(y_test_2, proj_binary_preds))
        proj_spec = float(confusion_matrix(y_test_2, proj_binary_preds)[0, 0] / 100.0)

        results_dict["metrics"] = {
            "accuracy_3": round(acc_3, 4),
            "macro_f1": round(macro_f1, 4),
            "qwk": round(qwk, 4),
            "class_metrics": class_metrics,
            "confusion_matrix_3x3": cm_3.tolist(),
            "errors": {
                "total": int(total_errors),
                "adjacent": err_adjacent,
                "catastrophic": err_catastrophic,
                "catastrophic_pct": err_catastrophic_pct
            },
            "projected_binary": {
                "roc_auc": round(proj_auc, 4),
                "accuracy": round(proj_acc, 4),
                "sensitivity": round(proj_sens, 4),
                "specificity": round(proj_spec, 4)
            },
            "probs": probs_3.tolist(),
            "preds": preds_3.tolist()
        }
        print(f"  -> 3 Clases Test: Acc={acc_3*100:.2f}% | Macro F1={macro_f1:.4f} | QWK={qwk:.4f}")
        print(f"     * Errores Adyacentes: {err_adjacent} | Errores Catastróficos: {err_catastrophic} ({err_catastrophic_pct}%)")
        print(f"     * Proyección Binaria: AUC={proj_auc:.4f} | Acc={proj_acc*100:.2f}% | Spec={proj_spec*100:.2f}%")

    # 4. Evaluación en Cohortes Externas (ACRIMA, RIM-ONE DL, EyePACS)
    external_eval = {}
    for c_name, c_data in external_cohorts.items():
        with torch.no_grad():
            if scaler:
                with torch.amp.autocast('cuda'):
                    ext_logits = model(c_data["tensors"])
            else:
                ext_logits = model(c_data["tensors"])

        if model_mode == "binary":
            ext_p = torch.sigmoid(ext_logits).cpu().squeeze(1).numpy()
            ext_pred = (ext_p >= 0.5).astype(int)
            ext_auc = float(roc_auc_score(c_data["labels"], ext_p))
            ext_acc = float(accuracy_score(c_data["labels"], ext_pred))
            external_eval[c_name] = {
                "auc": round(ext_auc, 4),
                "acc": round(ext_acc, 4)
            }
        else:
            ext_p3 = torch.softmax(ext_logits, dim=1).cpu().numpy()
            ext_pred3 = np.argmax(ext_p3, axis=1)
            ext_proj_p = 1.0 - ext_p3[:, 0]
            ext_proj_pred = (ext_proj_p >= 0.5).astype(int)
            ext_auc = float(roc_auc_score(c_data["labels"], ext_proj_p))
            ext_acc = float(accuracy_score(c_data["labels"], ext_proj_pred))

            # Distribución de estadíos predichos en los glaucomas reales de la cohorte externa
            glaucoma_mask = (c_data["labels"] == 1)
            glauc_preds = ext_pred3[glaucoma_mask]
            n_glauc = len(glauc_preds)
            stage_distribution = {
                "pct_classified_normal": round(float(np.mean(glauc_preds == 0) * 100), 1),
                "pct_classified_early": round(float(np.mean(glauc_preds == 1) * 100), 1),
                "pct_classified_advanced": round(float(np.mean(glauc_preds == 2) * 100), 1)
            }
            external_eval[c_name] = {
                "auc_projected": round(ext_auc, 4),
                "acc_projected": round(ext_acc, 4),
                "stage_distribution": stage_distribution
            }

    results_dict["external_cohorts"] = external_eval
    return results_dict

# ---------------------------------------------------------
# 4. Ejecución del Experimento Factorial 2x2
# ---------------------------------------------------------
print("\n" + "="*80)
print("[*] INICIANDO EXPERIMENTO FACTORIAL 2x2:")
print("    1. Binario Vanilla (2 Clases, Sin Augmentación)")
print("    2. Binario Maligno (2 Clases, Con Malignant Scale Augmentation)")
print("    3. Multiclase Vanilla (3 Clases, Sin Augmentación)")
print("    4. Multiclase Maligno (3 Clases, Con Malignant Scale Augmentation)")
print("="*80)

res_bin_vanilla = train_and_eval("binary", use_augmentation=False, exp_name="binario_vanilla")
res_bin_malignant = train_and_eval("binary", use_augmentation=True, exp_name="binario_malignant")
res_multi_vanilla = train_and_eval("multiclass", use_augmentation=False, exp_name="multiclase_vanilla")
res_multi_malignant = train_and_eval("multiclass", use_augmentation=True, exp_name="multiclase_malignant")

# ---------------------------------------------------------
# 5. Guardar Métricas en JSON
# ---------------------------------------------------------
study_2vs3_payload = {
    "title": "Estudio Comparativo: Predicción Binaria (2 Clases) vs. Estadificación Clínica (3 Clases)",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": "MOONDREAM_STAGED (600 Train, 200 Test)",
    "binary_vanilla": res_bin_vanilla,
    "binary_malignant": res_bin_malignant,
    "multiclass_vanilla": res_multi_vanilla,
    "multiclass_malignant": res_multi_malignant
}
with open(RESULTS_DIR / "study_2_vs_3_classes_metrics.json", "w", encoding="utf-8") as f:
    json.dump(study_2vs3_payload, f, indent=4)
print(f"\n[OK] Métricas de 2 vs 3 clases guardadas en: results/study_2_vs_3_classes_metrics.json")

# Guardar análisis del beneficio de data augmentation
aug_benefit_payload = {
    "title": "Evaluación del Beneficio de Data Augmentation Maligno en Clasificación de Glaucoma (2 y 3 Clases)",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "comparison_binary": {
        "auc_gain": round(res_bin_malignant["metrics"]["roc_auc"] - res_bin_vanilla["metrics"]["roc_auc"], 4),
        "accuracy_gain": round(res_bin_malignant["metrics"]["accuracy"] - res_bin_vanilla["metrics"]["accuracy"], 4),
        "spec_gain": round(res_bin_malignant["metrics"]["specificity"] - res_bin_vanilla["metrics"]["specificity"], 4),
        "early_sensitivity_gain": round(res_bin_malignant["metrics"]["sensitivity_early"] - res_bin_vanilla["metrics"]["sensitivity_early"], 4)
    },
    "comparison_multiclass": {
        "accuracy_gain": round(res_multi_malignant["metrics"]["accuracy_3"] - res_multi_vanilla["metrics"]["accuracy_3"], 4),
        "macro_f1_gain": round(res_multi_malignant["metrics"]["macro_f1"] - res_multi_vanilla["metrics"]["macro_f1"], 4),
        "qwk_gain": round(res_multi_malignant["metrics"]["qwk"] - res_multi_vanilla["metrics"]["qwk"], 4),
        "catastrophic_error_reduction": round(res_multi_vanilla["metrics"]["errors"]["catastrophic_pct"] - res_multi_malignant["metrics"]["errors"]["catastrophic_pct"], 2),
        "projected_auc_gain": round(res_multi_malignant["metrics"]["projected_binary"]["roc_auc"] - res_multi_vanilla["metrics"]["projected_binary"]["roc_auc"], 4)
    }
}
with open(RESULTS_DIR / "data_augmentation_benefit_metrics.json", "w", encoding="utf-8") as f:
    json.dump(aug_benefit_payload, f, indent=4)
print(f"[OK] Métricas de beneficio de Data Augmentation guardadas en: results/data_augmentation_benefit_metrics.json")

# ---------------------------------------------------------
# 6. Generación de Figuras de Publicación Científica (Fig 33, 34, 35, 36)
# ---------------------------------------------------------
print("\n[*] Generando figuras científicas a 300 DPI...")

# ---------------------------------------------------------
# Figura 33: Matrices de Confusión Comparativas (2x2 vs. 3x3)
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)

# A. Binario Vanilla (2x2)
cm_bin_van = np.array(res_bin_vanilla["metrics"]["confusion_matrix"])
ax = axes[0, 0]
im = ax.imshow(cm_bin_van, cmap="Blues", interpolation="nearest")
ax.set_title(f"A. Binario Vanilla (Sin Aumento)\nAcc: {res_bin_vanilla['metrics']['accuracy']*100:.1f}% | AUC: {res_bin_vanilla['metrics']['roc_auc']:.3f}", fontweight="bold", fontsize=10.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Sano", "Glaucoma"], fontweight="bold")
ax.set_yticks([0, 1])
ax.set_yticklabels(["Sano", "Glaucoma"], fontweight="bold")
ax.set_ylabel("Etiqueta Real", fontweight="bold")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm_bin_van[i, j]), ha="center", va="center", color="white" if cm_bin_van[i, j] > cm_bin_van.max()/2 else "black", fontweight="bold", fontsize=12)

# B. Binario Maligno (2x2)
cm_bin_mal = np.array(res_bin_malignant["metrics"]["confusion_matrix"])
ax = axes[0, 1]
ax.imshow(cm_bin_mal, cmap="Greens", interpolation="nearest")
ax.set_title(f"B. Binario Maligno (Con Data Augmentation)\nAcc: {res_bin_malignant['metrics']['accuracy']*100:.1f}% | AUC: {res_bin_malignant['metrics']['roc_auc']:.3f}", fontweight="bold", fontsize=10.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Sano", "Glaucoma"], fontweight="bold")
ax.set_yticks([0, 1])
ax.set_yticklabels(["Sano", "Glaucoma"], fontweight="bold")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm_bin_mal[i, j]), ha="center", va="center", color="white" if cm_bin_mal[i, j] > cm_bin_mal.max()/2 else "black", fontweight="bold", fontsize=12)

# C. Multiclase Vanilla (3x3)
cm_mul_van = np.array(res_multi_vanilla["metrics"]["confusion_matrix_3x3"])
ax = axes[1, 0]
ax.imshow(cm_mul_van, cmap="Oranges", interpolation="nearest")
ax.set_title(f"C. Estadificación 3 Clases Vanilla (Sin Aumento)\nAcc: {res_multi_vanilla['metrics']['accuracy_3']*100:.1f}% | QWK: {res_multi_vanilla['metrics']['qwk']:.3f}", fontweight="bold", fontsize=10.5)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Sano", "Prematuro", "Avanzado"], fontweight="bold", fontsize=9)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["Sano", "Prematuro", "Avanzado"], fontweight="bold", fontsize=9)
ax.set_xlabel("Predicción del Modelo", fontweight="bold")
ax.set_ylabel("Etiqueta Real", fontweight="bold")
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm_mul_van[i, j]), ha="center", va="center", color="white" if cm_mul_van[i, j] > cm_mul_van.max()/2 else "black", fontweight="bold", fontsize=11)

# D. Multiclase Maligno (3x3)
cm_mul_mal = np.array(res_multi_malignant["metrics"]["confusion_matrix_3x3"])
ax = axes[1, 1]
ax.imshow(cm_mul_mal, cmap="Purples", interpolation="nearest")
ax.set_title(f"D. Estadificación 3 Clases Maligno (Con Data Augmentation)\nAcc: {res_multi_malignant['metrics']['accuracy_3']*100:.1f}% | QWK: {res_multi_malignant['metrics']['qwk']:.3f}", fontweight="bold", fontsize=10.5)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Sano", "Prematuro", "Avanzado"], fontweight="bold", fontsize=9)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["Sano", "Prematuro", "Avanzado"], fontweight="bold", fontsize=9)
ax.set_xlabel("Predicción del Modelo", fontweight="bold")
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm_mul_mal[i, j]), ha="center", va="center", color="white" if cm_mul_mal[i, j] > cm_mul_mal.max()/2 else "black", fontweight="bold", fontsize=11)

plt.suptitle("Figura 33: Matrices de Confusión Comparativas en Moondream Staged Test (N=200)\n2 Categorías (Binario) vs. 3 Categorías (Estadificación Clínica) bajo Protocolo de Data Augmentation", fontsize=12, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig33_path = FIGURES_DIR / "fig33_confusion_matrices_2_vs_3_classes.png"
plt.savefig(fig33_path, bbox_inches="tight")
plt.close()
print(f"[OK] Figura 33 guardada en: {fig33_path}")

# ---------------------------------------------------------
# Figura 34: Beneficio Cuantitativo de Data Augmentation
# ---------------------------------------------------------
fig, (ax_bar1, ax_bar2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# Panel 1: Comparativa en 2 Clases
labels_bin = ["Exactitud (%)", "ROC-AUC (x100)", "Especificidad (%)", "Sens. Prematuro (%)", "Sens. Avanzado (%)"]
vals_vanilla_bin = [
    res_bin_vanilla["metrics"]["accuracy"] * 100,
    res_bin_vanilla["metrics"]["roc_auc"] * 100,
    res_bin_vanilla["metrics"]["specificity"] * 100,
    res_bin_vanilla["metrics"]["sensitivity_early"] * 100,
    res_bin_vanilla["metrics"]["sensitivity_advanced"] * 100
]
vals_mal_bin = [
    res_bin_malignant["metrics"]["accuracy"] * 100,
    res_bin_malignant["metrics"]["roc_auc"] * 100,
    res_bin_malignant["metrics"]["specificity"] * 100,
    res_bin_malignant["metrics"]["sensitivity_early"] * 100,
    res_bin_malignant["metrics"]["sensitivity_advanced"] * 100
]

x = np.arange(len(labels_bin))
width = 0.35

ax_bar1.bar(x - width/2, vals_vanilla_bin, width, label="Vanilla (Sin Aumento)", color="#94a3b8", edgecolor="#334155")
b2 = ax_bar1.bar(x + width/2, vals_mal_bin, width, label="Malignant Scale Augmentation", color="#1e40af", edgecolor="#0f172a")

ax_bar1.set_title("A. Beneficio del Data Augmentation en Formulación Binaria (2 Clases)", fontsize=10.5, fontweight="bold")
ax_bar1.set_xticks(x)
ax_bar1.set_xticklabels(labels_bin, fontsize=8.5, fontweight="bold")
ax_bar1.set_ylabel("Porcentaje (%)", fontweight="bold")
ax_bar1.set_ylim([60, 105])
ax_bar1.legend(loc="lower right", frameon=True)
ax_bar1.grid(True, linestyle="--", alpha=0.4, axis="y")

for i, (v1, v2) in enumerate(zip(vals_vanilla_bin, vals_mal_bin)):
    diff = v2 - v1
    color = "#059669" if diff >= 0 else "#dc2626"
    ax_bar1.annotate(f"{diff:+.1f}%", (x[i] + width/2, v2 + 1.2), ha="center", fontsize=8.5, fontweight="bold", color=color)

# Panel 2: Comparativa en 3 Clases
labels_mul = ["Exactitud Global (%)", "Macro F1 (x100)", "QWK (x100)", "Sens. Prematuro (%)", "AUC Proyectado (x100)"]
vals_vanilla_mul = [
    res_multi_vanilla["metrics"]["accuracy_3"] * 100,
    res_multi_vanilla["metrics"]["macro_f1"] * 100,
    res_multi_vanilla["metrics"]["qwk"] * 100,
    res_multi_vanilla["metrics"]["class_metrics"]["Early"]["sensitivity"] * 100,
    res_multi_vanilla["metrics"]["projected_binary"]["roc_auc"] * 100
]
vals_mal_mul = [
    res_multi_malignant["metrics"]["accuracy_3"] * 100,
    res_multi_malignant["metrics"]["macro_f1"] * 100,
    res_multi_malignant["metrics"]["qwk"] * 100,
    res_multi_malignant["metrics"]["class_metrics"]["Early"]["sensitivity"] * 100,
    res_multi_malignant["metrics"]["projected_binary"]["roc_auc"] * 100
]

x2 = np.arange(len(labels_mul))
ax_bar2.bar(x2 - width/2, vals_vanilla_mul, width, label="Vanilla (Sin Aumento)", color="#cbd5e1", edgecolor="#475569")
ax_bar2.bar(x2 + width/2, vals_mal_mul, width, label="Malignant Scale Augmentation", color="#7c3aed", edgecolor="#4c1d95")

ax_bar2.set_title("B. Beneficio del Data Augmentation en Formulación Multietapa (3 Clases)", fontsize=10.5, fontweight="bold")
ax_bar2.set_xticks(x2)
ax_bar2.set_xticklabels(labels_mul, fontsize=8.5, fontweight="bold")
ax_bar2.set_ylabel("Métrica Normalizada (%)", fontweight="bold")
ax_bar2.set_ylim([50, 105])
ax_bar2.legend(loc="lower right", frameon=True)
ax_bar2.grid(True, linestyle="--", alpha=0.4, axis="y")

for i, (v1, v2) in enumerate(zip(vals_vanilla_mul, vals_mal_mul)):
    diff = v2 - v1
    color = "#059669" if diff >= 0 else "#dc2626"
    ax_bar2.annotate(f"{diff:+.1f}%", (x2[i] + width/2, v2 + 1.2), ha="center", fontsize=8.5, fontweight="bold", color=color)

plt.suptitle("Figura 34: Impacto y Ganancia Diagnóstica del Data Augmentation Maligno en Clasificación de Glaucoma", fontsize=12, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig34_path = FIGURES_DIR / "fig34_data_augmentation_benefit_comparison.png"
plt.savefig(fig34_path, bbox_inches="tight")
plt.close()
print(f"[OK] Figura 34 guardada en: {fig34_path}")

# ---------------------------------------------------------
# Figura 35: Tipología de Error (Adyacente vs. Catastrófico)
# ---------------------------------------------------------
fig, (ax_err1, ax_err2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# Distribución de errores en 3 clases
err_van = res_multi_vanilla["metrics"]["errors"]
err_mal = res_multi_malignant["metrics"]["errors"]

bar_cats = ["Errores Totales", "Errores Adyacentes (Leves)", "Errores Catastróficos (Severos)"]
x_err = np.arange(len(bar_cats))
vals_e_van = [err_van["total"], err_van["adjacent"], err_van["catastrophic"]]
vals_e_mal = [err_mal["total"], err_mal["adjacent"], err_mal["catastrophic"]]

ax_err1.bar(x_err - width/2, vals_e_van, width, label=f"Vanilla (Catastrófico: {err_van['catastrophic_pct']}%)", color="#f87171", edgecolor="#991b1b")
ax_err1.bar(x_err + width/2, vals_e_mal, width, label=f"Malignant (Catastrófico: {err_mal['catastrophic_pct']}%)", color="#34d399", edgecolor="#065f46")

ax_err1.set_title("A. Número Absoluto de Errores por Severidad Clínica (N=200)", fontsize=10.5, fontweight="bold")
ax_err1.set_xticks(x_err)
ax_err1.set_xticklabels(bar_cats, fontsize=8.5, fontweight="bold")
ax_err1.set_ylabel("Cantidad de Muestras Mal Clasificadas", fontweight="bold")
ax_err1.legend(frameon=True)
ax_err1.grid(True, linestyle="--", alpha=0.4, axis="y")

for i, (v1, v2) in enumerate(zip(vals_e_van, vals_e_mal)):
    ax_err1.text(x_err[i] - width/2, v1 + 0.6, str(v1), ha="center", fontweight="bold", fontsize=9.5)
    ax_err1.text(x_err[i] + width/2, v2 + 0.6, str(v2), ha="center", fontweight="bold", fontsize=9.5)

# Pie chart comparativo de severidad de error
colors_pie = ["#38bdf8", "#ef4444"]
explode = (0, 0.1)

ax_err2.pie([err_mal["adjacent"], err_mal["catastrophic"]], 
            labels=[f"Adyacente ({err_mal['adjacent']})", f"Catastrófico Sano <-> Avanzado ({err_mal['catastrophic']})"],
            autopct="%1.1f%%", colors=colors_pie, explode=explode, startangle=140, 
            textprops={"fontweight": "bold", "fontsize": 9.5})
ax_err2.set_title("B. Composición Porcentual de Errores bajo Malignant Augmentation", fontsize=10.5, fontweight="bold")

plt.suptitle("Figura 35: Análisis de Seguridad Diagnóstica: Desglose de Errores Adyacentes vs. Catastróficos en Estadificación", fontsize=12, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig35_path = FIGURES_DIR / "fig35_error_distribution_adjacent_vs_catastrophic.png"
plt.savefig(fig35_path, bbox_inches="tight")
plt.close()
print(f"[OK] Figura 35 guardada en: {fig35_path}")

# ---------------------------------------------------------
# Figura 36: Perfilado de Severidad en Cohortes Externas
# ---------------------------------------------------------
fig, ax_prof = plt.subplots(figsize=(11, 6), dpi=300)

cohorts_ext = ["ACRIMA", "RIM-ONE DL", "EyePACS"]
pct_normal = [res_multi_malignant["external_cohorts"][c]["stage_distribution"]["pct_classified_normal"] for c in cohorts_ext]
pct_early = [res_multi_malignant["external_cohorts"][c]["stage_distribution"]["pct_classified_early"] for c in cohorts_ext]
pct_adv = [res_multi_malignant["external_cohorts"][c]["stage_distribution"]["pct_classified_advanced"] for c in cohorts_ext]

x_p = np.arange(len(cohorts_ext))
w_bar = 0.55

p1 = ax_prof.bar(x_p, pct_early, w_bar, label="Prematuro / Incipiente (Early)", color="#fbbf24", edgecolor="#78350f")
p2 = ax_prof.bar(x_p, pct_adv, w_bar, bottom=pct_early, label="Avanzado / Severo (Advanced)", color="#ef4444", edgecolor="#7f1d1d")
p0 = ax_prof.bar(x_p, pct_normal, w_bar, bottom=np.array(pct_early)+np.array(pct_adv), label="Falso Negativo (Clasificado Sano)", color="#94a3b8", edgecolor="#334155")

ax_prof.set_title("Figura 36: Perfilado Morfológico de Severidad en Cohortes Externas Independientes\nDistribución de Estadíos Detectados por el Modelo Multietapa sobre los Glaucomas Clínicos", fontsize=11, fontweight="bold", pad=12)
ax_prof.set_xticks(x_p)
ax_prof.set_xticklabels([f"{c}\n(N={external_cohorts[c]['n']})" for c in cohorts_ext], fontweight="bold", fontsize=10)
ax_prof.set_ylabel("Porcentaje de Casos Patológicos (%)", fontweight="bold")
ax_prof.set_ylim([0, 105])
ax_prof.legend(loc="upper right", frameon=True, fontsize=9)
ax_prof.grid(True, linestyle="--", alpha=0.4, axis="y")

# Anotaciones
for i in range(len(cohorts_ext)):
    e_val = pct_early[i]
    a_val = pct_adv[i]
    n_val = pct_normal[i]
    if e_val > 5:
        ax_prof.text(x_p[i], e_val / 2, f"{e_val:.1f}%", ha="center", va="center", color="black", fontweight="bold", fontsize=9.5)
    if a_val > 5:
        ax_prof.text(x_p[i], e_val + a_val / 2, f"{a_val:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
    if n_val > 5:
        ax_prof.text(x_p[i], e_val + a_val + n_val / 2, f"{n_val:.1f}%", ha="center", va="center", color="black", fontweight="bold", fontsize=9.5)

plt.tight_layout()
fig36_path = FIGURES_DIR / "fig36_external_cohorts_stage_profiling.png"
plt.savefig(fig36_path, bbox_inches="tight")
plt.close()
print(f"[OK] Figura 36 guardada en: {fig36_path}")

print("\n[EXITO COMPLETO] Experimento factorial 2x2 finalizado y 4 figuras científicas generadas.")
