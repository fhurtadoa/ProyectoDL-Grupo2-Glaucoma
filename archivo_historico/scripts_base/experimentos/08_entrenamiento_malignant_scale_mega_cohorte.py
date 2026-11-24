#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo 08: Entrenamiento y Evaluación Bajo Protocolo de Estrés Maligno
(Malignant Scale Augmentation) sobre la Mega-Cohorte Médica Armonizada.

Especificaciones Técnicas:
1. Arquitectura: ResNet-18 pre-entrenada en ImageNet-1K (fc -> Linear(512, 1)).
2. Pérdida: BCEWithLogitsLoss.
3. Optimizador: AdamW (lr=1e-4, weight_decay=1e-4).
4. Scheduler: CosineAnnealingLR (T_max=15).
5. Augmentation: MalignantScaleAugmentation (Zoom-Out 45%, Zoom-In 35%, Photometric Chaos 65%).
6. Datos: data/HARMONIZED_MEGA_COHORT (THIRD_DATASET estrictamente excluido).
7. Checkpoint: modelos/best_resnet18_malignant_scale.pth (según ROC-AUC de validación).
8. Métricas: results/malignant_stress_test_metrics.json.
9. Visualización: figures/fig25_malignant_scale_stress_test_benchmark.png.
"""

import os, sys, glob, random, time, json
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
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Implementación Estricta de MalignantScaleAugmentation
# -----------------------------------------------------------------------------
class MalignantScaleAugmentation:
    """
    Protocolo de Data Augmentation Maligno para forzar invarianza a escala y campo visual:
    - 45% Zoom-Out severo (0.25x - 0.65x) con fondo ruidoso sintético y viñeteo óptico peripapilar.
    - 35% Zoom-In focal (1.25x - 1.85x) simulando micro-recorte peripapilar/copa óptica.
    - 20% Escala estándar sin alteración espacial.
    - 65% Perturbación fotométrica espectral: gamma U(0.75, 1.35) y ganancia verde U(0.90, 1.15).
    """
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size

    def __call__(self, img):
        mode = random.choices(["zoom_out", "zoom_in", "standard"], weights=[0.45, 0.35, 0.20])[0]
        w_out, h_out = self.output_size

        if mode == "zoom_out":
            scale = random.uniform(0.25, 0.65)
            new_w = max(24, int(w_out * scale))
            new_h = max(24, int(h_out * scale))
            img_scaled = img.resize((new_w, new_h), Image.BILINEAR)

            arr = np.array(img)
            edge_colors = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0)
            mean_bg = np.mean(edge_colors, axis=0).astype(np.float32)

            bg_arr = np.ones((h_out, w_out, 3), dtype=np.float32) * mean_bg
            noise = np.random.normal(0, 7, bg_arr.shape).astype(np.float32)
            bg_arr = np.clip(bg_arr + noise, 0, 255).astype(np.uint8)
            canvas = Image.fromarray(bg_arr)

            max_x = max(0, w_out - new_w)
            max_y = max(0, h_out - new_h)
            off_x = random.randint(0, max_x)
            off_y = random.randint(0, max_y)
            canvas.paste(img_scaled, (off_x, off_y))

            # Viñeteo circular óptico de apertura
            Y, X = np.ogrid[:h_out, :w_out]
            dist_from_center = np.sqrt((X - w_out / 2.0) ** 2 + (Y - h_out / 2.0) ** 2)
            radius = w_out * 0.48
            vignette = np.clip(1.0 - (dist_from_center - radius) / 8.0, 0.0, 1.0)
            canvas_arr = np.array(canvas, dtype=np.float32)
            for c in range(3):
                canvas_arr[:, :, c] *= vignette
            result = Image.fromarray(np.clip(canvas_arr, 0, 255).astype(np.uint8))

        elif mode == "zoom_in":
            scale = random.uniform(1.25, 1.85)
            orig_w, orig_h = img.size
            crop_w = max(10, int(orig_w / scale))
            crop_h = max(10, int(orig_h / scale))
            max_x = max(0, orig_w - crop_w)
            max_y = max(0, orig_h - crop_h)
            x_min = int(max_x * 0.15)
            x_max = int(max_x * 0.85)
            y_min = int(max_y * 0.15)
            y_max = int(max_y * 0.85)
            x0 = random.randint(min(x_min, x_max), max(x_min, x_max))
            y0 = random.randint(min(y_min, y_max), max(y_min, y_max))
            img_crop = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
            result = img_crop.resize((w_out, h_out), Image.BILINEAR)

        else:
            result = img.resize((w_out, h_out), Image.BILINEAR)

        # Perturbación fotométrica espectral
        if random.random() < 0.65:
            gamma = random.uniform(0.75, 1.35)
            arr = np.array(result, dtype=np.float32) / 255.0
            arr = np.power(np.clip(arr, 1e-6, 1.0), gamma)
            green_gain = random.uniform(0.90, 1.15)
            arr[:, :, 1] = np.clip(arr[:, :, 1] * green_gain, 0.0, 1.0)
            result = Image.fromarray((arr * 255).astype(np.uint8))

        return result

# -----------------------------------------------------------------------------
# 2. Dataset PyTorch
# -----------------------------------------------------------------------------
class HarmonizedFundusDataset(Dataset):
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
        label = torch.tensor(row["label"], dtype=torch.float32)
        return img, label

# -----------------------------------------------------------------------------
# 3. Función Principal de Entrenamiento y Evaluación de Estrés
# -----------------------------------------------------------------------------
def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[INFO] Dispositivo de cómputo seleccionado: {DEVICE}", flush=True)
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    RESULTS_DIR = BASE_DIR / "results"
    FIGURES_DIR = BASE_DIR / "figures"
    MODELS_DIR = BASE_DIR / "modelos"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Carga del catálogo maestro
    index_csv = RESULTS_DIR / "harmonized_mega_cohort_index.csv"
    if not index_csv.exists():
        raise FileNotFoundError(f"No se encontró el índice de cohorte en: {index_csv}")

    df_all = pd.read_csv(index_csv)
    print(f"[INFO] Catálogo maestro cargado: {len(df_all)} registros.", flush=True)

    # Verificación de exclusión de THIRD_DATASET
    assert "THIRD_DATASET" not in df_all["source"].values, "ALERTA: THIRD_DATASET detectado en el índice. Debe estar excluido."
    print("[OK] Verificación completada: THIRD_DATASET estrictamente excluido.", flush=True)

    # Resolver rutas de archivos
    def resolve_sample_path(row):
        p = Path(row["path"])
        if p.exists():
            return str(p)
        cat = "glaucoma" if row["label"] == 1 else "normal"
        local_p = DATA_DIR / "HARMONIZED_MEGA_COHORT" / row["split"] / cat / row["sample_id"]
        if local_p.exists():
            return str(local_p)
        raise FileNotFoundError(f"Imagen no localizada: {row['sample_id']}")

    df_all["filepath"] = df_all.apply(resolve_sample_path, axis=1)

    # Particiones Train/Val/Test
    df_train_pool = df_all[df_all["split"] == "train"].copy().reset_index(drop=True)
    df_test_pool = df_all[df_all["split"] == "test"].copy().reset_index(drop=True)

    # Validación estratificada por cohorte y etiqueta
    df_train_pool["strat_col"] = df_train_pool["source"] + "_" + df_train_pool["label"].astype(str)
    train_df, val_df = train_test_split(
        df_train_pool,
        test_size=0.15,
        stratify=df_train_pool["strat_col"],
        random_state=SEED
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"\n================ DISTRIBUCIÓN DE PARTICIONES MÉDICAS ================")
    print(f"Total Entrenamiento: {len(train_df)} imágenes | Glaucoma: {sum(train_df['label']==1)}, Normal: {sum(train_df['label']==0)}")
    print(f"Total Validación:    {len(val_df)} imágenes | Glaucoma: {sum(val_df['label']==1)}, Normal: {sum(val_df['label']==0)}")
    print(f"Total Test Held-Out: {len(df_test_pool)} imágenes | Glaucoma: {sum(df_test_pool['label']==1)}, Normal: {sum(df_test_pool['label']==0)}")

    test_cohorts = {
        "ACRIMA": df_test_pool[df_test_pool["source"] == "ACRIMA"].copy().reset_index(drop=True),
        "RIM-ONE DL": df_test_pool[df_test_pool["source"] == "RIM-ONE DL"].copy().reset_index(drop=True),
        "EyePACS": df_test_pool[df_test_pool["source"] == "EyePACS"].copy().reset_index(drop=True),
        "Moondream": df_test_pool[df_test_pool["source"] == "Moondream"].copy().reset_index(drop=True)
    }

    for c_name, c_df in test_cohorts.items():
        print(f"  - Cohorte Test {c_name:12s}: N={len(c_df):3d} (Glaucoma: {sum(c_df['label']==1):2d}, Normal: {sum(c_df['label']==0):2d})")

    # Pipelines de Transformación
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        MalignantScaleAugmentation(output_size=(224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # DataLoaders con num_workers=0 para garantizar estabilidad en Windows
    train_loader = DataLoader(HarmonizedFundusDataset(train_df, transform=train_transform), batch_size=32, shuffle=True, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(HarmonizedFundusDataset(val_df, transform=eval_transform), batch_size=32, shuffle=False, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)

    # 4. Arquitectura ResNet-18, Optimizador AdamW y Scheduler CosineAnnealingLR
    print("\n[INFO] Inicializando arquitectura ResNet-18 con pesos de ImageNet-1K...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

    # 5. Bucle de Entrenamiento (15 Épocas)
    EPOCHS = 15
    best_val_auc = -1.0
    best_model_path = MODELS_DIR / "best_resnet18_malignant_scale.pth"
    training_history = []

    print(f"\n================ INICIANDO ENTRENAMIENTO (15 ÉPOCAS COMPLETAS) ================", flush=True)
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total_train = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            total_train += imgs.size(0)

        scheduler.step()
        epoch_train_loss = running_loss / total_train

        # Validación
        model.eval()
        val_probs, val_targets = [], []
        val_running_loss = 0.0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                targets = labels.to(DEVICE).unsqueeze(1)
                logits = model(imgs)
                v_loss = criterion(logits, targets)

                probs = torch.sigmoid(logits).cpu().squeeze(1).numpy().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                val_probs.extend(probs)
                val_targets.extend(labels.numpy().tolist())

                val_running_loss += v_loss.item() * imgs.size(0)
                val_total += imgs.size(0)

        epoch_val_loss = val_running_loss / val_total
        epoch_val_auc = roc_auc_score(val_targets, val_probs)
        epoch_val_acc = accuracy_score(val_targets, [1 if p >= 0.5 else 0 for p in val_probs])

        training_history.append({
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": round(epoch_train_loss, 5),
            "val_loss": round(epoch_val_loss, 5),
            "val_auc": round(epoch_val_auc, 5),
            "val_acc": round(epoch_val_acc, 5)
        })

        is_best = epoch_val_auc > best_val_auc
        if is_best:
            best_val_auc = epoch_val_auc
            torch.save(model.state_dict(), best_model_path)
            star = "★ [BEST CHECKPOINT GUARDADO]"
        else:
            star = ""

        print(
            f"Época [{epoch:2d}/{EPOCHS:2d}] | Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val ROC-AUC: {epoch_val_auc:.4f} | "
            f"Val Acc: {epoch_val_acc*100:.2f}% {star}",
            flush=True
        )

    t_elapsed = time.time() - t_start
    print(f"\n[OK] Entrenamiento completado en {t_elapsed:.2f}s ({t_elapsed/60:.2f} min).")
    print(f"[OK] Mejor ROC-AUC de Validación alcanzado: {best_val_auc:.4f}")
    print(f"[OK] Checkpoint óptimo guardado en: {best_model_path}")

    # 6. Evaluación de la Matriz de Estrés en las 4 Cohortes Independientes
    print("\n================ EVALUACIÓN DE ESTRÉS EN COHORTES INDEPENDIENTES ================", flush=True)

    # Cargar checkpoint óptimo
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    def evaluate_cohort(sub_df, cohort_name):
        ds = HarmonizedFundusDataset(sub_df, transform=eval_transform)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        probs, targets = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(DEVICE)
                p = torch.sigmoid(model(imgs)).cpu().squeeze(1).numpy().tolist()
                if isinstance(p, float):
                    p = [p]
                probs.extend(p)
                targets.extend(labels.numpy().tolist())

        y_true = np.array(targets, dtype=int)
        y_prob = np.array(probs, dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)

        auc = float(roc_auc_score(y_true, y_prob))
        acc = float(accuracy_score(y_true, y_pred))
        sens = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        spec = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
        prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

        return {
            "cohort_name": cohort_name,
            "n_samples": len(sub_df),
            "n_glaucoma": int(sum(y_true == 1)),
            "n_normal": int(sum(y_true == 0)),
            "roc_auc": round(auc, 4),
            "accuracy": round(acc, 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {
                "TN": cm[0][0],
                "FP": cm[0][1],
                "FN": cm[1][0],
                "TP": cm[1][1]
            },
            "cm_raw": cm,
            "y_true": y_true.tolist(),
            "y_prob": y_prob.tolist()
        }

    # 1. ACRIMA (Canon 35°)
    res_acrima = evaluate_cohort(test_cohorts["ACRIMA"], "ACRIMA (Canon CR-DGi 35°)")
    # 2. RIM-ONE DL (Kowa 3D 35°)
    res_rimone = evaluate_cohort(test_cohorts["RIM-ONE DL"], "RIM-ONE DL (Kowa WX 3D Estéreo)")
    # 3. EyePACS (45° armonizado a 35°)
    res_eyepacs = evaluate_cohort(test_cohorts["EyePACS"], "EyePACS (45° Armonizado ROI)")
    # 4. Moondream (Estadíos Clínicos)
    res_moondream = evaluate_cohort(test_cohorts["Moondream"], "Moondream (Multicéntrico Estadificado)")
    # 5. Cohorte Global Pooled
    res_global = evaluate_cohort(df_test_pool, "Global Harmonized Mega-Cohort")

    # Análisis por estadío clínico en Moondream
    moon_test_df = test_cohorts["Moondream"].copy()
    moon_files = sorted(glob.glob(str(DATA_DIR / "MOONDREAM_STAGED/**/*.jpg"), recursive=True))
    stages = []
    for s_id in moon_test_df["sample_id"]:
        idx = int(s_id.replace("MOONDREAM_", "").replace(".jpg", ""))
        orig_path = moon_files[idx]
        if "early" in orig_path.lower():
            stages.append("early")
        elif "advanced" in orig_path.lower():
            stages.append("advanced")
        else:
            stages.append("normal")
    moon_test_df["stage"] = stages
    moon_test_df["y_prob"] = res_moondream["y_prob"]
    moon_test_df["y_pred"] = [1 if p >= 0.5 else 0 for p in res_moondream["y_prob"]]

    early_sub = moon_test_df[moon_test_df["stage"] == "early"]
    adv_sub = moon_test_df[moon_test_df["stage"] == "advanced"]
    norm_sub = moon_test_df[moon_test_df["stage"] == "normal"]

    early_sens = float((early_sub["y_pred"] == 1).mean()) if len(early_sub) > 0 else 0.0
    adv_sens = float((adv_sub["y_pred"] == 1).mean()) if len(adv_sub) > 0 else 0.0
    norm_spec = float((norm_sub["y_pred"] == 0).mean()) if len(norm_sub) > 0 else 0.0

    staged_metrics = {
        "early_glaucoma_sensitivity": round(early_sens, 4),
        "early_glaucoma_n": len(early_sub),
        "advanced_glaucoma_sensitivity": round(adv_sens, 4),
        "advanced_glaucoma_n": len(adv_sub),
        "normal_specificity": round(norm_spec, 4),
        "normal_n": len(norm_sub)
    }

    # Impresión formal en consola
    for r in [res_acrima, res_rimone, res_eyepacs, res_moondream, res_global]:
        cm = r["confusion_matrix"]
        print(
            f"\n[{r['cohort_name'].upper()}] (N={r['n_samples']})\n"
            f"  - ROC-AUC:       {r['roc_auc']:.4f}\n"
            f"  - Exactitud:     {r['accuracy']*100:.2f}%\n"
            f"  - Sensibilidad:  {r['sensitivity']*100:.2f}%\n"
            f"  - Especificidad: {r['specificity']*100:.2f}%\n"
            f"  - F1-Score:      {r['f1_score']:.4f}\n"
            f"  - Matriz:        [[TN={cm['TN']:2d}, FP={cm['FP']:2d}], [FN={cm['FN']:2d}, TP={cm['TP']:2d}]]"
        )

    print(
        f"\n[ESTADIFICACIÓN CLÍNICA MOONDREAM]\n"
        f"  - Sensibilidad Glaucoma Temprano (Early):    {early_sens*100:.2f}% ({sum(early_sub['y_pred']==1)}/{len(early_sub)})\n"
        f"  - Sensibilidad Glaucoma Avanzado (Advanced): {adv_sens*100:.2f}% ({sum(adv_sub['y_pred']==1)}/{len(adv_sub)})\n"
        f"  - Especificidad Control Sano (Normal):       {norm_spec*100:.2f}% ({sum(norm_sub['y_pred']==0)}/{len(norm_sub)})"
    )

    # 7. Serialización de Métricas a JSON
    metrics_payload = {
        "protocol": "Malignant Scale Augmentation Stress Testing",
        "backbone": "ResNet-18 (ImageNet-1K pretrained)",
        "loss_function": "BCEWithLogitsLoss",
        "optimizer": "AdamW (lr=1e-4, weight_decay=1e-4)",
        "scheduler": "CosineAnnealingLR (15 epochs)",
        "data_path": "data/HARMONIZED_MEGA_COHORT",
        "excluded_datasets": ["THIRD_DATASET (vessel edge skeletons)"],
        "best_checkpoint": str(best_model_path.relative_to(BASE_DIR)).replace("\\", "/"),
        "best_validation_roc_auc": round(best_val_auc, 4),
        "training_time_seconds": round(t_elapsed, 2),
        "cohort_benchmarks": {
            "acrima": {k: v for k, v in res_acrima.items() if k not in ["y_true", "y_prob", "cm_raw"]},
            "rim_one_dl": {k: v for k, v in res_rimone.items() if k not in ["y_true", "y_prob", "cm_raw"]},
            "eyepacs": {k: v for k, v in res_eyepacs.items() if k not in ["y_true", "y_prob", "cm_raw"]},
            "moondream": {k: v for k, v in res_moondream.items() if k not in ["y_true", "y_prob", "cm_raw"]},
            "global_mega_cohort": {k: v for k, v in res_global.items() if k not in ["y_true", "y_prob", "cm_raw"]}
        },
        "moondream_clinical_staging": staged_metrics,
        "training_history": training_history
    }

    json_output_path = RESULTS_DIR / "malignant_stress_test_metrics.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"\n[OK] Métricas completas exportadas exitosamente en: {json_output_path}")

    # 8. Generación de Figura de Estrés y Publicación (fig25)
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    # Panel 1: Curvas ROC Multicéntricas
    ax_roc = fig.add_subplot(gs[0, 0])
    colors = {"ACRIMA": "#e74c3c", "RIM-ONE DL": "#3498db", "EyePACS": "#2ecc71", "Moondream": "#9b59b6", "Global": "#2c3e50"}

    for r, name in zip(
        [res_acrima, res_rimone, res_eyepacs, res_moondream, res_global],
        ["ACRIMA", "RIM-ONE DL", "EyePACS", "Moondream", "Global"]
    ):
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        lw = 2.5 if name == "Global" else 1.8
        ls = "-" if name == "Global" else "--"
        ax_roc.plot(fpr, tpr, color=colors[name], lw=lw, linestyle=ls,
                    label=f"{name} (AUC={r['roc_auc']:.3f})")

    ax_roc.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("1 - Especificidad (Tasa Falsos Positivos)", fontweight="bold")
    ax_roc.set_ylabel("Sensibilidad (Tasa Verdaderos Positivos)", fontweight="bold")
    ax_roc.set_title("A. Curvas ROC Bajo Prueba de Estrés", fontweight="bold", fontsize=11)
    ax_roc.legend(loc="lower right", frameon=True, fontsize=8.5)
    ax_roc.grid(True, alpha=0.3)

    # Paneles 2-5: Matrices de Confusión por Sensor
    cohort_list = [
        (res_acrima, "B. Confusión: ACRIMA (Canon 35°)", gs[0, 1]),
        (res_rimone, "C. Confusión: RIM-ONE DL (Kowa 3D)", gs[0, 2]),
        (res_eyepacs, "D. Confusión: EyePACS (45° ROI)", gs[1, 0]),
        (res_moondream, "E. Confusión: Moondream (Estadíos)", gs[1, 1])
    ]

    for item, title, pos in cohort_list:
        ax_cm = fig.add_subplot(pos)
        cm_arr = np.array(item["cm_raw"])
        im = ax_cm.imshow(cm_arr, interpolation="nearest", cmap="Blues")
        ax_cm.set_title(title, fontweight="bold", fontsize=10)
        tick_marks = np.arange(2)
        ax_cm.set_xticks(tick_marks)
        ax_cm.set_xticklabels(["Sano (0)", "Glaucoma (1)"], fontweight="bold", fontsize=8.5)
        ax_cm.set_yticks(tick_marks)
        ax_cm.set_yticklabels(["Sano (0)", "Glaucoma (1)"], fontweight="bold", fontsize=8.5)
        ax_cm.set_xlabel("Predicción del Modelo", fontweight="bold")
        ax_cm.set_ylabel("Etiqueta Clínica Real", fontweight="bold")

        thresh = cm_arr.max() / 2.0
        total = cm_arr.sum()
        for i in range(2):
            for j in range(2):
                val = cm_arr[i, j]
                pct = (val / total) * 100
                ax_cm.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                           color="white" if val > thresh else "black", fontweight="bold", fontsize=10)

    # Panel 6: Rendimiento Comparativo en Barras Agrupadas
    ax_bar = fig.add_subplot(gs[1, 2])
    metrics_names = ["ROC-AUC", "Exactitud", "Sensibilidad", "Especificidad", "F1-Score"]
    cohort_names = ["ACRIMA", "RIM-ONE DL", "EyePACS", "Moondream"]
    c_objs = [res_acrima, res_rimone, res_eyepacs, res_moondream]

    x = np.arange(len(metrics_names))
    width = 0.18

    for idx, (c_name, c_res) in enumerate(zip(cohort_names, c_objs)):
        vals = [c_res["roc_auc"], c_res["accuracy"], c_res["sensitivity"], c_res["specificity"], c_res["f1_score"]]
        ax_bar.bar(x + idx * width, vals, width, label=c_name, color=colors[c_name], alpha=0.85)

    ax_bar.set_ylabel("Puntuación (0.0 - 1.0)", fontweight="bold")
    ax_bar.set_title("F. Métricas Comparativas Multicéntricas", fontweight="bold", fontsize=11)
    ax_bar.set_xticks(x + width * 1.5)
    ax_bar.set_xticklabels(metrics_names, fontweight="bold", fontsize=8.5)
    ax_bar.set_ylim([0.4, 1.05])
    ax_bar.legend(loc="lower right", frameon=True, fontsize=8)
    ax_bar.grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        "ResNet-18 + Malignant Scale Augmentation: Evaluación de Estrés en la Mega-Cohorte Armonizada\n"
        "Invarianza a Escalas Extremas (0.25x - 1.85x), Perturbación Fotométrica y Generalización Inter-Cámara",
        fontsize=13, fontweight="bold", y=0.98
    )

    fig_path = FIGURES_DIR / "fig25_malignant_scale_stress_test_benchmark.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figura benchmark de alta resolución guardada en: {fig_path}")
    print("================================================================================")

if __name__ == "__main__":
    main()
