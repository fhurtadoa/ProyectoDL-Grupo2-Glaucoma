#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo 09: Experimento Científico de Sobre-Entrenamiento Extremo (1000 Épocas)
Evaluación del Comportamiento Asintótico, Curvas de Pérdida, Fenómeno de Sobreajuste
y Resiliencia de Representaciones Bajo Malignant Scale Augmentation en la Mega-Cohorte Médica.

Hipótesis Científica:
¿El aumento de datos estocástico agresivo (Malignant Scale Augmentation) actúa como
un regularizador implícito infinito que previene la memorización catastrófica, o la red
ResNet-18 colapsa por sobreajuste tras 1000 épocas de optimización continua?
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
# 1. Reproducibilidad y Configuración
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Dispositivo seleccionado: {DEVICE}", flush=True)
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

# -----------------------------------------------------------------------------
# 2. Malignant Scale Augmentation
# -----------------------------------------------------------------------------
class MalignantScaleAugmentation:
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
            x_min, x_max = int(max_x * 0.15), int(max_x * 0.85)
            y_min, y_max = int(max_y * 0.15), int(max_y * 0.85)
            x0 = random.randint(min(x_min, x_max), max(x_min, x_max))
            y0 = random.randint(min(y_min, y_max), max(y_min, y_max))
            img_crop = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
            result = img_crop.resize((w_out, h_out), Image.BILINEAR)

        else:
            result = img.resize((w_out, h_out), Image.BILINEAR)

        if random.random() < 0.65:
            gamma = random.uniform(0.75, 1.35)
            arr = np.array(result, dtype=np.float32) / 255.0
            arr = np.power(np.clip(arr, 1e-6, 1.0), gamma)
            green_gain = random.uniform(0.90, 1.15)
            arr[:, :, 1] = np.clip(arr[:, :, 1] * green_gain, 0.0, 1.0)
            result = Image.fromarray((arr * 255).astype(np.uint8))

        return result

# -----------------------------------------------------------------------------
# 3. Dataset Rápido con Precarga en RAM
# -----------------------------------------------------------------------------
class FastPreloadedDataset(Dataset):
    def __init__(self, filepaths, labels, transform):
        self.transform = transform
        self.labels = [torch.tensor(lbl, dtype=torch.float32) for lbl in labels]
        print(f"[INFO] Precargando {len(filepaths)} imágenes en memoria RAM para acelerar 1000 épocas...", flush=True)
        self.images = []
        for p in filepaths:
            self.images.append(Image.open(p).convert("RGB"))
        print("[OK] Todas las imágenes precargadas en RAM exitosamente.", flush=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = self.images[idx]
        return self.transform(im), self.labels[idx]

def main():
    # Cargar catálogo
    index_csv = RESULTS_DIR / "harmonized_mega_cohort_index.csv"
    df_all = pd.read_csv(index_csv)

    def resolve_sample_path(row):
        p = Path(row["path"])
        if p.exists():
            return str(p)
        cat = "glaucoma" if row["label"] == 1 else "normal"
        local_p = DATA_DIR / "HARMONIZED_MEGA_COHORT" / row["split"] / cat / row["sample_id"]
        if local_p.exists():
            return str(local_p)
        raise FileNotFoundError(f"No encontrada: {row['sample_id']}")

    df_all["filepath"] = df_all.apply(resolve_sample_path, axis=1)

    df_train_pool = df_all[df_all["split"] == "train"].copy().reset_index(drop=True)
    df_test_pool = df_all[df_all["split"] == "test"].copy().reset_index(drop=True)

    # Validación estratificada
    df_train_pool["strat_col"] = df_train_pool["source"] + "_" + df_train_pool["label"].astype(str)
    train_df, val_df = train_test_split(
        df_train_pool,
        test_size=0.15,
        stratify=df_train_pool["strat_col"],
        random_state=SEED
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"\n================ CONFIGURACIÓN EXPERIMENTO 1000 ÉPOCAS ================")
    print(f"Total Entrenamiento: {len(train_df)} | Glaucoma: {sum(train_df['label']==1)}, Normal: {sum(train_df['label']==0)}")
    print(f"Total Validación:    {len(val_df)} | Glaucoma: {sum(val_df['label']==1)}, Normal: {sum(val_df['label']==0)}")
    print(f"Total Test Held-Out: {len(df_test_pool)}")

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

    # Precarga en RAM
    train_dataset = FastPreloadedDataset(train_df["filepath"].tolist(), train_df["label"].tolist(), train_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)

    # Precargar Validación directamente en GPU como Tensor único para validación instantánea (< 10 ms)
    val_tensors = []
    for p in val_df["filepath"]:
        im = Image.open(p).convert("RGB")
        val_tensors.append(eval_transform(im))
    val_x = torch.stack(val_tensors).to(DEVICE)
    val_y = torch.tensor(val_df["label"].tolist(), dtype=torch.float32, device=DEVICE).unsqueeze(1)
    val_targets_np = val_df["label"].to_numpy()

    print("[OK] Tensores de validación transferidos a GPU para inferencia ultra-rápida.", flush=True)

    # 4. Inicializar Red ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    model.to(DEVICE)

    TOTAL_EPOCHS = 1000
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # Checkpoints y Métricas
    best_val_auc = -1.0
    best_epoch = -1
    best_model_path = MODELS_DIR / "best_resnet18_1000ep_val_auc.pth"
    final_model_path = MODELS_DIR / "final_resnet18_1000ep.pth"

    training_log = []
    checkpoint_milestones = {}

    print(f"\n================ INICIANDO 1000 ÉPOCAS DE APRENDIZAJE PROFUNDO ================", flush=True)
    t_start = time.time()

    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        epoch_train_loss = running_loss / n_samples

        # Validación rápida en GPU
        model.eval()
        with torch.no_grad():
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    val_logits = model(val_x)
                    val_loss = criterion(val_logits, val_y).item()
            else:
                val_logits = model(val_x)
                val_loss = criterion(val_logits, val_y).item()

            val_probs = torch.sigmoid(val_logits).cpu().squeeze(1).numpy()

        val_auc = float(roc_auc_score(val_targets_np, val_probs))
        val_acc = float(accuracy_score(val_targets_np, (val_probs >= 0.5).astype(int)))

        # Guardar mejor modelo histórico
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            is_best_str = f"★ [MEJOR AUC HISTÓRICO: {best_val_auc:.4f}]"
        else:
            is_best_str = ""

        # Registrar hito en épocas clave
        if epoch in [1, 5, 15, 50, 100, 250, 500, 750, 1000]:
            ms_path = MODELS_DIR / f"checkpoint_resnet18_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ms_path)
            checkpoint_milestones[epoch] = {
                "epoch": epoch,
                "train_loss": round(epoch_train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_auc": round(val_auc, 5),
                "val_acc": round(val_acc, 5),
                "path": str(ms_path)
            }

        # Guardar registro cada época
        training_log.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": round(epoch_train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_auc": round(val_auc, 5),
            "val_acc": round(val_acc, 5)
        })

        # Impresión periódica (cada 10 épocas, o primeras 15)
        if epoch <= 15 or epoch % 25 == 0 or epoch == TOTAL_EPOCHS:
            elapsed = time.time() - t_start
            eta = (elapsed / epoch) * (TOTAL_EPOCHS - epoch)
            print(
                f"Época [{epoch:4d}/{TOTAL_EPOCHS:4d}] | LR: {current_lr:.2e} | "
                f"Train Loss: {epoch_train_loss:.5f} | Val Loss: {val_loss:.5f} | "
                f"Val ROC-AUC: {val_auc:.4f} | Val Acc: {val_acc*100:.2f}% | "
                f"ETA: {eta/60:.1f} min {is_best_str}",
                flush=True
            )

    t_total = time.time() - t_start
    torch.save(model.state_dict(), final_model_path)
    print(f"\n[OK] Entrenamiento de 1000 épocas finalizado en {t_total:.2f}s ({t_total/60:.2f} min).")
    print(f"[OK] Mejor ROC-AUC histórico: {best_val_auc:.4f} en Época {best_epoch}")

    # 5. Evaluación de los 3 Modelos Clave en las 4 Cohortes de Prueba
    # Modelo 1: 15 Épocas (Línea Base inicial)
    # Modelo 2: Mejor Checkpoint histórico de las 1000 épocas
    # Modelo 3: Modelo Final Época 1000 (Régimen asintótico de sobreajuste)
    print("\n================ EVALUACIÓN COMPARATIVA EN COHORTES INDEPENDIENTES ================", flush=True)

    def eval_on_test(state_dict):
        m = models.resnet18()
        m.fc = nn.Linear(512, 1)
        m.load_state_dict(state_dict)
        m.to(DEVICE).eval()

        probs, targets, sources = [], [], []
        with torch.no_grad():
            for idx in range(len(df_test_pool)):
                row = df_test_pool.iloc[idx]
                im = Image.open(row["filepath"]).convert("RGB")
                t_im = eval_transform(im).unsqueeze(0).to(DEVICE)
                prob = torch.sigmoid(m(t_im)).item()
                probs.append(prob)
                targets.append(row["label"])
                sources.append(row["source"])

        y_true = np.array(targets)
        y_prob = np.array(probs)
        y_pred = (y_prob >= 0.5).astype(int)

        cohort_metrics = {}
        for c in ["ACRIMA", "RIM-ONE DL", "EyePACS", "Moondream"]:
            mask = np.array(sources) == c
            c_true = y_true[mask]
            c_prob = y_prob[mask]
            c_pred = y_pred[mask]
            cohort_metrics[c] = {
                "auc": round(float(roc_auc_score(c_true, c_prob)), 4),
                "acc": round(float(accuracy_score(c_true, c_pred)), 4),
                "sens": round(float(recall_score(c_true, c_pred, pos_label=1, zero_division=0)), 4),
                "spec": round(float(recall_score(c_true, c_pred, pos_label=0, zero_division=0)), 4),
                "f1": round(float(f1_score(c_true, c_pred, zero_division=0)), 4),
                "cm": confusion_matrix(c_true, c_pred, labels=[0, 1]).tolist()
            }

        global_auc = round(float(roc_auc_score(y_true, y_prob)), 4)
        global_acc = round(float(accuracy_score(y_true, y_pred)), 4)
        global_sens = round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4)
        global_spec = round(float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)), 4)
        global_f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
        global_cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

        return {
            "global": {
                "auc": global_auc, "acc": global_acc, "sens": global_sens,
                "spec": global_spec, "f1": global_f1, "cm": global_cm
            },
            "cohorts": cohort_metrics,
            "y_prob": y_prob.tolist(),
            "y_true": y_true.tolist()
        }

    # Cargar los 3 estados
    ckpt_15 = torch.load(MODELS_DIR / "checkpoint_resnet18_epoch_15.pth", map_location=DEVICE)
    ckpt_best = torch.load(best_model_path, map_location=DEVICE)
    ckpt_1000 = torch.load(final_model_path, map_location=DEVICE)

    print("Evaluando Checkpoint Época 15...")
    res_15 = eval_on_test(ckpt_15)
    print("Evaluando Mejor Checkpoint Histórico...")
    res_best = eval_on_test(ckpt_best)
    print("Evaluando Checkpoint Final Época 1000...")
    res_1000 = eval_on_test(ckpt_1000)

    # 6. Guardar Resultados en JSON
    results_payload = {
        "experiment": "1000 Epochs Asymptotic Stress & Overfitting Curiosity Experiment",
        "total_epochs": TOTAL_EPOCHS,
        "training_time_seconds": round(t_total, 2),
        "best_epoch": best_epoch,
        "best_val_auc": round(best_val_auc, 5),
        "milestones": checkpoint_milestones,
        "eval_comparison": {
            "epoch_15": {k: v for k, v in res_15.items() if k not in ["y_prob", "y_true"]},
            "epoch_best": {k: v for k, v in res_best.items() if k not in ["y_prob", "y_true"]},
            "epoch_1000": {k: v for k, v in res_1000.items() if k not in ["y_prob", "y_true"]}
        },
        "training_history": training_log
    }

    results_json_path = RESULTS_DIR / "1000_epochs_stress_experiment_metrics.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=4)
    print(f"[OK] Métricas del experimento guardadas en: {results_json_path}")

    # 7. Generar Figuras de Publicación Científica (fig26 y fig27)
    df_history = pd.DataFrame(training_log)

    # Figura 26: Curvas de Pérdida y ROC-AUC a lo largo de las 1000 Épocas
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)

    # 1. Pérdidas en Escala Lineal
    axes[0, 0].plot(df_history["epoch"], df_history["train_loss"], label="Train Loss (Malignant Aug)", color="#1e40af", lw=1.8)
    axes[0, 0].plot(df_history["epoch"], df_history["val_loss"], label="Val Loss (Clean Baseline)", color="#e11d48", lw=1.8)
    axes[0, 0].axvline(best_epoch, color="#059669", linestyle="--", label=f"Mejor Val AUC (Época {best_epoch})", lw=1.5)
    axes[0, 0].axvline(15, color="#d97706", linestyle=":", label="Corte Convencional (Época 15)", lw=1.5)
    axes[0, 0].set_title("A. Evolución de Pérdidas de Entrenamiento vs. Validación (1000 Épocas)", fontweight="bold", fontsize=10)
    axes[0, 0].set_xlabel("Época de Entrenamiento", fontweight="bold")
    axes[0, 0].set_ylabel("Pérdida Binaria (BCE)", fontweight="bold")
    axes[0, 0].legend(frameon=True, fontsize=8.5)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Pérdidas en Escala Logarítmica (Para observar memorización extrema)
    axes[0, 1].plot(df_history["epoch"], df_history["train_loss"], label="Train Loss (Log Scale)", color="#1e40af", lw=1.8)
    axes[0, 1].plot(df_history["epoch"], df_history["val_loss"], label="Val Loss (Log Scale)", color="#e11d48", lw=1.8)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("B. Dinámica de Pérdidas en Escala Logarítmica (Convergencia Asintótica)", fontweight="bold", fontsize=10)
    axes[0, 1].set_xlabel("Época de Entrenamiento", fontweight="bold")
    axes[0, 1].set_ylabel("Log(BCE Loss)", fontweight="bold")
    axes[0, 1].legend(frameon=True, fontsize=8.5)
    axes[0, 1].grid(True, alpha=0.3, which="both")

    # 3. Trayectoria de ROC-AUC en Validación
    axes[1, 0].plot(df_history["epoch"], df_history["val_auc"], color="#059669", lw=2.0, label="Val ROC-AUC")
    # Media móvil suave
    axes[1, 0].plot(df_history["epoch"], df_history["val_auc"].rolling(20, min_periods=1).mean(), color="#047857", lw=1.2, linestyle="--", label="Media Móvil (20 ep)")
    axes[1, 0].axhline(best_val_auc, color="#dc2626", linestyle=":", label=f"Máximo Histórico = {best_val_auc:.4f}")
    axes[1, 0].set_title("C. Trayectoria de Capacidad Discriminativa (Val ROC-AUC)", fontweight="bold", fontsize=10)
    axes[1, 0].set_xlabel("Época de Entrenamiento", fontweight="bold")
    axes[1, 0].set_ylabel("ROC-AUC", fontweight="bold")
    axes[1, 0].set_ylim([0.80, 1.0])
    axes[1, 0].legend(frameon=True, loc="lower right", fontsize=8.5)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Tasa de Aprendizaje y Exactitud
    ax2 = axes[1, 1].twinx()
    l1 = axes[1, 1].plot(df_history["epoch"], df_history["val_acc"]*100, color="#7c3aed", lw=1.8, label="Exactitud Validación (%)")
    l2 = ax2.plot(df_history["epoch"], df_history["lr"], color="#64748b", lw=1.5, linestyle="--", label="Cosine Annealing LR")
    axes[1, 1].set_title("D. Exactitud de Validación vs. Calendario Cosine Annealing", fontweight="bold", fontsize=10)
    axes[1, 1].set_xlabel("Época de Entrenamiento", fontweight="bold")
    axes[1, 1].set_ylabel("Exactitud (%)", fontweight="bold", color="#7c3aed")
    ax2.set_ylabel("Tasa de Aprendizaje (LR)", fontweight="bold", color="#64748b")
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    axes[1, 1].legend(lines, labels, frameon=True, loc="lower right", fontsize=8.5)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Experimento 1000 Épocas: Dinámica de Optimización Extrema bajo Malignant Scale Augmentation\nResNet-18 en la Mega-Cohorte Armonizada de Glaucoma", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig26_path = FIGURES_DIR / "fig26_1000_epochs_loss_and_auc_curves.png"
    plt.savefig(fig26_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figura 26 guardada en: {fig26_path}")

    # Figura 27: Comparativa de Rendimiento entre Modelos (Época 15 vs. Mejor vs. Época 1000)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

    # 1. Comparativa Global
    models_names = ["Época 15 (Convencional)", f"Mejor Época ({best_epoch})", "Época 1000 (Asintótico)"]
    metrics_list = ["ROC-AUC", "Exactitud", "Sensibilidad", "Especificidad", "F1-Score"]

    m15_vals = [res_15["global"]["auc"], res_15["global"]["acc"], res_15["global"]["sens"], res_15["global"]["spec"], res_15["global"]["f1"]]
    mbest_vals = [res_best["global"]["auc"], res_best["global"]["acc"], res_best["global"]["sens"], res_best["global"]["spec"], res_best["global"]["f1"]]
    m1000_vals = [res_1000["global"]["auc"], res_1000["global"]["acc"], res_1000["global"]["sens"], res_1000["global"]["spec"], res_1000["global"]["f1"]]

    x = np.arange(len(metrics_list))
    width = 0.26

    axes[0].bar(x - width, m15_vals, width, label="Época 15 (Corte estándar)", color="#d97706", alpha=0.85)
    axes[0].bar(x, mbest_vals, width, label=f"Mejor Checkpoint (Ép. {best_epoch})", color="#059669", alpha=0.9)
    axes[0].bar(x + width, m1000_vals, width, label="Época 1000 (Régimen Extremo)", color="#4f46e5", alpha=0.85)

    axes[0].set_ylabel("Puntuación (0.0 - 1.0)", fontweight="bold")
    axes[0].set_title("A. Rendimiento Global en Test Set (N=431): Época 15 vs. Mejor vs. 1000", fontweight="bold", fontsize=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_list, fontweight="bold", fontsize=8.5)
    axes[0].set_ylim([0.65, 1.02])
    axes[0].legend(frameon=True, loc="lower right", fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)

    # 2. Desglose de ROC-AUC por Cohorte Independiente
    cohorts_names = ["ACRIMA", "RIM-ONE DL", "EyePACS", "Moondream"]
    c15_auc = [res_15["cohorts"][c]["auc"] for c in cohorts_names]
    cbest_auc = [res_best["cohorts"][c]["auc"] for c in cohorts_names]
    c1000_auc = [res_1000["cohorts"][c]["auc"] for c in cohorts_names]

    xc = np.arange(len(cohorts_names))
    axes[1].bar(xc - width, c15_auc, width, label="Época 15", color="#d97706", alpha=0.85)
    axes[1].bar(xc, cbest_auc, width, label=f"Mejor Época ({best_epoch})", color="#059669", alpha=0.9)
    axes[1].bar(xc + width, c1000_auc, width, label="Época 1000", color="#4f46e5", alpha=0.85)

    axes[1].set_ylabel("ROC-AUC", fontweight="bold")
    axes[1].set_title("B. ROC-AUC en las 4 Cohortes Clínicas Independientes", fontweight="bold", fontsize=10)
    axes[1].set_xticks(xc)
    axes[1].set_xticklabels(cohorts_names, fontweight="bold", fontsize=8.5)
    axes[1].set_ylim([0.65, 1.02])
    axes[1].legend(frameon=True, loc="lower right", fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.suptitle("Impacto del Entrenamiento Ultra-Extenso en Generalización Inter-Cámara", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig27_path = FIGURES_DIR / "fig27_1000_epochs_cross_cohort_comparison.png"
    plt.savefig(fig27_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figura 27 guardada en: {fig27_path}")

    print("\n================ SÍNTESIS CIENTÍFICA DEL EXPERIMENTO ================")
    print(f"Modelo Época 15   | Global ROC-AUC: {res_15['global']['auc']:.4f} | Exactitud: {res_15['global']['acc']*100:.2f}% | F1: {res_15['global']['f1']:.4f}")
    print(f"Mejor Época ({best_epoch:3d}) | Global ROC-AUC: {res_best['global']['auc']:.4f} | Exactitud: {res_best['global']['acc']*100:.2f}% | F1: {res_best['global']['f1']:.4f}")
    print(f"Modelo Época 1000 | Global ROC-AUC: {res_1000['global']['auc']:.4f} | Exactitud: {res_1000['global']['acc']*100:.2f}% | F1: {res_1000['global']['f1']:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
