#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluación de estrés en tiempo de inferencia:
Compara el rendimiento en el Test Set con entrada limpia vs entrada sometida
a MalignantScaleAugmentation (degradación estocástica de escala y fotometría).
"""

import os, random, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def run_evaluation():
    model = models.resnet18()
    model.fc = torch.nn.Linear(512, 1)
    ckpt = BASE_DIR / "modelos" / "best_resnet18_malignant_scale.pth"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.to(DEVICE).eval()

    df_all = pd.read_csv(BASE_DIR / "results" / "harmonized_mega_cohort_index.csv")
    test_df = df_all[df_all["split"] == "test"].copy().reset_index(drop=True)

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    clean_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    malignant_tf = transforms.Compose([
        MalignantScaleAugmentation(output_size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    def eval_pipeline(tf, name):
        probs, targets, sources = [], [], []
        with torch.no_grad():
            for idx in range(len(test_df)):
                row = test_df.iloc[idx]
                p = Path(row["path"])
                if not p.exists():
                    p = BASE_DIR / "data" / "HARMONIZED_MEGA_COHORT" / row["split"] / ("glaucoma" if row["label"]==1 else "normal") / row["sample_id"]
                img = Image.open(p).convert("RGB")
                t_img = tf(img).unsqueeze(0).to(DEVICE)
                prob = torch.sigmoid(model(t_img)).item()
                probs.append(prob)
                targets.append(row["label"])
                sources.append(row["source"])

        y_true = np.array(targets)
        y_prob = np.array(probs)
        y_pred = (y_prob >= 0.5).astype(int)

        global_auc = float(roc_auc_score(y_true, y_prob))
        global_acc = float(accuracy_score(y_true, y_pred))
        global_sens = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        global_spec = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
        global_f1 = float(f1_score(y_true, y_pred, zero_division=0))

        cohorts = {}
        for c in ["ACRIMA", "RIM-ONE DL", "EyePACS", "Moondream"]:
            mask = np.array(sources) == c
            c_auc = float(roc_auc_score(y_true[mask], y_prob[mask]))
            c_acc = float(accuracy_score(y_true[mask], y_pred[mask]))
            c_sens = float(recall_score(y_true[mask], y_pred[mask], pos_label=1, zero_division=0))
            c_spec = float(recall_score(y_true[mask], y_pred[mask], pos_label=0, zero_division=0))
            c_f1 = float(f1_score(y_true[mask], y_pred[mask], zero_division=0))
            cm = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1]).tolist()
            cohorts[c] = {
                "auc": round(c_auc, 4),
                "acc": round(c_acc, 4),
                "sens": round(c_sens, 4),
                "spec": round(c_spec, 4),
                "f1": round(c_f1, 4),
                "cm": cm
            }

        return {
            "regime": name,
            "global_auc": round(global_auc, 4),
            "global_acc": round(global_acc, 4),
            "global_sens": round(global_sens, 4),
            "global_spec": round(global_spec, 4),
            "global_f1": round(global_f1, 4),
            "cohorts": cohorts
        }

    print("Evaluando Test Set Limpio (Baseline)...")
    res_clean = eval_pipeline(clean_tf, "Limpio / Estandar")

    print("Evaluando Test Set con Inferencia Bajo Malignant Scale Augmentation (Seed 42)...")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    res_mal_42 = eval_pipeline(malignant_tf, "Malignant Augmentation (Run 1)")

    print("Evaluando Test Set con Inferencia Bajo Malignant Scale Augmentation (Seed 123)...")
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    res_mal_123 = eval_pipeline(malignant_tf, "Malignant Augmentation (Run 2)")

    comparison = {
        "clean_baseline": res_clean,
        "malignant_test_run1": res_mal_42,
        "malignant_test_run2": res_mal_123
    }

    out_json = BASE_DIR / "results" / "inference_time_malignant_stress_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=4)
    print(f"[OK] Comparativa guardada en: {out_json}")

    print("\n" + "="*80)
    print(f"{'MÉTRICA':<25} | {'TEST LIMPIO':<15} | {'TEST CON MALIGNANT AUG (R1)':<25} | {'TEST CON MALIGNANT AUG (R2)'}")
    print("="*80)
    print(f"{'Global ROC-AUC':<25} | {res_clean['global_auc']:<15.4f} | {res_mal_42['global_auc']:<25.4f} | {res_mal_123['global_auc']:.4f}")
    print(f"{'Global Exactitud':<25} | {res_clean['global_acc']*100:<14.2f}% | {res_mal_42['global_acc']*100:<24.2f}% | {res_mal_123['global_acc']*100:.2f}%")
    print(f"{'Global Sensibilidad':<25} | {res_clean['global_sens']*100:<14.2f}% | {res_mal_42['global_sens']*100:<24.2f}% | {res_mal_123['global_sens']*100:.2f}%")
    print(f"{'Global Especificidad':<25} | {res_clean['global_spec']*100:<14.2f}% | {res_mal_42['global_spec']*100:<24.2f}% | {res_mal_123['global_spec']*100:.2f}%")
    print(f"{'Global F1-Score':<25} | {res_clean['global_f1']:<15.4f} | {res_mal_42['global_f1']:<25.4f} | {res_mal_123['global_f1']:.4f}")
    print("-"*80)
    for c in ["ACRIMA", "RIM-ONE DL", "EyePACS", "Moondream"]:
        print(f"ROC-AUC {c:<17} | {res_clean['cohorts'][c]['auc']:<15.4f} | {res_mal_42['cohorts'][c]['auc']:<25.4f} | {res_mal_123['cohorts'][c]['auc']:.4f}")
    print("="*80)

if __name__ == "__main__":
    run_evaluation()
