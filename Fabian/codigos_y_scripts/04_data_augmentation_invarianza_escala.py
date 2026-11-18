import os, glob, random, time, json
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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[INFO] Dispositivo de computo: {DEVICE}")

# 1. Custom Malignant Scale Augmentation
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

            max_x = w_out - new_w
            max_y = h_out - new_h
            off_x = random.randint(0, max_x)
            off_y = random.randint(0, max_y)
            canvas.paste(img_scaled, (off_x, off_y))

            # Aperture vignette
            Y, X = np.ogrid[:h_out, :w_out]
            dist_from_center = np.sqrt((X - w_out/2)**2 + (Y - h_out/2)**2)
            radius = w_out * 0.48
            vignette = np.clip(1.0 - (dist_from_center - radius) / 8.0, 0.0, 1.0)
            canvas_arr = np.array(canvas, dtype=np.float32)
            for c in range(3):
                canvas_arr[:, :, c] *= vignette
            result = Image.fromarray(np.clip(canvas_arr, 0, 255).astype(np.uint8))

        elif mode == "zoom_in":
            scale = random.uniform(1.25, 1.85)
            orig_w, orig_h = img.size
            crop_w = int(orig_w / scale)
            crop_h = int(orig_h / scale)
            max_x = orig_w - crop_w
            max_y = orig_h - crop_h
            x0 = random.randint(int(max_x * 0.15), int(max_x * 0.85))
            y0 = random.randint(int(max_y * 0.15), int(max_y * 0.85))
            img_crop = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
            result = img_crop.resize((w_out, h_out), Image.BILINEAR)

        else:
            result = img.resize((w_out, h_out), Image.BILINEAR)

        # Photometric chaos
        if random.random() < 0.65:
            gamma = random.uniform(0.75, 1.35)
            arr = np.array(result, dtype=np.float32) / 255.0
            arr = np.power(arr, gamma)
            green_gain = random.uniform(0.90, 1.15)
            arr[:, :, 1] = np.clip(arr[:, :, 1] * green_gain, 0, 1)
            result = Image.fromarray((arr * 255).astype(np.uint8))

        return result

# 2. Datasets & Dataloaders
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
        label = torch.tensor(row["label"], dtype=torch.float32)
        return img, label

# Load ACRIMA metadata (Official label: '_g_' in filename indicates glaucoma)
acrima_files = sorted(glob.glob("data/Database/Images/*.jpg"))
acrima_rows = []
for f in acrima_files:
    fname = os.path.basename(f)
    lbl = 1 if "_g_" in fname.lower() else 0
    acrima_rows.append({"filepath": f, "label": lbl, "center": "ACRIMA"})
acrima_df = pd.DataFrame(acrima_rows)

# ACRIMA stratified 70/15/15
acr_train_val, acr_test = train_test_split(acrima_df, test_size=0.15, stratify=acrima_df["label"], random_state=SEED)
acr_train, acr_val = train_test_split(acr_train_val, test_size=0.1765, stratify=acr_train_val["label"], random_state=SEED)

# Load RIM-ONE DL metadata
rim_df = pd.read_csv("data/RIM_ONE_DL/metadata.csv")
rim_df["center"] = "RIM_ONE"
rim_train_val, rim_test = train_test_split(rim_df, test_size=0.15, stratify=rim_df["label"], random_state=SEED)
rim_train, rim_val = train_test_split(rim_train_val, test_size=0.1765, stratify=rim_train_val["label"], random_state=SEED)

# Combine training and validation
train_df = pd.concat([acr_train, rim_train], ignore_index=True)
val_df = pd.concat([acr_val, rim_val], ignore_index=True)

print(f"[DATA] Train total: {len(train_df)} (ACRIMA: {len(acr_train)}, RIM-ONE: {len(rim_train)})")
print(f"[DATA] Val total:   {len(val_df)} (ACRIMA: {len(acr_val)}, RIM-ONE: {len(rim_val)})")
print(f"[DATA] Test ACRIMA: {len(acr_test)} (Glaucoma: {sum(acr_test['label'])}, Normal: {len(acr_test)-sum(acr_test['label'])})")
print(f"[DATA] Test RIM-ONE: {len(rim_test)} (Glaucoma: {sum(rim_test['label'])}, Normal: {len(rim_test)-sum(rim_test['label'])})")

# 3. Pipelines
train_transform_malignant = transforms.Compose([
    MalignantScaleAugmentation(output_size=(224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(FundusDataset(train_df, transform=train_transform_malignant), batch_size=32, shuffle=True)
val_loader = DataLoader(FundusDataset(val_df, transform=val_transform), batch_size=32, shuffle=False)

# 4. Model Training
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 1)
model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

best_val_auc = 0.0
best_model_path = "modelos/best_resnet18_malignant_scale.pth"
epochs = 15

print("\n--- Iniciando Entrenamiento con Data Augmentation Maligno (Invariancia de Escala) ---")
train_losses, val_aucs = [], []

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    scheduler.step()

    train_loss = running_loss / len(train_df)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_probs, val_targets = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            probs = torch.sigmoid(model(imgs)).cpu().squeeze(1).numpy()
            val_probs.extend(probs)
            val_targets.extend(labels.numpy())

    val_auc = roc_auc_score(val_targets, val_probs)
    val_aucs.append(val_auc)
    print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val ROC-AUC: {val_auc:.4f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), best_model_path)
        print(f"  --> [CHECKPOINT] Mejor modelo guardado con Val AUC: {best_val_auc:.4f}")

print(f"\n[OK] Entrenamiento completado. Mejor Val AUC: {best_val_auc:.4f}")

# 5. Comprehensive Benchmark Evaluation
model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
model.to(DEVICE).eval()

def evaluate_split(model, df_or_paths, labels, name, transform):
    probs = []
    with torch.no_grad():
        for item in df_or_paths:
            im = Image.open(item).convert("RGB")
            t_im = transform(im).unsqueeze(0).to(DEVICE)
            prob = torch.sigmoid(model(t_im)).item()
            probs.append(prob)
    auc = roc_auc_score(labels, probs)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    acc = accuracy_score(labels, preds)
    sens = recall_score(labels, preds, zero_division=0)
    spec = recall_score(labels, preds, pos_label=0, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return {
        "dataset": name,
        "auc": float(auc),
        "acc": float(acc),
        "sens": float(sens),
        "spec": float(spec),
        "f1": float(f1),
        "cm": cm.tolist(),
        "probs": probs
    }

# Load EyePACS Real Test Set (200 images)
eyepacs_g = sorted(glob.glob("data/EYEPACS_REAL/glaucoma/*.png"))
eyepacs_ng = sorted(glob.glob("data/EYEPACS_REAL/normal/*.png"))
eyepacs_paths = eyepacs_g + eyepacs_ng
eyepacs_labels = [1]*len(eyepacs_g) + [0]*len(eyepacs_ng)

res_acrima = evaluate_split(model, acr_test["filepath"].tolist(), acr_test["label"].tolist(), "ACRIMA Test (n=106)", val_transform)
res_rimone = evaluate_split(model, rim_test["filepath"].tolist(), rim_test["label"].tolist(), "RIM-ONE Test (n=68)", val_transform)
res_eyepacs = evaluate_split(model, eyepacs_paths, eyepacs_labels, "EyePACS Real Test (N=200)", val_transform)

print("\n================ BENCHMARK DEL MODELO CON AUGMENTATION MALIGNO ================")
for r in [res_acrima, res_rimone, res_eyepacs]:
    print(f"Dataset: {r['dataset']:25s} | ROC-AUC: {r['auc']:.4f} | Acc: {r['acc']*100:.2f}% | Sens: {r['sens']*100:.2f}% | Spec: {r['spec']*100:.2f}%")
    print(f"  Matriz de Confusion: [[TN={r['cm'][0][0]}, FP={r['cm'][0][1]}], [FN={r['cm'][1][0]}, TP={r['cm'][1][1]}]]")

# Also evaluate EyePACS with a Center ROI crop (simulating an automated optic disc detector)
def eval_eyepacs_center_crop(model, paths, labels):
    probs = []
    with torch.no_grad():
        for p in paths:
            im = Image.open(p).convert("RGB")
            w, h = im.size
            crop_im = im.crop((int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)))
            t_im = val_transform(crop_im).unsqueeze(0).to(DEVICE)
            prob = torch.sigmoid(model(t_im)).item()
            probs.append(prob)
    auc = roc_auc_score(labels, probs)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    acc = accuracy_score(labels, preds)
    return float(auc), float(acc)

crop_auc, crop_acc = eval_eyepacs_center_crop(model, eyepacs_paths, eyepacs_labels)
print(f"EyePACS con Recorte Central ROI 50%      | ROC-AUC: {crop_auc:.4f} | Acc: {crop_acc*100:.2f}%")

# 6. Save JSON Summary
summary = {
    "experiment": "Malignant Scale and FOV Augmentation",
    "best_val_auc": float(best_val_auc),
    "acrima_test": res_acrima,
    "rimone_test": res_rimone,
    "eyepacs_test": res_eyepacs,
    "eyepacs_crop_auc": float(crop_auc),
    "eyepacs_crop_acc": float(crop_acc)
}
with open("results/resumen_experimento_malignant_augmentation.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n[OK] Resumen JSON guardado en results/resumen_experimento_malignant_augmentation.json")

# 7. Generate Diagnostic Figures
# Figure 11: Confusion matrices of the 3 test datasets
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200)
cms = [res_acrima["cm"], res_rimone["cm"], res_eyepacs["cm"]]
titles = [f"ACRIMA Test (n=106)\nAUC: {res_acrima['auc']:.4f} | Acc: {res_acrima['acc']*100:.1f}%",
          f"RIM-ONE Test (n=68)\nAUC: {res_rimone['auc']:.4f} | Acc: {res_rimone['acc']*100:.1f}%",
          f"EyePACS Real (N=200)\nAUC: {res_eyepacs['auc']:.4f} | Acc: {res_eyepacs['acc']*100:.1f}%"]
classes = ["Normal", "Glaucoma"]

for ax, cm, title in zip(axes, cms, titles):
    cm_arr = np.array(cm)
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_ylabel("Clase Real", fontsize=9, fontweight="bold")
    ax.set_xlabel("Predicción", fontsize=9, fontweight="bold")
    thresh = cm_arr.max() / 2.
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, f"{cm_arr[i, j]}", ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black", fontsize=11, fontweight="bold")

plt.suptitle("Matrices de Confusión del Modelo con Data Augmentation Maligno de Escala", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("figures/fig11_malignant_augmentation_confusion_matrices.png", dpi=200, bbox_inches="tight")
print("[OK] Guardado figures/fig11_malignant_augmentation_confusion_matrices.png")

# Figure 12: Comparison across the 3 paradigms
fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
categories = ["ACRIMA Test (Recorte)", "RIM-ONE Test (Recorte)", "EyePACS Real (Panorámica)", "EyePACS (Recorte ROI 50%)"]
x = np.arange(len(categories))
width = 0.25

# Values
base_vals = [0.9960, 0.4487, 0.5240, 0.5310]
multi_vals = [0.9986, 0.9219, 0.5197, 0.5420]
aug_vals = [res_acrima["auc"], res_rimone["auc"], res_eyepacs["auc"], crop_auc]

b1 = ax.bar(x - width, base_vals, width, label="Modelo Basal (Solo ACRIMA)", color="#e74c3c", alpha=0.85)
b2 = ax.bar(x, multi_vals, width, label="Multidominio Estándar", color="#3498db", alpha=0.85)
b3 = ax.bar(x + width, aug_vals, width, label="Augmentación Maligna de Escala", color="#2ecc71", alpha=0.85)

ax.axhline(0.50, color="gray", linestyle="--", alpha=0.7, label="Umbral Aleatorio (0.50)")
ax.set_ylabel("ROC-AUC", fontsize=11, fontweight="bold")
ax.set_title("Comparativa Tri-Céntrica: Impacto de la Invariancia de Escala Sintética", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9, fontweight="bold")
ax.set_ylim(0, 1.12)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", linestyle=":", alpha=0.6)

for bars in [b1, b2, b3]:
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig("figures/fig12_scale_invariance_comparison.png", dpi=200, bbox_inches="tight")
print("[OK] Guardado figures/fig12_scale_invariance_comparison.png")
