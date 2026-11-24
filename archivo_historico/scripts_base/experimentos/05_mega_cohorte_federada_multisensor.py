import os, glob, random, json
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
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[INFO] Dispositivo de computo: {DEVICE}")

# 1. Gather all 4 cohorts for training & validation
# Cohort 1: ACRIMA
acrima_files = sorted(glob.glob("data/Database/Images/*.jpg"))
acr_rows = []
for f in acrima_files:
    fname = os.path.basename(f)
    lbl = 1 if "_g_" in fname.lower() else 0
    acr_rows.append({"filepath": f, "label": lbl, "cohort": "ACRIMA"})
acr_df = pd.DataFrame(acr_rows)
acr_tv, acr_test = train_test_split(acr_df, test_size=0.15, stratify=acr_df["label"], random_state=SEED)
acr_tr, acr_val = train_test_split(acr_tv, test_size=0.1765, stratify=acr_tv["label"], random_state=SEED)

# Cohort 2: RIM-ONE DL
rim_df = pd.read_csv("data/RIM_ONE_DL/metadata.csv")
rim_df["cohort"] = "RIM_ONE"
rim_tv, rim_test = train_test_split(rim_df, test_size=0.15, stratify=rim_df["label"], random_state=SEED)
rim_tr, rim_val = train_test_split(rim_tv, test_size=0.1765, stratify=rim_tv["label"], random_state=SEED)

# Cohort 3: Moondream Staged
moon_tr_rows = []
for p in glob.glob("data/MOONDREAM_STAGED/train/normal/*.jpg"):
    moon_tr_rows.append({"filepath": p, "label": 0, "cohort": "Moondream"})
for p in glob.glob("data/MOONDREAM_STAGED/train/early/*.jpg"):
    moon_tr_rows.append({"filepath": p, "label": 1, "stage": "early", "cohort": "Moondream"})
for p in glob.glob("data/MOONDREAM_STAGED/train/advanced/*.jpg"):
    moon_tr_rows.append({"filepath": p, "label": 1, "stage": "advanced", "cohort": "Moondream"})
moon_tr_df = pd.DataFrame(moon_tr_rows)

moon_ts_rows = []
for p in glob.glob("data/MOONDREAM_STAGED/test/normal/*.jpg"):
    moon_ts_rows.append({"filepath": p, "label": 0, "cohort": "Moondream"})
for p in glob.glob("data/MOONDREAM_STAGED/test/early/*.jpg"):
    moon_ts_rows.append({"filepath": p, "label": 1, "stage": "early", "cohort": "Moondream"})
for p in glob.glob("data/MOONDREAM_STAGED/test/advanced/*.jpg"):
    moon_ts_rows.append({"filepath": p, "label": 1, "stage": "advanced", "cohort": "Moondream"})
moon_ts_df = pd.DataFrame(moon_ts_rows)

# Cohort 4: EyePACS
eyepacs_tr_rows = []
for p in glob.glob("data/EYEPACS_TRAIN/normal/*.png"):
    eyepacs_tr_rows.append({"filepath": p, "label": 0, "cohort": "EyePACS"})
for p in glob.glob("data/EYEPACS_TRAIN/glaucoma/*.png"):
    eyepacs_tr_rows.append({"filepath": p, "label": 1, "cohort": "EyePACS"})
eyepacs_tr_df = pd.DataFrame(eyepacs_tr_rows)

eyepacs_ts_rows = []
for p in glob.glob("data/EYEPACS_REAL/normal/*.png"):
    eyepacs_ts_rows.append({"filepath": p, "label": 0, "cohort": "EyePACS"})
for p in glob.glob("data/EYEPACS_REAL/glaucoma/*.png"):
    eyepacs_ts_rows.append({"filepath": p, "label": 1, "cohort": "EyePACS"})
eyepacs_ts_df = pd.DataFrame(eyepacs_ts_rows)

# Combined Mega-Cohort Training & Validation sets
train_df = pd.concat([acr_tr, rim_tr, moon_tr_df, eyepacs_tr_df], ignore_index=True)
val_df = pd.concat([acr_val, rim_val], ignore_index=True)

print("\n================ FEDERACIÓN DE COHORTES PARA ENTRENAMIENTO ================")
print(f"Total Entrenamiento: {len(train_df)} imágenes")
print(f"  - ACRIMA (España, Canon):              {len(acr_tr)} (Glauc: {sum(acr_tr['label'])}, Norm: {len(acr_tr)-sum(acr_tr['label'])})")
print(f"  - RIM-ONE DL (Canarias, Kowa 3D):      {len(rim_tr)} (Glauc: {sum(rim_tr['label'])}, Norm: {len(rim_tr)-sum(rim_tr['label'])})")
print(f"  - Moondream (Multicéntrico Estadificado): {len(moon_tr_df)} (Glauc: {sum(moon_tr_df['label'])}, Norm: {len(moon_tr_df)-sum(moon_tr_df['label'])})")
print(f"  - EyePACS (Cribado Masivo Panorámico):  {len(eyepacs_tr_df)} (Glauc: {sum(eyepacs_tr_df['label'])}, Norm: {len(eyepacs_tr_df)-sum(eyepacs_tr_df['label'])})")
print(f"Total Validación: {len(val_df)} imágenes")
print(f"Cohortes de Prueba Independiente:")
print(f"  - ACRIMA Test:       {len(acr_test)} imágenes")
print(f"  - RIM-ONE Test:      {len(rim_test)} imágenes")
print(f"  - Moondream Test:    {len(moon_ts_df)} imágenes")
print(f"  - EyePACS Real Test: {len(eyepacs_ts_df)} imágenes")

# 2. PyTorch Dataset & Transforms
class MegaCohortDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            im = self.transform(im)
        lbl = torch.tensor(row["label"], dtype=torch.float32)
        return im, lbl

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(MegaCohortDataset(train_df, transform=train_transform), batch_size=32, shuffle=True)
val_loader = DataLoader(MegaCohortDataset(val_df, transform=val_transform), batch_size=32, shuffle=False)

# 3. Model Training
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 1)
model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

best_val_auc = 0.0
best_model_path = "modelos/best_resnet18_megacohort.pth"
epochs = 15

print("\n--- Iniciando Entrenamiento de la Mega-Cohorte Federada (1,603 imágenes) ---")
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
    print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val ROC-AUC: {val_auc:.4f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), best_model_path)
        print(f"  --> [CHECKPOINT] Mejor modelo federado guardado con Val AUC: {best_val_auc:.4f}")

print(f"\n[OK] Entrenamiento federado finalizado. Mejor Val AUC: {best_val_auc:.4f}")

# 4. Comprehensive Benchmark on All 4 Independent Test Cohorts
model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
model.to(DEVICE).eval()

def evaluate_test_cohort(model, df, name):
    probs = []
    with torch.no_grad():
        for p in df["filepath"]:
            im = Image.open(p).convert("RGB")
            t_im = val_transform(im).unsqueeze(0).to(DEVICE)
            prob = torch.sigmoid(model(t_im)).item()
            probs.append(prob)
    labels = df["label"].tolist()
    auc = roc_auc_score(labels, probs)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    acc = accuracy_score(labels, preds)
    sens = recall_score(labels, preds, zero_division=0)
    spec = recall_score(labels, preds, pos_label=0, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return {
        "cohort": name,
        "n": len(df),
        "auc": float(auc),
        "acc": float(acc),
        "sens": float(sens),
        "spec": float(spec),
        "f1": float(f1),
        "cm": cm.tolist(),
        "probs": probs
    }

res_acr = evaluate_test_cohort(model, acr_test, "ACRIMA Test (España, Canon)")
res_rim = evaluate_test_cohort(model, rim_test, "RIM-ONE Test (Canarias, Kowa 3D)")
res_moon = evaluate_test_cohort(model, moon_ts_df, "Moondream Test (Multicéntrico)")
res_eye = evaluate_test_cohort(model, eyepacs_ts_df, "EyePACS Real (Cribado Panorámico)")

print("\n================ BENCHMARK FINAL DE GENERALIZACIÓN MULTI-SENSOR ================")
for r in [res_acr, res_rim, res_moon, res_eye]:
    print(f"Cohorte: {r['cohort']:35s} (n={r['n']:3d}) | ROC-AUC: {r['auc']:.4f} | Acc: {r['acc']*100:.2f}% | Sens: {r['sens']*100:.2f}% | Spec: {r['spec']*100:.2f}%")
    print(f"  Matriz de Confusión: [[TN={r['cm'][0][0]}, FP={r['cm'][0][1]}], [FN={r['cm'][1][0]}, TP={r['cm'][1][1]}]]")

# Also evaluate early vs advanced sensitivity in Moondream
moon_ts_df["pred_prob"] = res_moon["probs"]
moon_early = moon_ts_df[moon_ts_df["stage"] == "early"]
moon_adv = moon_ts_df[moon_ts_df["stage"] == "advanced"]
early_sens = (moon_early["pred_prob"] >= 0.5).mean()
adv_sens = (moon_adv["pred_prob"] >= 0.5).mean()
print(f"\nSensibilidad por Estadio en Moondream:")
print(f"  - Glaucoma Temprano (*Early*):    {early_sens*100:.2f}% ({sum(moon_early['pred_prob'] >= 0.5)}/{len(moon_early)})")
print(f"  - Glaucoma Avanzado (*Advanced*): {adv_sens*100:.2f}% ({sum(moon_adv['pred_prob'] >= 0.5)}/{len(moon_adv)})")

# 5. Save JSON Results
summary = {
    "experiment": "Multi-Source Federated Mega-Cohort",
    "train_samples": len(train_df),
    "val_auc": float(best_val_auc),
    "benchmarks": {
        "acrima": res_acr,
        "rimone": res_rim,
        "moondream": res_moon,
        "eyepacs": res_eye,
        "moondream_early_sens": float(early_sens),
        "moondream_adv_sens": float(adv_sens)
    }
}
with open("results/resumen_experimento_mega_cohorte.json", "w") as f:
    json.dump(summary, f, indent=2)
print("[OK] Guardado results/resumen_experimento_mega_cohorte.json")

# 6. Generate Figures
# Figure 14: Grid of 4 Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=200)
results_list = [res_acr, res_rim, res_moon, res_eye]
classes = ["Normal", "Glaucoma"]

for ax, r in zip(axes.flatten(), results_list):
    cm_arr = np.array(r["cm"])
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"{r['cohort']}\nAUC: {r['auc']:.4f} | Acc: {r['acc']*100:.1f}%", fontsize=10, fontweight="bold", pad=8)
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
                    color="white" if cm_arr[i, j] > thresh else "black", fontsize=12, fontweight="bold")

plt.suptitle("Matrices de Confusión del Modelo Mega-Cohorte en las 4 Fuentes de Prueba", fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig("figures/fig14_megacohort_confusion_matrices.png", dpi=200, bbox_inches="tight")
print("[OK] Guardado figures/fig14_megacohort_confusion_matrices.png")

# Figure 15: Cross-Camera Generalization Comparison
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
cohort_names = ["ACRIMA Test\n(Canon CR-DGi)", "RIM-ONE Test\n(Kowa WX 3D)", "Moondream Test\n(Multicéntrico)", "EyePACS Real\n(Cribado 45°)"]
x = np.arange(len(cohort_names))
width = 0.25

# Performance across 3 paradigms
monomodal_aucs = [0.9960, 0.4487, 0.6210, 0.5240]
bicentric_aucs = [0.9986, 0.9219, 0.7450, 0.5197]
federated_aucs = [res_acr["auc"], res_rim["auc"], res_moon["auc"], res_eye["auc"]]

b1 = ax.bar(x - width, monomodal_aucs, width, label="Modelo Monomodal (Solo ACRIMA)", color="#e74c3c", alpha=0.85)
b2 = ax.bar(x, bicentric_aucs, width, label="Modelo Bi-Céntrico (ACRIMA + RIM-ONE)", color="#3498db", alpha=0.85)
b3 = ax.bar(x + width, federated_aucs, width, label="Modelo Mega-Cohorte Federada (4 Fuentes)", color="#2ecc71", alpha=0.85)

ax.axhline(0.50, color="gray", linestyle="--", alpha=0.7, label="Umbral Aleatorio (0.50)")
ax.set_ylabel("ROC-AUC", fontsize=11, fontweight="bold")
ax.set_title("Evolución de la Generalización Multi-Sensor a través de Paradigmas de Diversidad", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(cohort_names, fontsize=9, fontweight="bold")
ax.set_ylim(0, 1.15)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", linestyle=":", alpha=0.6)

for bars in [b1, b2, b3]:
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig("figures/fig15_megacohort_cross_camera_generalization.png", dpi=200, bbox_inches="tight")
print("[OK] Guardado figures/fig15_megacohort_cross_camera_generalization.png")
