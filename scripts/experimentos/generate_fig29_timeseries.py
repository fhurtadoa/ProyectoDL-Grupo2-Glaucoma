import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_JSON = BASE_DIR / "results" / "1000_epochs_stress_experiment_metrics.json"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

history = data["training_history"]
df = pd.DataFrame(history)
total_time = data["training_time_seconds"] # 8468.10s
time_per_epoch_min = (total_time / 1000.0) / 60.0 # ~0.141135 min/epoch

df["time_min"] = df["epoch"] * time_per_epoch_min
df["generalization_gap"] = df["val_loss"] - df["train_loss"]

# Rolling smoothing
window = 15
df["val_loss_roll"] = df["val_loss"].rolling(window, min_periods=1).mean()
df["val_auc_roll"] = df["val_auc"].rolling(window, min_periods=1).mean()
df["val_auc_std"] = df["val_auc"].rolling(window, min_periods=1).std().fillna(0)
df["train_loss_roll"] = df["train_loss"].rolling(window, min_periods=1).mean()

# Estilo profesional de publicación médica
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(17, 12), dpi=300)

c_train = "#1e40af"       # Azul zafiro oscuro
c_val = "#dc2626"         # Rojo carmesí
c_auc = "#059669"         # Verde esmeralda
c_gap = "#d97706"         # Ámbar oscuro
c_lr = "#64748b"          # Pizarra
c_acc = "#7c3aed"         # Violeta

best_epoch = data["best_epoch"]
best_val_auc = data["best_val_auc"]

# Helper para añadir eje superior de tiempo
def add_time_axis(ax):
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    # Generar marcas en minutos
    epoch_ticks = [0, 200, 400, 600, 800, 1000]
    time_ticks = [e * time_per_epoch_min for e in epoch_ticks]
    ax_top.set_xticks(epoch_ticks)
    ax_top.set_xticklabels([f"{t:.1f} min" if t > 0 else "0 min" for t in time_ticks], fontsize=8.5, color="#334155")
    ax_top.set_xlabel("Tiempo Cronológico Transcurrido (Minutos)", fontsize=9, fontweight="bold", color="#334155", labelpad=6)
    ax_top.grid(False)
    return ax_top

# ==========================================
# PANEL A: SERIE TEMPORAL DE PÉRDIDAS (BCE LOSS)
# ==========================================
ax_a = axes[0, 0]
ax_a.plot(df["epoch"], df["train_loss"], color=c_train, alpha=0.35, lw=0.9, label="Train Loss (Raw, Malignant Aug)")
ax_a.plot(df["epoch"], df["train_loss_roll"], color=c_train, lw=2.0, label="Train Loss (Media Móvil 15 ep)")
ax_a.plot(df["epoch"], df["val_loss"], color=c_val, alpha=0.35, lw=0.9, label="Val Loss (Raw)")
ax_a.plot(df["epoch"], df["val_loss_roll"], color=c_val, lw=2.0, label="Val Loss (Media Móvil 15 ep)")

# Marcadores verticales clave
ax_a.axvline(15, color="#ea580c", linestyle=":", lw=1.8, label="Época 15 (Early Stop Estándar, t=2.1m)")
ax_a.axvline(100, color="#8b5cf6", linestyle="-.", lw=1.4, label="Época 100 (Ruptura Mínimo Local, t=14.1m)")
ax_a.axvline(best_epoch, color=c_auc, linestyle="--", lw=1.8, label=f"Época {best_epoch} (Pico ROC-AUC, t=101.6m)")

ax_a.set_title("A. Serie Temporal de Función de Pérdida (Epoch vs. BCE Loss)", fontsize=11, fontweight="bold", color="#0f172a", pad=10)
ax_a.set_xlabel("Épocas de Entrenamiento", fontsize=9.5, fontweight="bold")
ax_a.set_ylabel("Pérdida Binaria (BCE Loss)", fontsize=9.5, fontweight="bold")
ax_a.set_xlim([0, 1000])
ax_a.set_ylim([0, 0.70])
ax_a.legend(loc="upper right", frameon=True, fontsize=8, facecolor="white", edgecolor="#cbd5e1")
ax_a.grid(True, linestyle="--", alpha=0.5)
add_time_axis(ax_a)

# ==========================================
# PANEL B: SERIE TEMPORAL DE ROC-AUC (DISCRIMINATIVE POWER)
# ==========================================
ax_b = axes[0, 1]
ax_b.plot(df["epoch"], df["val_auc"], color=c_auc, alpha=0.3, lw=0.8, label="Val ROC-AUC (Instantáneo)")
ax_b.plot(df["epoch"], df["val_auc_roll"], color="#047857", lw=2.2, label="Val ROC-AUC (Media Móvil 15 ep)")
ax_b.fill_between(df["epoch"], 
                  df["val_auc_roll"] - df["val_auc_std"], 
                  df["val_auc_roll"] + df["val_auc_std"], 
                  color=c_auc, alpha=0.15, label="Banda de Volatilidad (±1 SD)")

# Referencias
auc_ep15 = df.loc[df["epoch"]==15, "val_auc"].values[0]
ax_b.axhline(auc_ep15, color="#ea580c", linestyle=":", lw=1.5, label=f"AUC Época 15 = {auc_ep15:.4f}")
ax_b.axhline(best_val_auc, color="#b91c1c", linestyle="--", lw=1.6, label=f"Récord Absoluto (Ép. 720) = {best_val_auc:.4f}")
ax_b.axvline(15, color="#ea580c", linestyle=":", lw=1.5)
ax_b.axvline(best_epoch, color=c_auc, linestyle="--", lw=1.5)

# Zona de Grokking
ax_b.fill_betweenx([0.85, 1.0], 15, 1000, color="#10b981", alpha=0.06, label="Zona de Grokking / Ganancia Asintótica")

ax_b.set_title("B. Serie Temporal de Capacidad Discriminativa (Epoch vs. ROC-AUC)", fontsize=11, fontweight="bold", color="#0f172a", pad=10)
ax_b.set_xlabel("Épocas de Entrenamiento", fontsize=9.5, fontweight="bold")
ax_b.set_ylabel("ROC-AUC en Validación", fontsize=9.5, fontweight="bold")
ax_b.set_xlim([0, 1000])
ax_b.set_ylim([0.86, 0.995])
ax_b.legend(loc="lower right", frameon=True, fontsize=8, facecolor="white", edgecolor="#cbd5e1")
ax_b.grid(True, linestyle="--", alpha=0.5)
add_time_axis(ax_b)

# ==========================================
# PANEL C: SERIE TEMPORAL DE BRECHA DE GENERALIZACIÓN Y TASA LR
# ==========================================
ax_c = axes[1, 0]
ax_c.plot(df["epoch"], df["generalization_gap"], color=c_gap, lw=1.8, label=r"Brecha de Pérdida $\Delta L = L_{val} - L_{train}$")
ax_c.plot(df["epoch"], df["generalization_gap"].rolling(15, min_periods=1).mean(), color="#b45309", lw=2.2, linestyle="--", label="Tendencia Brecha (15 ep)")

ax_c_lr = ax_c.twinx()
ax_c_lr.plot(df["epoch"], df["lr"], color=c_lr, lw=1.6, linestyle=":", label=r"Cosine Annealing LR ($\eta$)")
ax_c_lr.set_ylabel(r"Tasa de Aprendizaje ($\eta$)", fontsize=9.5, fontweight="bold", color=c_lr)
ax_c_lr.tick_params(axis='y', labelcolor=c_lr)
ax_c_lr.grid(False)

ax_c.set_title(r"C. Serie Temporal de la Brecha de Generalización vs. LR Decay", fontsize=11, fontweight="bold", color="#0f172a", pad=10)
ax_c.set_xlabel("Épocas de Entrenamiento", fontsize=9.5, fontweight="bold")
ax_c.set_ylabel(r"Brecha $\Delta L$ (Escala Absoluta)", fontsize=9.5, fontweight="bold", color="#b45309")
ax_c.set_xlim([0, 1000])
ax_c.tick_params(axis='y', labelcolor="#b45309")
ax_c.grid(True, linestyle="--", alpha=0.5)

# Combinar leyendas
lines_c1, labels_c1 = ax_c.get_legend_handles_labels()
lines_c2, labels_c2 = ax_c_lr.get_legend_handles_labels()
ax_c.legend(lines_c1 + lines_c2, labels_c1 + labels_c2, loc="center right", frameon=True, fontsize=8, facecolor="white", edgecolor="#cbd5e1")
add_time_axis(ax_c)

# ==========================================
# PANEL D: SERIE TEMPORAL DE EXACTITUD GLOBAL (ACCURACY %)
# ==========================================
ax_d = axes[1, 1]
ax_d.plot(df["epoch"], df["val_acc"]*100, color=c_acc, alpha=0.3, lw=0.8, label="Val Accuracy Instantánea (%)")
ax_d.plot(df["epoch"], (df["val_acc"]*100).rolling(15, min_periods=1).mean(), color="#5b21b6", lw=2.2, label="Val Accuracy (Media Móvil 15 ep)")

# Puntos clave
acc_ep15 = df.loc[df["epoch"]==15, "val_acc"].values[0] * 100
acc_ep720 = df.loc[df["epoch"]==best_epoch, "val_acc"].values[0] * 100
acc_ep1000 = df.loc[df["epoch"]==1000, "val_acc"].values[0] * 100

ax_d.axhline(acc_ep15, color="#ea580c", linestyle=":", lw=1.5, label=f"Acc Época 15 = {acc_ep15:.2f}%")
ax_d.axhline(93.05, color="#059669", linestyle="--", lw=1.5, label="Pico Máximo Acc = 93.05% (Ép. 600)")
ax_d.scatter([15, 600, best_epoch, 1000], [acc_ep15, 93.05, acc_ep720, acc_ep1000], color=["#ea580c", "#059669", c_auc, "#5b21b6"], s=50, zorder=5)

ax_d.set_title("D. Serie Temporal de Exactitud en Validación (Epoch vs. Accuracy %)", fontsize=11, fontweight="bold", color="#0f172a", pad=10)
ax_d.set_xlabel("Épocas de Entrenamiento", fontsize=9.5, fontweight="bold")
ax_d.set_ylabel("Exactitud de Clasificación (%)", fontsize=9.5, fontweight="bold")
ax_d.set_xlim([0, 1000])
ax_d.set_ylim([75, 96])
ax_d.legend(loc="lower right", frameon=True, fontsize=8, facecolor="white", edgecolor="#cbd5e1")
ax_d.grid(True, linestyle="--", alpha=0.5)
add_time_axis(ax_d)

plt.suptitle("Serie Temporal Continua de Optimización: Épocas y Tiempo Transcurrido vs. Pérdida (Loss) y ROC-AUC (1.000 Épocas)\nResNet-18 en la Mega-Cohorte Médica Armonizada bajo Malignant Scale Augmentation (8.468 s / 141,13 min)", fontsize=13, fontweight="bold", color="#0f172a", y=0.985)
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_fig = FIGURES_DIR / "fig29_time_series_epoch_vs_loss_auc.png"
plt.savefig(out_fig, bbox_inches="tight")
plt.close()

print(f"[EXITO] Figura 29 guardada en: {out_fig}")
