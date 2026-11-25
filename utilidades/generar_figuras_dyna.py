"""Exporta las figuras oficiales en formato Revista DYNA (escala de grises, 1 columna = 3,31 in, 300 dpi)."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utilidades"))
import utilidades as u


def _configurar_estilo():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 7.4,
        "axes.titlesize": 7.8,
        "axes.labelsize": 7.3,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.6,
    })


def _guardar_grises(fig, ruta: Path):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=300)
    plt.close(fig)
    with Image.open(ruta) as im:
        im.convert("L").save(ruta, dpi=(300, 300))
    return ruta


def generar_todas(root: Path = ROOT) -> dict:
    _configurar_estilo()
    df_datos, _ = u.cargar_dataset(root)
    salidas = {}

    # ---- 03 · Muestras por cohorte (5 fuentes locales de 4.1.1) ----
    dir03 = root / "03_experimento_caracteristicas_retinografias/figuras_dyna"
    hrows = pd.read_csv(root / "03_experimento_caracteristicas_retinografias/datos/por_imagen.csv", low_memory=False)
    fuentes = hrows[hrows["en_comparacion_fuentes"] == True].copy()
    eyepacs_t_rel = fuentes.loc[fuentes["coleccion"] == "EyePACS TRAIN", "ruta_relativa"].iloc[0]
    muestras = [
        ("ACRIMA", df_datos[df_datos["coleccion"] == "ACRIMA"].iloc[0]["id"]),
        ("RIM-ONE", df_datos[df_datos["coleccion"] == "RIM-ONE DL"].iloc[0]["id"]),
        ("Moondream", df_datos[df_datos["coleccion"] == "Moondream Staged"].iloc[0]["id"]),
        ("EyePACS R", df_datos[df_datos["coleccion"] == "EyePACS REAL"].iloc[0]["id"]),
        ("EyePACS T", eyepacs_t_rel),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(3.31, 0.88), dpi=300, layout="constrained")
    for ax, (nom, rel) in zip(axes, muestras):
        ax.imshow(u.leer_rgb(root, rel).resize((224, 224)).convert("L"), cmap="gray")
        ax.set_title(nom, fontsize=5.8, pad=2)
        ax.axis("off")
    salidas["03_muestras"] = _guardar_grises(fig, dir03 / "fig_03_muestras_cohortes_grises.png")

    # ---- 03 · Encuadre, brillo, saturación y nitidez (2x2 DYNA) ----
    orden = ["ACRIMA", "RIM-ONE DL", "Moondream Staged", "EyePACS REAL", "EyePACS TRAIN"]
    etq = ["ACRIMA", "RIM-ONE", "Moondream", "EyePACS R", "EyePACS T"]
    fig, axes = plt.subplots(2, 2, figsize=(3.31, 2.85), dpi=300, sharex=True, layout="constrained")
    paneles_03 = [
        ("region_pct", "Área visible (%)", False),
        ("brillo_mediano_yn", "Brillo (luma)", False),
        ("saturacion_media_pct", "Saturación (%)", False),
        ("laplaciano_lado512", "Nitidez (512 px)", True),
    ]
    for ax, (col, ylab, log_y) in zip(axes.ravel(), paneles_03):
        datos = [fuentes.loc[fuentes["coleccion"] == s, col].dropna().to_numpy() for s in orden]
        bp = ax.boxplot(datos, patch_artist=True, showfliers=False, widths=0.52)
        for box in bp["boxes"]:
            box.set_facecolor("#bdbdbd")
            box.set_edgecolor("#222222")
            box.set_linewidth(0.65)
        for med in bp["medians"]:
            med.set_color("#000000")
            med.set_linewidth(1.0)
        for w in bp["whiskers"] + bp["caps"]:
            w.set_color("#444444")
            w.set_linewidth(0.55)
        if log_y:
            ax.set_yscale("symlog", linthresh=1)
        ax.set_ylabel(ylab, fontsize=6.4)
        ax.tick_params(axis="y", labelsize=5.8)
        ax.grid(axis="y", linestyle=":", alpha=0.45, color="#888888")
    for ax in axes[1]:
        ax.set_xticks(range(1, len(orden) + 1), etq, rotation=28, ha="right", fontsize=5.8)
    salidas["03_encuadre"] = _guardar_grises(fig, dir03 / "fig_03_encuadre_brillo_grises.png")

    # ---- 03 · Asimetría intra-fuente por clase (señales espurias) ----
    fig, (ax_m, ax_a) = plt.subplots(1, 2, figsize=(3.31, 1.55), dpi=300, layout="constrained")
    m_sub = fuentes[fuentes["coleccion"] == "Moondream Staged"]
    m_ord = ["normal", "early", "advanced"]
    m_datos = [m_sub.loc[m_sub["etiqueta"] == c, "brillo_mediano_yn"].dropna().to_numpy() for c in m_ord]
    bp_m = ax_m.boxplot(m_datos, patch_artist=True, showfliers=False, widths=0.52)
    for box, col in zip(bp_m["boxes"], ["#d9d9d9", "#9e9e9e", "#525252"]):
        box.set_facecolor(col); box.set_edgecolor("#222222"); box.set_linewidth(0.7)
    for med in bp_m["medians"]:
        med.set_color("#000000"); med.set_linewidth(1.1)
    for w in bp_m["whiskers"] + bp_m["caps"]:
        w.set_color("#444444"); w.set_linewidth(0.6)
    ax_m.set_xticks([1, 2, 3], ["Sano", "Inic.", "Avanz."], fontsize=6.0)
    ax_m.set_ylabel("Brillo (luma)", fontsize=6.5)
    ax_m.set_title("Moondream por estadio", fontsize=6.8)
    ax_m.grid(axis="y", linestyle=":", alpha=0.45, color="#888888")

    a_sub = fuentes[fuentes["coleccion"] == "ACRIMA"]
    a_datos = [a_sub.loc[a_sub["etiqueta"] == c, "ancho_px"].dropna().to_numpy() for c in ["normal", "glaucoma"]]
    bp_a = ax_a.boxplot(a_datos, patch_artist=True, showfliers=False, widths=0.48)
    for box, col in zip(bp_a["boxes"], ["#d9d9d9", "#636363"]):
        box.set_facecolor(col); box.set_edgecolor("#222222"); box.set_linewidth(0.7)
    for med in bp_a["medians"]:
        med.set_color("#000000"); med.set_linewidth(1.1)
    for w in bp_a["whiskers"] + bp_a["caps"]:
        w.set_color("#444444"); w.set_linewidth(0.6)
    ax_a.set_xticks([1, 2], ["Sano", "Glaucoma"], fontsize=6.0)
    ax_a.set_ylabel("Ancho nativo (px)", fontsize=6.5)
    ax_a.set_title("ACRIMA por clase", fontsize=6.8)
    ax_a.grid(axis="y", linestyle=":", alpha=0.45, color="#888888")
    salidas["03_asimetria"] = _guardar_grises(fig, dir03 / "fig_03_asimetria_clases_grises.png")

    # ---- 04 · Matrices 3x3 (referencia previa n=115 vs auditado n=95) ----
    dir04 = root / "04_experimento_trinario_vs_binario/figuras"
    cm_ref = np.array([[55, 2, 2], [6, 14, 7], [5, 4, 20]])
    tf_eval = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(u.MEAN, u.STD)])
    test_moon = df_datos[df_datos["split"] == "test"].reset_index(drop=True)
    ckpt3 = torch.load(root / "modelos/convnext_tiny_3c_estandar.pth", map_location="cpu", weights_only=True)
    m3 = u.crear_convnext(3, preentrenado=False, device="cpu")
    m3.load_state_dict(ckpt3["state_dict"], strict=True)
    val_moon = df_datos[df_datos["split"] == "val"].reset_index(drop=True)
    _, val_p3 = u.predecir(m3, u.crear_loader(val_moon, root, 3, tf_eval, batch_size=32, device="cpu"), "cpu", calcular_loss=False)
    tau3, _ = u.calibrar_umbral(val_moon["y3"].to_numpy(int), val_p3, 3)
    _, test_p3 = u.predecir(m3, u.crear_loader(test_moon, root, 3, tf_eval, batch_size=32, device="cpu"), "cpu", calcular_loss=False)
    _, cm_aud = u.calcular_metricas(test_moon["y3"].to_numpy(int), u.decidir_con_umbral(test_p3, tau3), 3)

    fig, axes = plt.subplots(1, 2, figsize=(3.31, 1.85), dpi=300, sharey=True, layout="constrained")
    cl_c = ["Sano", "Inic.", "Avanz."]
    for ax, cm, tit in zip(axes, [cm_ref, cm_aud], ["Referencia (n=115)", "Auditado (n=95)"]):
        sop = cm.sum(axis=1, keepdims=True)
        pct = np.divide(cm, sop, out=np.zeros_like(cm, dtype=float), where=sop != 0)
        ax.imshow(pct, vmin=0, vmax=1, cmap="Greys")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f"{val}\n({pct[i, j]:.0%})", ha="center", va="center",
                    color="white" if pct[i, j] > 0.52 else "#111111", fontsize=5.8)
        ax.set_xticks(range(3), cl_c)
        ax.set_yticks(range(3), cl_c)
        ax.set_title(tit, fontsize=7.0)
        ax.set_xlabel("Predicción", fontsize=6.6)
    axes[0].set_ylabel("Real", fontsize=6.6)
    salidas["04_matrices"] = _guardar_grises(fig, dir04 / "fig_04_matrices_3c_grises.png")

    # ---- 04 · Proyección trinaria vs binario directo (F1 y AUROC) ----
    m04 = pd.read_csv(root / "04_experimento_trinario_vs_binario/resultados/01_metricas_binarias.csv")
    cohortes_ord = ["Moondream · test", "ACRIMA", "RIM-ONE DL", "EyePACS REAL"]
    coh_lbl = ["Moondream\n(n=95)", "ACRIMA\n(n=703)", "RIM-ONE\n(n=447)", "EyePACS R\n(n=200)"]
    proy = m04[m04["modelo"] == "Proyección trinario"].set_index("cohorte").loc[cohortes_ord]
    bin_ = m04[m04["modelo"] == "Entrenado binario"].set_index("cohorte").loc[cohortes_ord]
    fig, axes = plt.subplots(2, 1, figsize=(3.31, 2.70), dpi=300, sharex=True, layout="constrained")
    x = np.arange(len(cohortes_ord))
    w = 0.34
    for ax, col, ylab in zip(axes, ["F1 macro", "AUROC"], ["F1 macro", "AUROC"]):
        b1 = ax.bar(x - w / 2, proy[col], w, color="#303030", edgecolor="#111111", lw=0.6, label="Proyección trinario→binario")
        b2 = ax.bar(x + w / 2, bin_[col], w, color="#d4d4d4", edgecolor="#222222", lw=0.6, hatch="///", label="Entrenado binario")
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=5.8)
        ax.set_ylim(0, 1.16)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", linestyle=":", alpha=0.45, color="#888888")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=2, frameon=False, fontsize=6.1)
    axes[1].set_xticks(x, coh_lbl)
    salidas["04_comp"] = _guardar_grises(fig, dir04 / "fig_04_trinario_vs_binario_grises.png")

    # ---- 04 · Sensibilidad al umbral ----
    s04 = pd.read_csv(root / "04_experimento_trinario_vs_binario/resultados/01_sensibilidad_umbral.csv")
    reglas04 = ["Espec. val ≥ 0.95", "Youden (val)", "Estándar (τ = 0.50)"]
    reg_lbl = ["Espec≥0,95", "Youden", "τ=0,50"]
    fig, axes = plt.subplots(1, 2, figsize=(3.31, 1.95), dpi=300, sharey=True, layout="constrained")
    styles = [("o-", "#111111"), ("s--", "#555555"), ("^:", "#888888")]
    for ax, mod in zip(axes, ["Proyección trinario", "Entrenado binario"]):
        sub = s04[s04["modelo"] == mod]
        for (rg, rl), (st, col) in zip(zip(reglas04, reg_lbl), styles):
            vals = sub[sub["regla"] == rg].set_index("cohorte").loc[cohortes_ord, "F1 macro"].to_numpy()
            ax.plot(x, vals, st, color=col, lw=1.0, ms=3.5, label=rl)
        ax.set_title(mod.replace("Proyección trinario", "Proyección 3→2").replace("Entrenado binario", "Binario 2→2"), fontsize=7.0)
        ax.set_xticks(x, ["Moon", "ACRI", "RIM", "EyeP"], rotation=15)
        ax.set_ylim(0.25, 1.0)
        ax.grid(True, linestyle=":", alpha=0.45, color="#888888")
    axes[0].set_ylabel("F1 macro")
    axes[1].legend(fontsize=5.6, frameon=True, loc="lower left")
    salidas["04_umbral"] = _guardar_grises(fig, dir04 / "fig_04_sensibilidad_umbral_grises.png")

    # ---- 05 · Muestras Base vs Con aumento + Delta LODO (3 semillas, media ± DE) ----
    dir05 = root / "05_experimento_data_augmentation/figuras"
    ej = df_datos[(df_datos["split"] == "train") & (df_datos["clase"] == "advanced")].iloc[0]
    img_o = u.leer_rgb(root, ej["id"])
    aug = u.AumentoEscalaFotometrico(output_size=(224, 224))
    u.fijar_semilla(1)
    v_out = aug(img_o)
    u.fijar_semilla(2)
    v_in = aug(img_o)

    df_s = pd.read_csv(root / "05_experimento_data_augmentation/resultados/02_metricas_lodo_por_semilla.csv")
    piv = df_s.pivot_table(index=["cohorte_fuera", "seed"], columns="brazo", values=["F1 macro", "AUROC", "sensibilidad"])
    cohs5 = ["EyePACS REAL", "Moondream Staged", "ACRIMA", "RIM-ONE DL"]
    cohs5_lbl = ["EyePACS R (n=200)", "Moondream (n=706)", "ACRIMA (n=703)", "RIM-ONE (n=447)"]
    df1 = 100 * (piv[("F1 macro", "Con aumento")] - piv[("F1 macro", "Base")])
    dauc = 100 * (piv[("AUROC", "Con aumento")] - piv[("AUROC", "Base")])
    dsens = 100 * (piv[("sensibilidad", "Con aumento")] - piv[("sensibilidad", "Base")])

    fig = plt.figure(figsize=(3.31, 3.25), dpi=300, layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 2.0])
    for j, (im, tit) in enumerate([(img_o.resize((224, 224)), "Base"), (v_out, "Zoom-out"), (v_in, "Zoom-in")]):
        ax_i = fig.add_subplot(gs[0, j])
        ax_i.imshow(im.convert("L"), cmap="gray")
        ax_i.set_title(tit, fontsize=6.4, pad=2)
        ax_i.axis("off")

    ax_b = fig.add_subplot(gs[1, :])
    yp = np.arange(len(cohs5))
    h = 0.26
    for off, ser, col, ht, lbl in [
        (-h, df1, "#222222", None, "Δ F1 macro"),
        (0.0, dauc, "#8a8a8a", None, "Δ AUROC"),
        (h, dsens, "#e0e0e0", "///", "Δ Sensibilidad"),
    ]:
        mu = ser.groupby("cohorte_fuera").mean().reindex(cohs5).to_numpy()
        sd = ser.groupby("cohorte_fuera").std().reindex(cohs5).to_numpy()
        ax_b.barh(yp + off, mu, h, xerr=sd, color=col, edgecolor="#111111", lw=0.5, hatch=ht,
                  error_kw={"elinewidth": 0.6, "capsize": 1.8}, label=lbl)
    ax_b.axvline(0, color="#000000", lw=0.8)
    ax_b.set_yticks(yp, cohs5_lbl, fontsize=6.2)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Con aumento − Base (pp · media ± DE en 3 semillas)")
    ax_b.grid(axis="x", linestyle=":", alpha=0.45, color="#888888")
    ax_b.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=5.8)
    salidas["05_lodo"] = _guardar_grises(fig, dir05 / "fig_05_lodo_efecto_aumento_grises.png")

    # ---- 05 · Sensibilidad a tau en LODO (Base vs Con aumento en EyePACS y ACRIMA) ----
    s05 = pd.read_csv(root / "05_experimento_data_augmentation/resultados/02_sensibilidad_umbral_lodo.csv")
    s05_m = s05.groupby(["cohorte_fuera", "regla", "brazo"], as_index=False)[["F1 macro", "TFN"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(3.31, 1.95), dpi=300, sharey=True, layout="constrained")
    xr = np.arange(3)
    for ax, coh, tit in zip(axes, ["EyePACS REAL", "ACRIMA"], ["EyePACS REAL (n=200)", "ACRIMA (n=703)"]):
        for br, sty, col in [("Base", "o--", "#777777"), ("Con aumento", "s-", "#111111")]:
            sub = s05_m[(s05_m["cohorte_fuera"] == coh) & (s05_m["brazo"] == br)].set_index("regla").loc[reglas04]
            ax.plot(xr, sub["F1 macro"], sty, color=col, lw=1.0, ms=3.8, label=br)
        ax.set_title(tit, fontsize=7.0)
        ax.set_xticks(xr, ["Esp≥0,95", "Youden", "τ=0,50"], rotation=15)
        ax.set_ylim(0.28, 0.80)
        ax.grid(True, linestyle=":", alpha=0.45, color="#888888")
    axes[0].set_ylabel("F1 macro (media 3 semillas)")
    axes[1].legend(fontsize=5.8, frameon=True, loc="lower right")
    salidas["05_umbral"] = _guardar_grises(fig, dir05 / "fig_05_lodo_sensibilidad_umbral_grises.png")

    return salidas


if __name__ == "__main__":
    out = generar_todas()
    for k, p in out.items():
        with Image.open(p) as im:
            print(f"{k}: {p.relative_to(ROOT)} | size={im.size} mode={im.mode} dpi={im.info.get('dpi')}")
