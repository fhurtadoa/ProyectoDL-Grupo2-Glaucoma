#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 07: Análisis Bibliométrico y Generación de Figuras Científicas (Revista DYNA)
Proyecto: RSL Glaucoma DL - Capacidad de Generalización
Genera figuras en ESCALA DE GRISES (sin color), 300 DPI, tipografía Times New Roman:
- Fig. 1: Evolución temporal de publicaciones por año (2016-2026).
- Fig. 2: Distribución de publicaciones por país e institución.
- Fig. 3: Frecuencia de uso de bases de datos públicas de glaucoma.
- Fig. 4: Frecuencia de protocolos de evaluación (Intra-base vs. Cross-dataset vs. LODO vs. Externa).
- Fig. 5: Distribución de estrategias contra domain shift (Datos vs. Modelo).
- Fig. 6: Magnitud de caída diagnóstica (ΔAUC) según tipo de cambio de dominio.
Exporta a estado_del_arte/figuras/.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuración formal Times New Roman y escala de grises para DYNA
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLAS_DIR = os.path.join(BASE_DIR, "tablas")
FIG_DIR = os.path.join(BASE_DIR, "figuras")
os.makedirs(FIG_DIR, exist_ok=True)

EXTRACCION_CSV = os.path.join(TABLAS_DIR, "extraccion.csv")

def plot_publicaciones_por_año(df):
    plt.figure(figsize=(7, 4.2), dpi=300)
    conteo_años = df["año"].value_counts().sort_index()
    años = [int(a) for a in conteo_años.index]
    valores = list(conteo_años.values)
    
    # Rellenar años intermedios si faltan
    rango_años = list(range(min(años), max(años) + 1))
    valores_completos = [conteo_años.get(y, 0) for y in rango_años]
    
    bars = plt.bar(rango_años, valores_completos, color="#404040", edgecolor="black", width=0.65)
    for b in bars:
        h = b.get_height()
        if h > 0:
            plt.text(b.get_x() + b.get_width()/2., h + 0.1, f"{int(h)}", ha="center", va="bottom", fontsize=9)
            
    plt.xlabel("Año de Publicación", fontsize=10)
    plt.ylabel("Número de Publicaciones Incluidas", fontsize=10)
    plt.xticks(rango_años, fontsize=9)
    plt.yticks(range(0, max(valores_completos) + 3), fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.5, color="#808080")
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, "fig01_publicaciones_por_ano_escala_grises.png")
    plt.savefig(out_path, dpi=300, facecolor="white")
    plt.close()
    print(f"[*] Guardada: {out_path}")

def plot_bases_de_datos(df):
    plt.figure(figsize=(7.5, 4.5), dpi=300)
    bases_lista = []
    for b_str in df["bases_entrenamiento"].dropna():
        for b in ["ACRIMA", "RIM-ONE", "REFUGE", "Drishti-GS", "ORIGA", "LAG", "EyePACS", "AIROGS", "SMDG", "G1020", "BRSET", "PAPILA"]:
            if b.lower() in b_str.lower():
                bases_lista.append(b)
    for b_str in df["bases_prueba"].dropna():
        for b in ["ACRIMA", "RIM-ONE", "REFUGE", "Drishti-GS", "ORIGA", "LAG", "EyePACS", "AIROGS", "SMDG", "G1020", "BRSET", "PAPILA"]:
            if b.lower() in b_str.lower():
                bases_lista.append(b)
                
    s_bases = pd.Series(bases_lista).value_counts()
    
    y_pos = np.arange(len(s_bases))
    # Tonos de gris diferenciados
    colores_gris = ["#2b2b2b" if i%2==0 else "#666666" for i in range(len(s_bases))]
    
    plt.barh(y_pos, s_bases.values, color=colores_gris, edgecolor="black", height=0.6)
    plt.yticks(y_pos, s_bases.index, fontsize=9)
    plt.xlabel("Frecuencia de Uso en Estudios Primarios (Entrenamiento y Evaluación)", fontsize=10)
    plt.gca().invert_yaxis()
    for i, v in enumerate(s_bases.values):
        plt.text(v + 0.15, i, f"{v}", va="center", fontsize=9)
        
    plt.xlim(0, max(s_bases.values) + 2)
    plt.grid(axis="x", linestyle="--", alpha=0.5, color="#808080")
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, "fig02_bases_datos_frecuencia_escala_grises.png")
    plt.savefig(out_path, dpi=300, facecolor="white")
    plt.close()
    print(f"[*] Guardada: {out_path}")

def plot_protocolos_evaluacion(df):
    plt.figure(figsize=(6.8, 3.8), dpi=300)
    conteo_prot = df["protocolo_evaluacion"].value_counts()
    
    # Agrupar categorías estándar
    cat_map = {
        "Cross-dataset independiente": "Cross-Dataset",
        "Leave-One-Dataset-Out (LODO) exhaustivo": "Leave-One-Domain-Out",
        "Leave-one-domain-out (LODO)": "Leave-One-Domain-Out",
        "Validación externa prospectiva/ciega a especificidad 95%": "Validación Externa Ciega",
        "Validación externa independiente en diversas etnias": "Validación Externa Ciega",
        "Validación externa OOD con adaptación few-shot": "Validación Externa Ciega",
        "Cross-device challenge independiente": "Cross-Device Challenge",
        "Cross-dataset y Out-of-Domain bajo estrés": "Cross-Dataset bajo Estrés",
        "Cross-dataset Unsupervised Domain Adaptation (UDA)": "Cross-Dataset con UDA",
        "Cross-dataset independiente guiado por atención": "Cross-Dataset",
        "Validación cruzada 5-fold por paciente": "Intra-Base por Paciente",
        "Partición intra-hospitalaria estratificada por severidad": "Intra-Base por Paciente",
        "Partición independiente con análisis de ROI papilar vs imagen completa": "Intra-Base por Paciente",
        "Evaluación independiente con análisis de equidad demográfica": "Validación Externa Ciega"
    }
    
    categorias = [cat_map.get(p, "Otros") for p in df["protocolo_evaluacion"]]
    s_cat = pd.Series(categorias).value_counts()
    
    bars = plt.bar(s_cat.index, s_cat.values, color=["#1f1f1f", "#474747", "#737373", "#999999", "#bfbfbf", "#d9d9d9"], edgecolor="black", width=0.55)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2., h + 0.1, f"{int(h)}", ha="center", va="bottom", fontsize=9)
        
    plt.xlabel("Protocolo de Evaluación Reportado", fontsize=10)
    plt.ylabel("Número de Estudios Primarios", fontsize=10)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.yticks(range(0, max(s_cat.values) + 3), fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.5, color="#808080")
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, "fig03_protocolos_evaluacion_escala_grises.png")
    plt.savefig(out_path, dpi=300, facecolor="white")
    plt.close()
    print(f"[*] Guardada: {out_path}")

def plot_estrategias_mitigacion(df):
    plt.figure(figsize=(7.2, 4.2), dpi=300)
    
    estrategias_simpl = []
    for e in df["estrategia_mitigacion_cat"]:
        if "Armonización ROI" in e:
            estrategias_simpl.append("Armonización ROI Papilar")
        elif "UDA" in e:
            estrategias_simpl.append("Adaptación de Dominio (UDA)")
        elif "Domain Generalization" in e:
            estrategias_simpl.append("Domain Generalization (DG)")
        elif "Modelo Fundacional" in e:
            estrategias_simpl.append("Modelo Fundacional Retiniano")
        elif "Filtro" in e:
            estrategias_simpl.append("Filtro Automático de Calidad")
        elif "Multifuente" in e:
            estrategias_simpl.append("Entrenamiento Multifuente Masivo")
        elif "Estadificación" in e:
            estrategias_simpl.append("Estadificación Multietapa")
        elif "Normalización de color" in e:
            estrategias_simpl.append("Normalización Cromática")
        elif "Atención" in e:
            estrategias_simpl.append("Atención Visual Guiada")
        else:
            estrategias_simpl.append("Aumento Estándar / Transfer")
            
    s_est = pd.Series(estrategias_simpl).value_counts()
    
    y_pos = np.arange(len(s_est))
    plt.barh(y_pos, s_est.values, color="#383838", edgecolor="black", height=0.6)
    plt.yticks(y_pos, s_est.index, fontsize=9)
    plt.xlabel("Frecuencia de Estrategias contra Domain Shift", fontsize=10)
    plt.gca().invert_yaxis()
    for i, v in enumerate(s_est.values):
        plt.text(v + 0.1, i, f"{v}", va="center", fontsize=9)
        
    plt.xlim(0, max(s_est.values) + 1.5)
    plt.grid(axis="x", linestyle="--", alpha=0.5, color="#808080")
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, "fig04_estrategias_mitigacion_escala_grises.png")
    plt.savefig(out_path, dpi=300, facecolor="white")
    plt.close()
    print(f"[*] Guardada: {out_path}")

def plot_caida_delta_auc(df):
    plt.figure(figsize=(7.5, 4.2), dpi=300)
    sub_df = df.dropna(subset=["delta_auc"]).copy()
    sub_df = sub_df.sort_values(by="delta_auc", ascending=False)
    
    refs = [r.split(",")[0] for r in sub_df["referencia"]]
    deltas = sub_df["delta_auc"].values
    
    # Patrón de barras según magnitud
    colores = ["#1a1a1a" if d > 0.15 else "#555555" if d > 0.06 else "#8c8c8c" for d in deltas]
    
    bars = plt.bar(range(len(deltas)), deltas, color=colores, edgecolor="black", width=0.6)
    for b, d in zip(bars, deltas):
        plt.text(b.get_x() + b.get_width()/2., d + 0.005, f"{d:.3f}", ha="center", va="bottom", fontsize=8, rotation=0)
        
    plt.axhline(0.10, color="black", linestyle=":", linewidth=1.2, label="Umbral crítico de caída (ΔAUC = 0.10)")
    plt.xlabel("Estudio Primario Evaluado Fuera de Dominio", fontsize=10)
    plt.ylabel("Caída Neta de Desempeño (ΔAUC = AUC_in - AUC_out)", fontsize=10)
    plt.xticks(range(len(deltas)), refs, rotation=35, ha="right", fontsize=8.5)
    plt.ylim(0, max(deltas) + 0.04)
    plt.legend(frameon=True, edgecolor="black", fontsize=8.5)
    plt.grid(axis="y", linestyle="--", alpha=0.5, color="#808080")
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, "fig05_caida_delta_auc_por_estudio_escala_grises.png")
    plt.savefig(out_path, dpi=300, facecolor="white")
    plt.close()
    print(f"[*] Guardada: {out_path}")

def main():
    if not os.path.exists(EXTRACCION_CSV):
        print(f"[!] No existe {EXTRACCION_CSV}. Ejecute scripts/06_calidad_y_extraccion.py.")
        return
    df = pd.read_csv(EXTRACCION_CSV)
    print(f"[*] Generando 5 figuras bibliométricas en escala de grises para {len(df)} estudios...")
    plot_publicaciones_por_año(df)
    plot_bases_de_datos(df)
    plot_protocolos_evaluacion(df)
    plot_estrategias_mitigacion(df)
    plot_caida_delta_auc(df)
    print("[*] Todas las figuras bibliométricas generadas exitosamente en estado_del_arte/figuras/.")

if __name__ == "__main__":
    main()
