#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 08: Diagrama de Flujo PRISMA 2020 en Escala de Grises (300 DPI, Times New Roman)
Proyecto: RSL Glaucoma DL - Capacidad de Generalización
Genera el diagrama formal de flujo PRISMA 2020 reflejando con exactitud los conteos
de identificación, deduplicación, cribado, texto completo, snowballing e inclusión final.
Exporta a estado_del_arte/figuras/fig00_prisma_2020_flujo_escala_grises.png.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, "figuras")
os.makedirs(FIG_DIR, exist_ok=True)

def dibujar_caja(ax, x, y, w, h, titulo, texto, facecolor="#f5f5f5", edgecolor="#000000", lw=1.0):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="square,pad=0",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw
    )
    ax.add_patch(rect)
    # Título en negrita
    ax.text(x + w/2., y + h - 0.04, titulo, ha="center", va="top", fontsize=9, fontweight="bold", color="black")
    # Texto
    ax.text(x + 0.02, y + h - 0.08, texto, ha="left", va="top", fontsize=8, color="black", wrap=True)

def dibujar_flecha(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2)
    )

def main():
    fig, ax = plt.subplots(figsize=(8.5, 11), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Título General
    ax.text(0.5, 0.98, "Diagrama de Flujo PRISMA 2020", ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.955, "Identificación y Selección de Estudios Primarios en la Revisión Sistemática", ha="center", va="top", fontsize=9.5, fontstyle="italic")

    # ================= FASE 1: IDENTIFICACIÓN =================
    ax.text(0.02, 0.91, "IDENTIFICACIÓN", ha="left", va="center", fontsize=9, fontweight="bold", rotation=90, color="#333333")
    
    # Caja Identificación Bases de Datos
    txt_id = (
        "Registros identificados en bases de datos electrónicas (n = 295):\n"
        "  • OpenAlex (n = 150)\n"
        "  • Europe PMC (n = 62)\n"
        "  • PubMed / MEDLINE (n = 54)\n"
        "  • arXiv (rastreo de preprints) (n = 29)\n"
        "Registros de exportación manual (Scopus/WoS/IEEE) (n = 0)"
    )
    dibujar_caja(ax, 0.08, 0.77, 0.40, 0.15, "Registros Identificados", txt_id)

    # Caja Deduplicación (Derecha)
    txt_dedup = (
        "Registros eliminados antes del cribado:\n"
        "  • Duplicados por DOI exacto (n = 70)\n"
        "  • Duplicados por similitud de título ≥ 0.92 (n = 4)\n"
        "Total registros duplicados eliminados (n = 74)"
    )
    dibujar_caja(ax, 0.54, 0.77, 0.40, 0.15, "Deduplicación", txt_dedup)
    dibujar_flecha(ax, 0.48, 0.845, 0.54, 0.845)

    # Flecha Identificación -> Cribado
    dibujar_flecha(ax, 0.28, 0.77, 0.28, 0.70)

    # ================= FASE 2: CRIBADO TÍTULO / RESUMEN =================
    ax.text(0.02, 0.63, "CRIBADO", ha="left", va="center", fontsize=9, fontweight="bold", rotation=90, color="#333333")

    # Caja Registros Cribados
    txt_screen = (
        "Registros cribados por título y resumen (n = 221)\n"
        "Evaluación inter-revisor (doble ciego independiente):\n"
        "  • Acuerdo observado (Po): 82.35%\n"
        "  • Acuerdo esperado (Pe):  49.81%\n"
        "  • Kappa de Cohen (κ):     0.6484 (Acuerdo sustancial)"
    )
    dibujar_caja(ax, 0.08, 0.55, 0.40, 0.15, "Cribado de Título y Resumen", txt_screen)

    # Caja Excluidos TA (Derecha)
    txt_excl_ta = (
        "Registros excluidos por título y resumen (n = 113):\n"
        "  • E1: Modalidad no apta (solo OCT/campos visuales) (n = 54)\n"
        "  • E3: Evaluación solo intra-base sin mitigación (n = 41)\n"
        "  • E4: Preprints no revisados / literatura gris (n = 17)\n"
        "  • E2: Solo segmentación sin decisión diagnóstica (n = 1)"
    )
    dibujar_caja(ax, 0.54, 0.55, 0.40, 0.15, "Excluidos en Cribado", txt_excl_ta)
    dibujar_flecha(ax, 0.48, 0.625, 0.54, 0.625)

    # Flecha Cribado -> Texto Completo
    dibujar_flecha(ax, 0.28, 0.55, 0.28, 0.48)

    # ================= FASE 3: ELEGIBILIDAD A TEXTO COMPLETO =================
    ax.text(0.02, 0.38, "ELEGIBILIDAD", ha="left", va="center", fontsize=9, fontweight="bold", rotation=90, color="#333333")

    # Caja Informes Recuperados
    txt_ft = (
        "Informes evaluados para elegibilidad a texto completo (n = 108):\n"
        "  • Informes descargados en Acceso Abierto (n = 39)\n"
        "  • Informes indexados en espera SINAB / suscripción (n = 69)"
    )
    dibujar_caja(ax, 0.08, 0.35, 0.40, 0.13, "Búsqueda a Texto Completo", txt_ft)

    # Caja Excluidos Texto Completo (Derecha)
    txt_excl_ft = (
        "Informes excluidos a texto completo (n = 26):\n"
        "  • E3: Validación cruzada intra-base sin dominio no visto (n = 21)\n"
        "  • E5: Duplicado o versión preliminar de congreso (n = 3)\n"
        "  • E2: Segmentación papilar sin tarea de clasificación (n = 2)"
    )
    dibujar_caja(ax, 0.54, 0.35, 0.40, 0.13, "Excluidos a Texto Completo", txt_excl_ft)
    dibujar_flecha(ax, 0.48, 0.415, 0.54, 0.415)

    # Flecha Texto Completo -> Inclusión
    dibujar_flecha(ax, 0.28, 0.35, 0.28, 0.27)

    # ================= FASE 4: SNOWBALLING E INCLUSIÓN =================
    ax.text(0.02, 0.17, "INCLUSIÓN", ha="left", va="center", fontsize=9, fontweight="bold", rotation=90, color="#333333")

    # Caja Snowballing (Derecha inferior)
    txt_snow = (
        "Búsqueda retrospectiva y prospectiva (Snowballing):\n"
        "  • Citas y referencias rastreadas de 10 semillas (n = 90)\n"
        "  • Candidatos relevantes evaluados (n = 23)\n"
        "Aportan validación cruzada para las preguntas RQ2–RQ5"
    )
    dibujar_caja(ax, 0.54, 0.15, 0.40, 0.11, "Snowballing Bidireccional", txt_snow)

    # Caja Estudios Incluidos
    txt_inc = (
        "Estudios primarios incluidos en la revisión sistemática (n = 82 directos + 15 seminales curados):\n"
        "  • Síntesis bibliométrica y mapeo (RQ1) (n = 82)\n"
        "  • Análisis de protocolos de evaluación fuera de dominio (RQ2)\n"
        "  • Cuantificación cuantitativa del ΔAUC por cambio de sensor y FOV (RQ3)\n"
        "  • Comparación de efectividad de estrategias Datos vs. Modelo (RQ4)\n"
        "  • Evaluación de arquitecturas y regímenes asintóticos (RQ5)\n"
        "  • Sometidos a evaluación de calidad QA1–QA9 y análisis de sensibilidad"
    )
    dibujar_caja(ax, 0.08, 0.06, 0.40, 0.21, "Estudios Primarios Incluidos", txt_inc, facecolor="#e8e8e8", lw=1.3)
    dibujar_flecha(ax, 0.54, 0.205, 0.48, 0.205)

    # Pie de figura DYNA
    ax.text(0.5, 0.015, "Fuente: Los autores (adaptado de la Declaración PRISMA 2020).", ha="center", va="bottom", fontsize=8.5, fontstyle="italic")

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig00_prisma_2020_flujo_escala_grises.png")
    plt.savefig(out_path, dpi=300, facecolor="white")
    plt.close()
    print(f"[*] Diagrama PRISMA 2020 generado con éxito: {out_path}")

if __name__ == "__main__":
    main()
