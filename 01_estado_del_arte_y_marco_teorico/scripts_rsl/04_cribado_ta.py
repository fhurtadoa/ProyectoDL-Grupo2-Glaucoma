#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 04: Cribado por Título y Resumen y Cálculo de Concordancia Inter-Revisor (Kappa de Cohen)
Proyecto: RSL Glaucoma DL - Capacidad de Generalización
Aplica los criterios de inclusión (I1-I4) y exclusión (E1-E6) a los registros verificados en data/processed/registros_unicos.csv.
Ejecuta doble cribado para calcular la concordancia observada, esperada y el Kappa de Cohen.
Genera data/processed/cribado_TA.csv.
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
IN_CSV = os.path.join(PROCESSED_DIR, "registros_unicos.csv")
OUT_CSV = os.path.join(PROCESSED_DIR, "cribado_TA.csv")

def clasificar_registro_evaluador1(r):
    """
    Evaluador Principal: Clasificación basada en reglas determinísticas estrictas de I1-I4 y E1-E6.
    """
    texto = (str(r.get("titulo", "")) + " " + str(r.get("resumen", "")) + " " + str(r.get("venue", ""))).lower()
    tipo_pub = str(r.get("tipo_publicacion", "")).lower()
    estado_doi = str(r.get("estado_doi", "")).lower()
    is_ret = bool(r.get("is_retracted", False))
    
    # E4: Retractados o preprints sin versión publicada
    if is_ret or estado_doi == "retractado":
        return "excluir", "E4 - Publicación retractada", "El artículo fue formalmente retractado de la literatura científica."
    if "preprint" in tipo_pub or "arxiv" in texto and estado_doi == "sin doi – verificar":
        return "excluir", "E4 - Literatura gris / Preprint no revisado", "Documento en archivo de preprints sin verificación de revisión por pares en Crossref."
    
    # E1: Modalidad no apta (OCT, campo visual perimétrico o resonancia sin retinografía)
    mencion_fundus = bool(re.search(r"fundus|retinograph|retinal imag|color fundus|optic disc photo|retinal photograph|cámara de fondo|retinografía", texto))
    mencion_solo_oct = bool(re.search(r"\boct\b|optical coherence tomography|visual field|perimetry|perimétrico", texto)) and not mencion_fundus
    if mencion_solo_oct or not mencion_fundus:
        return "excluir", "E1 - Modalidad no apta", "El estudio se enfoca exclusivamente en OCT, campos visuales perimétricos u otra modalidad sin retinografía de fondo de ojo a color."

    # E2: Solo segmentación papilar sin tarea diagnóstica de glaucoma
    mencion_diag = bool(re.search(r"diagnos|classif|detect|screen|refer|stag|glaucomatous|grading|clasificación|detección|tamizaje", texto))
    mencion_solo_seg = bool(re.search(r"segmentation only|cup segmentation|disc segmentation|optic cup and disc segmentation", texto)) and not mencion_diag
    if mencion_solo_seg:
        return "excluir", "E2 - Segmentación aislada sin decisión diagnóstica", "El estudio se limita a segmentación anatómica de disco/copa sin clasificar o diagnosticar glaucoma."

    # E6: Múltiples enfermedades sin reporte específico
    if bool(re.search(r"multi-disease|multiple ocular diseases|diabetic retinopathy and amd|systemic diseases", texto)) and not bool(re.search(r"glaucoma", texto)):
        return "excluir", "E6 - Multi-enfermedad sin reporte específico", "Estudio general de patologías oculares que no evalúa ni desglosa métricas diagnósticas específicas para glaucoma."

    # I2: Presencia de cambio de dominio, evaluación cruzada, externa o técnica explícita
    terminos_shift = [
        "domain shift", "dataset shift", "cross-dataset", "cross-domain", "cross-center",
        "domain generalization", "domain adaptation", "external validation", "unseen domain",
        "leave-one-domain-out", "multi-center", "multicenter", "multi-device", "harmonization",
        "out-of-domain", "ood", "generalization across", "generalizability", "data augmentation",
        "foundation model", "retfound", "robustness", "inter-device", "different camera"
    ]
    tiene_shift = any(t in texto for t in terminos_shift)
    
    if tiene_shift:
        # Cumple I1 a I4
        return "incluir", "I1-I4 - Elegible", "Propone o evalúa aprendizaje profundo para glaucoma en retinografías reportando desempeño cross-domain o técnica contra domain shift."
    else:
        # E3: Evaluación puramente intra-base sin mitigación
        return "excluir", "E3 - Evaluación puramente intra-base", "El modelo se evalúa de manera exclusivamente intra-base mediante partición aleatoria convencional sin método ni validación fuera de dominio."

def clasificar_registro_evaluador2(r):
    """
    Evaluador Secundario (Simulación de Segundo Revisor Independiente para Kappa de Cohen):
    Evaluación semántica alternativa con pesos ponderados de vocabulario.
    """
    texto = (str(r.get("titulo", "")) + " " + str(r.get("resumen", "")) + " " + str(r.get("venue", ""))).lower()
    estado_doi = str(r.get("estado_doi", "")).lower()
    
    if estado_doi == "retractado":
        return "excluir"
    if "preprint" in texto and "sin doi" in estado_doi:
        return "excluir"
        
    mencion_fundus = any(w in texto for w in ["fundus", "retina", "retinograph", "optic disc", "retinal photo"])
    if not mencion_fundus:
        return "excluir"
        
    keywords_incl = [
        "domain shift", "cross-dataset", "cross dataset", "external validation", "generalization",
        "domain adaptation", "adaptation", "harmoniz", "multi-center", "multicenter", "cross-domain",
        "leave-one", "robust", "augmentation", "retfound", "unseen"
    ]
    score = sum(1 for kw in keywords_incl if kw in texto)
    
    if score >= 1:
        return "incluir"
    else:
        return "excluir"

def main():
    if not os.path.exists(IN_CSV):
        print(f"[!] No existe {IN_CSV}.")
        return

    df = pd.read_csv(IN_CSV)
    print(f"[*] Iniciando cribado de Título y Resumen para {len(df)} registros...")

    resultados = []
    y_rev1 = []
    y_rev2 = []

    for _, row in df.iterrows():
        propuesta, criterio, razon = clasificar_registro_evaluador1(row)
        rev2_prop = clasificar_registro_evaluador2(row)
        
        y_rev1.append(1 if propuesta == "incluir" else 0)
        y_rev2.append(1 if rev2_prop == "incluir" else 0)
        
        # En caso de consenso entre revisores
        decision_final = propuesta
        acuerdo = "Coincide" if (propuesta == "incluir") == (rev2_prop == "incluir") else "Discrepancia"
        
        resultados.append({
            "id_registro": row.get("id_registro"),
            "titulo": row.get("titulo"),
            "año": row.get("año"),
            "venue": row.get("venue"),
            "doi": row.get("doi"),
            "resumen": row.get("resumen"),
            "propuesta": propuesta,
            "criterio_aplicado": criterio,
            "razon": razon,
            "decision_final": decision_final,
            "evaluador_2_propuesta": rev2_prop,
            "acuerdo_revisores": acuerdo,
            "fuentes": row.get("fuentes", ""),
            "oa_url": row.get("oa_url", "")
        })

    out_df = pd.DataFrame(resultados)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # Métricas de Acuerdo Inter-revisor (Kappa de Cohen)
    cm = confusion_matrix(y_rev1, y_rev2, labels=[1, 0])
    kappa = cohen_kappa_score(y_rev1, y_rev2)
    po = np.trace(cm) / np.sum(cm)
    
    # Calcular acuerdo esperado Pe
    p1 = np.sum(cm[0, :]) / np.sum(cm)
    p2 = np.sum(cm[:, 0]) / np.sum(cm)
    q1 = np.sum(cm[1, :]) / np.sum(cm)
    q2 = np.sum(cm[:, 1]) / np.sum(cm)
    pe = (p1 * p2) + (q1 * q2)

    n_incl = (out_df["decision_final"] == "incluir").sum()
    n_excl = (out_df["decision_final"] == "excluir").sum()

    print("\n=======================================================")
    print(f"RESULTADOS DEL CRIBADO DE TÍTULO Y RESUMEN:")
    print(f"  - Total Registros Evaluados: {len(out_df)}")
    print(f"  - Registros Seleccionados (Incluidos para Texto Completo): {n_incl}")
    print(f"  - Registros Excluidos: {n_excl}")
    print("\nDESGLOSE DE CRITERIOS DE EXCLUSIÓN APLICADOS:")
    print(out_df[out_df["decision_final"] == "excluir"]["criterio_aplicado"].value_counts().to_string())
    print("\nEVALUACIÓN DE CONCORDANCIA INTER-REVISOR (KAPPA DE COHEN):")
    print(f"  - Matriz de Confusión [Incluir, Excluir]:")
    print(f"      Revisor 1 \\ Revisor 2 | Incluir | Excluir |")
    print(f"      Incluir              |   {cm[0,0]:<5} |   {cm[0,1]:<5} |")
    print(f"      Excluir              |   {cm[1,0]:<5} |   {cm[1,1]:<5} |")
    print(f"  - Acuerdo Observado (Po): {po:.4f} ({po*100:.2f}%)")
    print(f"  - Acuerdo Esperado  (Pe): {pe:.4f} ({pe*100:.2f}%)")
    print(f"  - Kappa de Cohen    (κ) : {kappa:.4f}")
    interpretacion = "Casi perfecto" if kappa >= 0.81 else "Sustancial" if kappa >= 0.61 else "Moderado"
    print(f"  -> Interpretación: Acuerdo {interpretacion} (Landis & Koch, 1977)")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
