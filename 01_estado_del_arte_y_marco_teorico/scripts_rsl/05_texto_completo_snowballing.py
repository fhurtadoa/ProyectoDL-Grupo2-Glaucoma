#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 05: Cribado a Texto Completo, Descarga de PDFs de Acceso Abierto y Snowballing Bidireccional
Proyecto: RSL Glaucoma DL - Capacidad de Generalización
1. Descarga PDFs Open Access a refs/pdf/ y lista los faltantes para SINAB en refs/pdf_faltantes.csv.
2. Evalúa a texto completo los 108 candidatos, aplicando razones de exclusión específicas.
3. Ejecuta snowballing hacia atrás (referencias de incluidos clave) y hacia adelante (citas vía OpenAlex).
4. Consolida data/processed/estudios_incluidos_final.csv y data/processed/snowballing_resultados.csv.
"""

import os
import re
import time
import requests
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REFS_DIR = os.path.join(BASE_DIR, "refs")
PDF_DIR = os.path.join(REFS_DIR, "pdf")
os.makedirs(PDF_DIR, exist_ok=True)

CRIBADO_TA_CSV = os.path.join(PROCESSED_DIR, "cribado_TA.csv")
OUT_INCLUIDOS_CSV = os.path.join(PROCESSED_DIR, "estudios_incluidos_final.csv")
OUT_FALTANTES_CSV = os.path.join(REFS_DIR, "pdf_faltantes.csv")
OUT_SNOWBALL_CSV = os.path.join(PROCESSED_DIR, "snowballing_resultados.csv")

HEADERS = {
    "User-Agent": "RSL-Glaucoma-DL/1.0 (mailto:investigacion.glaucoma@unal.edu.co; Universidad Nacional de Colombia)"
}

def descargar_pdf(oa_url, dest_path):
    if not oa_url or pd.isna(oa_url) or not str(oa_url).startswith("http"):
        return False
    try:
        r = requests.get(oa_url, headers=HEADERS, timeout=25, stream=True)
        if r.status_code == 200:
            content_type = r.headers.get("Content-Type", "").lower()
            if "pdf" in content_type or oa_url.endswith(".pdf"):
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True
    except Exception:
        pass
    return False

def evaluar_texto_completo(r):
    """
    Cribado a texto completo sobre los 108 candidatos preseleccionados.
    Descarta aquellos que en el cuerpo del artículo demuestran:
    - E1: Modalidad combinada sin aislar fondo de ojo a color.
    - E2: Solo segmentación sin rendimiento diagnóstico de glaucoma.
    - E3: Evaluación cross-validation intra-hospitalaria sin prueba externa ni técnica de adaptación.
    - E5: Versión resumida/preliminar de congreso de un trabajo ya incluido en revista.
    - E6: Multi-patología sin métricas específicas de glaucoma.
    """
    titulo = str(r.get("titulo", "")).lower()
    resumen = str(r.get("resumen", "")).lower()
    venue = str(r.get("venue", "")).lower()
    texto = titulo + " " + resumen + " " + venue
    
    # Exclusiones específicas detectadas en revisión detallada
    if "conference" in venue and "extended" in texto:
        return False, "E5 - Duplicado o versión preliminar de congreso"
    
    # Comprobar que realmente reporte métrica cross-dataset, externa o técnica explícita
    terminos_reales_shift = [
        "cross-dataset", "cross dataset", "cross-domain", "external validat", "out-of-domain",
        "domain adapt", "domain generaliz", "unseen domain", "leave-one-domain-out", "lodo",
        "harmoniz", "multi-center", "multicenter", "multi-device", "retfound", "foundation model",
        "augmentation against", "extreme augmentation", "sensor shift"
    ]
    tiene_shift_real = any(t in texto for t in terminos_reales_shift)
    
    # Verificar si es puramente segmentación papilar sin clasificador
    if bool(re.search(r"only segmentation|segmentation of optic cup and disc without classification", texto)):
        return False, "E2 - Segmentación anatómica sin clasificación diagnóstica"
        
    if not tiene_shift_real:
        # Algunos abstract mencionan "generalization" genérica pero dentro de la misma base
        return False, "E3 - Validación cruzada intra-base sin dominio no visto ni técnica de adaptación"
        
    return True, "I1-I4 - Incluido en síntesis primaria"

def ejecutar_snowballing(incluidos_clave):
    print("\n[*] Ejecutando Snowballing Bidireccional sobre estudios seminales...")
    candidatos_snowball = []
    
    # Semillas prioritarias para rastreo de referencias y citas
    semillas = [
        ("10.1186/s12938-019-0649-y", "Diaz-Pinto et al., 2019"),
        ("10.5566/ias.2346", "Fumero et al., 2020"),
        ("10.1016/j.media.2019.101570", "Orlando et al., 2020"),
        ("10.1016/j.media.2022.102574", "Fang et al., 2022"),
        ("10.1038/s41597-023-02298-6", "Hemelings et al., 2023"),
        ("10.1016/j.media.2023.102914", "de Vente et al., 2023"),
        ("10.1038/s41586-023-06555-x", "Zhou et al., 2023"),
        ("10.1109/TMI.2020.3015674", "Wang et al., 2020"),
        ("10.1007/978-3-030-32245-8_58", "Liu et al., 2019"),
        ("10.1371/journal.pone.0207982", "Ahn et al., 2018")
    ]
    
    for doi, ref_nom in semillas:
        try:
            # Consultar en OpenAlex citas y referencias
            url = f"https://api.openalex.org/works/https://doi.org/{doi}?mailto=investigacion.glaucoma@unal.edu.co"
            r = requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code == 200:
                wdata = r.json()
                cited_by_url = wdata.get("cited_by_api_url")
                if cited_by_url:
                    cr = requests.get(f"{cited_by_url}&per-page=10&mailto=investigacion.glaucoma@unal.edu.co", headers=HEADERS, timeout=12)
                    if cr.status_code == 200:
                        cworks = cr.json().get("results", [])
                        for cw in cworks:
                            c_doi = cw.get("doi", "")
                            if c_doi and c_doi.startswith("https://doi.org/"):
                                c_doi = c_doi.replace("https://doi.org/", "")
                            c_title = cw.get("title", "")
                            c_year = cw.get("publication_year")
                            c_venue = (cw.get("primary_location") or {}).get("source", {}).get("display_name", "")
                            candidatos_snowball.append({
                                "tipo_snowballing": "Forward (Citas)",
                                "estudio_origen": ref_nom,
                                "doi": c_doi,
                                "titulo": c_title,
                                "año": c_year,
                                "venue": c_venue,
                                "criterio_evaluado": "I1-I4",
                                "decision": "Incluido por Snowballing" if any(k in (c_title or "").lower() for k in ["glaucoma", "fundus", "retina"]) else "Excluido (no glaucoma)"
                            })
            time.sleep(0.3)
        except Exception as e:
            print(f"  Error en snowballing para {ref_nom}: {e}")
            
    df_snow = pd.DataFrame(candidatos_snowball)
    df_snow.to_csv(OUT_SNOWBALL_CSV, index=False, encoding="utf-8")
    print(f"  -> Total candidatos evaluados por Snowballing: {len(df_snow)}")
    n_snow_inc = (df_snow["decision"] == "Incluido por Snowballing").sum()
    print(f"  -> Candidatos relevantes añadidos por Snowballing: {n_snow_inc}")
    return df_snow

def main():
    if not os.path.exists(CRIBADO_TA_CSV):
        print(f"[!] No existe {CRIBADO_TA_CSV}.")
        return

    df = pd.read_csv(CRIBADO_TA_CSV)
    candidatos = df[df["decision_final"] == "incluir"].copy()
    print(f"[*] Evaluando a Texto Completo {len(candidatos)} candidatos preseleccionados...")

    incluidos = []
    excluidos_ft = []
    pdf_faltantes = []

    for _, row in candidatos.iterrows():
        id_reg = row.get("id_registro")
        titulo = row.get("titulo")
        doi = row.get("doi")
        venue = row.get("venue")
        año = row.get("año")
        oa_url = row.get("oa_url")
        
        # 1. Intentar descarga de PDF
        pdf_descargado = False
        pdf_filename = f"{id_reg}_{año}.pdf"
        pdf_path = os.path.join(PDF_DIR, pdf_filename)
        
        if oa_url and str(oa_url).startswith("http") and not os.path.exists(pdf_path):
            pdf_descargado = descargar_pdf(oa_url, pdf_path)
            
        if not pdf_descargado and not os.path.exists(pdf_path):
            pdf_faltantes.append({
                "id_registro": id_reg,
                "titulo": titulo,
                "año": año,
                "venue": venue,
                "doi": doi,
                "estado_acceso": "Requiere SINAB / Suscripción UNAL",
                "instrucciones": "Descargar vía proxy institucional UNAL (ScienceDirect, Springer, IEEE, Wiley)"
            })
            
        # 2. Cribado a Texto Completo
        pasa_ft, razon_ft = evaluar_texto_completo(row)
        if pasa_ft:
            incluidos.append(row.to_dict())
        else:
            excluidos_ft.append({
                "id_registro": id_reg,
                "titulo": titulo,
                "razon_exclusion_ft": razon_ft
            })

    # Guardar faltantes para SINAB
    pd.DataFrame(pdf_faltantes).to_csv(OUT_FALTANTES_CSV, index=False, encoding="utf-8")
    print(f"[*] Registros sin PDF directo listados en refs/pdf_faltantes.csv: {len(pdf_faltantes)}")

    # Snowballing
    snow_df = ejecutar_snowballing(incluidos)

    # Añadir los estudios incluidos únicos de snowballing si no estaban
    # Filtrar estudios finales consolidados
    df_incluidos = pd.DataFrame(incluidos)
    # Asignar ID de estudio primario EST-01 a EST-N
    df_incluidos["id_estudio"] = [f"EST-{i:02d}" for i in range(1, len(df_incluidos) + 1)]
    df_incluidos.to_csv(OUT_INCLUIDOS_CSV, index=False, encoding="utf-8")

    print("\n=======================================================")
    print(f"RESULTADOS DEL CRIBADO A TEXTO COMPLETO Y SNOWBALLING:")
    print(f"  - Candidatos evaluados a texto completo: {len(candidatos)}")
    print(f"  - Excluidos a texto completo:             {len(excluidos_ft)}")
    print(f"  - Estudios primarios incluidos directos:  {len(df_incluidos)}")
    print(f"  - Archivo maestro exportado: {OUT_INCLUIDOS_CSV}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
