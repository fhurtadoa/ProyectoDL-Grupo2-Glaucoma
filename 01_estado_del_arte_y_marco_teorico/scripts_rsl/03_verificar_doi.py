#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 03: Verificación Rigurosa de DOI y Retracciones contra Crossref API
Proyecto: RSL Glaucoma DL - Capacidad de Generalización
Valida cada registro de data/processed/registros_deduplicados.csv contra la API oficial de Crossref.
Comprueba retracciones, correcciones y actualiza preprints con DOI formal si existe.
Exporta data/processed/registros_unicos.csv con columnas de verificación exhaustivas.
"""

import os
import time
import re
import string
import requests
import pandas as pd
from difflib import SequenceMatcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
IN_CSV = os.path.join(PROCESSED_DIR, "registros_deduplicados.csv")
OUT_CSV = os.path.join(PROCESSED_DIR, "registros_unicos.csv")

HEADERS = {
    "User-Agent": "RSL-Glaucoma-DL/1.0 (mailto:investigacion.glaucoma@unal.edu.co; Universidad Nacional de Colombia)"
}

def normalizar_titulo(titulo):
    if not titulo or pd.isna(titulo):
        return ""
    t = str(titulo).lower()
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.translate(str.maketrans("", "", string.punctuation + "“”‘’«»"))
    return re.sub(r"\s+", " ", t).strip()

def verificar_doi_crossref(doi):
    if not doi or pd.isna(doi) or not str(doi).strip():
        return None
    doi_clean = str(doi).strip().lower()
    doi_clean = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi_clean)
    url = f"https://api.crossref.org/works/{doi_clean}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code == 200:
            msg = r.json().get("message", {})
            title = (msg.get("title") or [""])[0]
            container = (msg.get("container-title") or [""])[0]
            pub_date = msg.get("published-print") or msg.get("published-online") or msg.get("issued") or {}
            parts = pub_date.get("date-parts", [[None]])[0]
            year = parts[0] if parts and parts[0] is not None else ""
            pub_type = msg.get("type", "")
            
            # Detectar retracciones y actualizaciones
            updates = msg.get("update-to", [])
            relation = msg.get("relation", {})
            es_retractado = False
            detalle_act = ""
            for u in updates:
                utype = u.get("type", "").lower()
                if "retract" in utype:
                    es_retractado = True
                    detalle_act = f"Retractado por {u.get('updated', {}).get('DOI', '')}"
                elif "correct" in utype or "errat" in utype:
                    detalle_act = f"Corrección/Erratum: {u.get('updated', {}).get('DOI', '')}"
            
            # Formatear autores
            authors = []
            for a in msg.get("author", []):
                given = a.get("given", "")
                family = a.get("family", "")
                if family:
                    authors.append(f"{family}, {given}".strip(", "))
            
            return {
                "crossref_valido": True,
                "crossref_doi": doi_clean,
                "crossref_titulo": title,
                "crossref_venue": container,
                "crossref_año": year,
                "crossref_tipo": pub_type,
                "crossref_autores": "; ".join(authors),
                "crossref_retractado": es_retractado,
                "crossref_detalle": detalle_act
            }
        elif r.status_code == 404:
            return {"crossref_valido": False, "crossref_doi": doi_clean, "crossref_detalle": "DOI no encontrado en Crossref"}
        else:
            return {"crossref_valido": False, "crossref_doi": doi_clean, "crossref_detalle": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"crossref_valido": False, "crossref_doi": doi_clean, "crossref_detalle": str(e)}

def buscar_titulo_crossref(titulo):
    if not titulo or len(titulo) < 15:
        return None
    url = "https://api.crossref.org/works"
    params = {"query.title": titulo, "rows": 1}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=12)
        if r.status_code == 200:
            items = r.json().get("message", {}).get("items", [])
            if items:
                cand = items[0]
                cand_title = (cand.get("title") or [""])[0]
                sim = SequenceMatcher(None, normalizar_titulo(titulo), normalizar_titulo(cand_title)).ratio()
                if sim >= 0.90:
                    cand_doi = cand.get("DOI", "")
                    cand_venue = (cand.get("container-title") or [""])[0]
                    return {
                        "encontrado": True,
                        "doi_publicado": cand_doi,
                        "venue_publicado": cand_venue,
                        "titulo_publicado": cand_title
                    }
    except Exception:
        pass
    return None

def main():
    if not os.path.exists(IN_CSV):
        print(f"[!] No existe {IN_CSV}. Ejecute scripts/02_unir_deduplicar.py primero.")
        return

    df = pd.read_csv(IN_CSV)
    print(f"[*] Iniciando verificación de DOIs en Crossref para {len(df)} registros...")
    
    validados = []
    total = len(df)
    
    for i, row in df.iterrows():
        id_reg = row.get("id_registro", f"REG-{i+1:04d}")
        doi = str(row.get("doi", "")).strip() if pd.notna(row.get("doi")) else ""
        titulo = str(row.get("titulo", "")).strip() if pd.notna(row.get("titulo")) else ""
        is_ret_source = bool(row.get("is_retracted", False))
        
        info_cr = None
        if doi and doi.lower() not in ["none", "nan", ""]:
            info_cr = verificar_doi_crossref(doi)
            time.sleep(0.1) # Respetar límites
        elif titulo:
            # Intentar encontrar si un preprint ya fue publicado en revista
            match_pub = buscar_titulo_crossref(titulo)
            if match_pub and match_pub.get("encontrado"):
                doi = match_pub["doi_publicado"]
                info_cr = verificar_doi_crossref(doi)
                if info_cr:
                    info_cr["crossref_detalle"] = f"Preprint vinculado a publicación formal: {match_pub['venue_publicado']}"
                time.sleep(0.1)
        
        if info_cr and info_cr.get("crossref_valido"):
            es_ret = is_ret_source or info_cr.get("crossref_retractado", False)
            estado_doi = "RETRACTADO" if es_ret else "VERIFICADO"
            validados.append({
                "id_registro": id_reg,
                "titulo": info_cr.get("crossref_titulo") or titulo,
                "autores": info_cr.get("crossref_autores") or row.get("autores", ""),
                "año": info_cr.get("crossref_año") or row.get("año", ""),
                "venue": info_cr.get("crossref_venue") or row.get("venue", ""),
                "doi": info_cr.get("crossref_doi") or doi,
                "resumen": row.get("resumen", ""),
                "fuentes": row.get("fuentes", ""),
                "estado_doi": estado_doi,
                "tipo_publicacion": info_cr.get("crossref_tipo", ""),
                "detalles_verificacion": info_cr.get("crossref_detalle", "DOI validado exitosamente en Crossref"),
                "is_retracted": es_ret,
                "oa_url": row.get("oa_url", "")
            })
        elif doi and doi.lower() not in ["none", "nan", ""]:
            validados.append({
                "id_registro": id_reg,
                "titulo": titulo,
                "autores": row.get("autores", ""),
                "año": row.get("año", ""),
                "venue": row.get("venue", ""),
                "doi": doi,
                "resumen": row.get("resumen", ""),
                "fuentes": row.get("fuentes", ""),
                "estado_doi": "DOI NO ENCONTRADO EN CROSSREF",
                "tipo_publicacion": "desconocido",
                "detalles_verificacion": info_cr.get("crossref_detalle", "Falla de resolución en Crossref") if info_cr else "Sin respuesta",
                "is_retracted": is_ret_source,
                "oa_url": row.get("oa_url", "")
            })
        else:
            validados.append({
                "id_registro": id_reg,
                "titulo": titulo,
                "autores": row.get("autores", ""),
                "año": row.get("año", ""),
                "venue": row.get("venue", "arXiv / Preprint"),
                "doi": "SIN DOI – VERIFICAR",
                "resumen": row.get("resumen", ""),
                "fuentes": row.get("fuentes", ""),
                "estado_doi": "SIN DOI – VERIFICAR",
                "tipo_publicacion": "preprint_o_manuscrito",
                "detalles_verificacion": "Registro sin DOI formal identificado",
                "is_retracted": is_ret_source,
                "oa_url": row.get("oa_url", "")
            })
        
        if (i + 1) % 25 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] Registros verificados...")

    out_df = pd.DataFrame(validados)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    
    n_verif = (out_df["estado_doi"] == "VERIFICADO").sum()
    n_ret = (out_df["estado_doi"] == "RETRACTADO").sum()
    n_no_cr = (out_df["estado_doi"] == "DOI NO ENCONTRADO EN CROSSREF").sum()
    n_sin_doi = (out_df["estado_doi"] == "SIN DOI – VERIFICAR").sum()
    
    print("\n=======================================================")
    print(f"RESULTADOS DE VERIFICACIÓN EN CROSSREF:")
    print(f"  - DOIs Verificados con éxito: {n_verif}")
    print(f"  - Registros Retractados:       {n_ret}")
    print(f"  - DOI no hallado en Crossref:  {n_no_cr}")
    print(f"  - Sin DOI (Preprints/otros):   {n_sin_doi}")
    print(f"  -> Total exportado en data/processed/registros_unicos.csv: {len(out_df)}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
