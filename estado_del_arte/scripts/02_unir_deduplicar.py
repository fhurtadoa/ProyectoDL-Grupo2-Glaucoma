#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 02: Unión y Deduplicación de Registros Científicos
Proyecto: RSL Glaucoma DL - Capacidad de Generalización
Lee los JSON y CSV de data/raw/ generados en el Paso 3, normaliza DOIs y títulos,
deduplica por DOI exacto y por similitud de título >= 0.92, fusionando metadatos.
Exporta data/processed/registros_deduplicados.csv.
"""

import os
import glob
import re
import string
import json
from difflib import SequenceMatcher
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def normalizar_doi(doi):
    if not doi or pd.isna(doi):
        return ""
    doi = str(doi).strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = doi.strip("/ ")
    return doi

def normalizar_titulo(titulo):
    if not titulo or pd.isna(titulo):
        return ""
    t = str(titulo).lower()
    # Eliminar etiquetas html o xml
    t = re.sub(r"<[^>]+>", " ", t)
    # Quitar puntuación y caracteres raros
    t = t.translate(str.maketrans("", "", string.punctuation + "“”‘’«»"))
    # Colapsar espacios
    t = re.sub(r"\s+", " ", t).strip()
    return t

def similitud_titulos(t1, t2):
    if not t1 or not t2:
        return 0.0
    return SequenceMatcher(None, t1, t2).ratio()

def cargar_datos_crudos():
    archivos = glob.glob(os.path.join(RAW_DIR, "*.json"))
    todos_registros = []
    print(f"[*] Encontrados {len(archivos)} archivos JSON en data/raw/:")
    for f in archivos:
        try:
            with open(f, "r", encoding="utf-8") as jf:
                items = json.load(jf)
                print(f"  - {os.path.basename(f)}: {len(items)} registros")
                todos_registros.extend(items)
        except Exception as e:
            print(f"  Error leyendo {f}: {e}")
    return todos_registros

def fusionar_registros(r_base, r_nuevo):
    # Fusionar fuentes
    fuentes = set(r_base.get("fuentes", "").split("; "))
    fuentes.add(r_nuevo.get("fuente", ""))
    r_base["fuentes"] = "; ".join(sorted([f for f in fuentes if f]))
    
    # Preservar DOI si falta
    if not r_base.get("doi") and r_nuevo.get("doi"):
        r_base["doi"] = r_nuevo["doi"]
    # Preservar resumen más largo
    if len(str(r_nuevo.get("resumen", ""))) > len(str(r_base.get("resumen", ""))):
        r_base["resumen"] = r_nuevo["resumen"]
    # Preservar OA URL si existe
    if not r_base.get("oa_url") and r_nuevo.get("oa_url"):
        r_base["oa_url"] = r_nuevo["oa_url"]
    # Preservar venue si falta
    if not r_base.get("venue") and r_nuevo.get("venue"):
        r_base["venue"] = r_nuevo["venue"]
    # Marcar retractado si cualquiera lo indica
    if r_nuevo.get("is_retracted", False):
        r_base["is_retracted"] = True

def deduplicar(registros):
    print(f"\n[*] Iniciando proceso de deduplicación sobre {len(registros)} registros brutos...")
    
    # Normalizar campos
    for r in registros:
        r["doi_norm"] = normalizar_doi(r.get("doi", ""))
        r["titulo_norm"] = normalizar_titulo(r.get("titulo", ""))
        r["fuentes"] = r.get("fuente", "")

    unicos = []
    doi_map = {} # doi_norm -> indice en unicos
    
    # 1. Deduplicación por DOI exacto
    for r in registros:
        d = r["doi_norm"]
        if d and d in doi_map:
            idx = doi_map[d]
            fusionar_registros(unicos[idx], r)
        else:
            if d:
                doi_map[d] = len(unicos)
            unicos.append(r)
            
    print(f"  -> Tras deduplicación por DOI: {len(unicos)} registros únicos.")

    # 2. Deduplicación por similitud de título >= 0.92
    final_unicos = []
    fusionados_por_titulo = 0
    
    for r in unicos:
        t_cand = r["titulo_norm"]
        if not t_cand:
            final_unicos.append(r)
            continue
        
        match_idx = None
        for i, ref in enumerate(final_unicos):
            t_ref = ref["titulo_norm"]
            sim = similitud_titulos(t_cand, t_ref)
            if sim >= 0.92:
                match_idx = i
                break
        
        if match_idx is not None:
            fusionar_registros(final_unicos[match_idx], r)
            fusionados_por_titulo += 1
        else:
            final_unicos.append(r)

    print(f"  -> Fusionados adicionales por similitud de título (>= 0.92): {fusionados_por_titulo}")
    print(f"  -> Total final de registros únicos deduplicados: {len(final_unicos)}")

    # Asignar IDs unívocos
    for i, r in enumerate(final_unicos, 1):
        r["id_registro"] = f"REG-{i:04d}"

    out_csv = os.path.join(PROCESSED_DIR, "registros_deduplicados.csv")
    df = pd.DataFrame(final_unicos)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[*] Guardado en: {out_csv}")
    return final_unicos

if __name__ == "__main__":
    registros = cargar_datos_crudos()
    if registros:
        deduplicar(registros)
    else:
        print("[!] No hay registros crudos para procesar.")
