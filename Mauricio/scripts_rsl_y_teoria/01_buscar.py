#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 01: Búsqueda Programática Reproducible en APIs Científicas
Proyecto: RSL Glaucoma DL - Capacidad de Generalización
Consulta: OpenAlex, PubMed (E-utilities), Europe PMC, Semantic Scholar y arXiv.
Guarda resultados crudos en data/raw/ con fecha de ejecución.
"""

import os
import sys
import json
import time
import glob
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
import requests
import pandas as pd

TODAY_STR = datetime.now().strftime("%Y%m%d")
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)
MANUAL_DIR = os.path.join(RAW_DIR, "manual")
os.makedirs(MANUAL_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "RSL-Glaucoma-DL/1.0 (mailto:investigacion.glaucoma@unal.edu.co; Universidad Nacional de Colombia)"
}

def buscar_openalex(max_results=200):
    print("\n[1/5] Consultando OpenAlex API (Polite Pool)...")
    url_base = "https://api.openalex.org/works"
    query_search = 'glaucoma "deep learning" fundus ("domain shift" OR "cross-dataset" OR "domain generalization" OR "domain adaptation" OR "external validation" OR "harmonization")'
    params = {
        "search": query_search,
        "filter": "from_publication_date:2016-01-01,has_doi:true",
        "per-page": 50,
        "page": 1,
        "mailto": "investigacion.glaucoma@unal.edu.co"
    }
    all_results = []
    page = 1
    total_found = 0
    while len(all_results) < max_results:
        params["page"] = page
        try:
            r = requests.get(url_base, params=params, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                print(f"  Aviso: OpenAlex retornó código HTTP {r.status_code}")
                break
            data = r.json()
            if page == 1:
                total_found = data.get("meta", {}).get("count", 0)
                print(f"  Total registros reportados por OpenAlex para la consulta: {total_found}")
            works = data.get("results", [])
            if not works:
                break
            for w in works:
                doi = w.get("doi")
                if doi and doi.startswith("https://doi.org/"):
                    doi = doi.replace("https://doi.org/", "").strip()
                title = w.get("title") or ""
                year = w.get("publication_year")
                venue = (w.get("primary_location") or {}).get("source", {})
                venue_name = venue.get("display_name") if venue else ""
                authors = [a.get("author", {}).get("display_name") for a in w.get("authorships", []) if a.get("author")]
                abstract_inverted = w.get("abstract_inverted_index")
                abstract = ""
                if abstract_inverted:
                    word_positions = []
                    for word, positions in abstract_inverted.items():
                        for pos in positions:
                            word_positions.append((pos, word))
                    word_positions.sort()
                    abstract = " ".join([w[1] for w in word_positions])
                is_retracted = w.get("is_retracted", False)
                oa_url = (w.get("open_access") or {}).get("oa_url", "")
                all_results.append({
                    "id_origen": w.get("id"),
                    "fuente": "OpenAlex",
                    "doi": doi or "",
                    "titulo": title.strip(),
                    "autores": "; ".join(authors),
                    "año": year,
                    "venue": venue_name,
                    "resumen": abstract,
                    "is_retracted": is_retracted,
                    "oa_url": oa_url or "",
                    "fecha_consulta": TODAY_STR
                })
            page += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"  Error consultando OpenAlex en página {page}: {e}")
            break

    print(f"  -> Recuperados de OpenAlex: {len(all_results)} registros.")
    json_path = os.path.join(RAW_DIR, f"openalex_{TODAY_STR}.json")
    csv_path = os.path.join(RAW_DIR, f"openalex_{TODAY_STR}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    pd.DataFrame(all_results).to_csv(csv_path, index=False, encoding="utf-8")
    return all_results

def buscar_pubmed(max_results=200):
    print("\n[2/5] Consultando PubMed E-utilities API...")
    query = (
        '(glaucoma[Title/Abstract] OR "glaucomatous optic neuropathy"[Title/Abstract]) AND '
        '("deep learning"[Title/Abstract] OR "convolutional neural network"[Title/Abstract] OR "CNN"[Title/Abstract] OR "transfer learning"[Title/Abstract] OR "foundation model"[Title/Abstract]) AND '
        '("fundus"[Title/Abstract] OR "retinal photograph"[Title/Abstract] OR "retinography"[Title/Abstract] OR "color fundus"[Title/Abstract]) AND '
        '("domain shift"[Title/Abstract] OR "dataset shift"[Title/Abstract] OR "cross-dataset"[Title/Abstract] OR "domain generalization"[Title/Abstract] OR "domain adaptation"[Title/Abstract] OR "external validation"[Title/Abstract] OR "generalization"[Title/Abstract] OR "multi-center"[Title/Abstract] OR "harmonization"[Title/Abstract]) AND '
        '("2016/01/01"[Date - Publication] : "2026/11/17"[Date - Publication])'
    )
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results
    }
    all_results = []
    try:
        r = requests.get(esearch_url, params=params, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            print(f"  Aviso: PubMed esearch retornó {r.status_code}")
            return []
        data = r.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        total_found = data.get("esearchresult", {}).get("count", 0)
        print(f"  Total registros reportados por PubMed: {total_found}; IDs recuperados: {len(id_list)}")
        
        # Consultar resumen en bloques de 50
        esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        chunk_size = 50
        for i in range(0, len(id_list), chunk_size):
            chunk = id_list[i:i+chunk_size]
            sum_params = {
                "db": "pubmed",
                "id": ",".join(chunk),
                "retmode": "json"
            }
            sr = requests.get(esummary_url, params=sum_params, headers=HEADERS, timeout=20)
            if sr.status_code == 200:
                sdata = sr.json().get("result", {})
                for pmid in chunk:
                    pdoc = sdata.get(pmid)
                    if not pdoc:
                        continue
                    title = pdoc.get("title", "")
                    pubdate = pdoc.get("pubdate", "")
                    year = pubdate.split(" ")[0] if pubdate else ""
                    source = pdoc.get("source", "")
                    authors = [a.get("name", "") for a in pdoc.get("authors", [])]
                    articleids = pdoc.get("articleids", [])
                    doi = ""
                    for aid in articleids:
                        if aid.get("idtype") == "doi":
                            doi = aid.get("value", "")
                            break
                    all_results.append({
                        "id_origen": f"PMID:{pmid}",
                        "fuente": "PubMed",
                        "doi": doi.strip(),
                        "titulo": title.strip().rstrip("."),
                        "autores": "; ".join(authors),
                        "año": year,
                        "venue": source,
                        "resumen": "",
                        "is_retracted": False,
                        "oa_url": "",
                        "fecha_consulta": TODAY_STR
                    })
            time.sleep(0.4)
    except Exception as e:
        print(f"  Error consultando PubMed: {e}")

    print(f"  -> Recuperados de PubMed: {len(all_results)} registros.")
    json_path = os.path.join(RAW_DIR, f"pubmed_{TODAY_STR}.json")
    csv_path = os.path.join(RAW_DIR, f"pubmed_{TODAY_STR}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    pd.DataFrame(all_results).to_csv(csv_path, index=False, encoding="utf-8")
    return all_results

def buscar_europepmc(max_results=200):
    print("\n[3/5] Consultando Europe PMC API...")
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    query = (
        '(TITLE_ABS:"glaucoma" OR TITLE_ABS:"glaucomatous") AND '
        '(TITLE_ABS:"deep learning" OR TITLE_ABS:"convolutional" OR TITLE_ABS:"CNN" OR TITLE_ABS:"transfer learning") AND '
        '(TITLE_ABS:"fundus" OR TITLE_ABS:"retinal photograph" OR TITLE_ABS:"retinography") AND '
        '(TITLE_ABS:"domain shift" OR TITLE_ABS:"cross-dataset" OR TITLE_ABS:"generalization" OR TITLE_ABS:"external validation" OR TITLE_ABS:"domain adaptation" OR TITLE_ABS:"harmonization") AND '
        '(FIRST_PDATE:[2016-01-01 TO 2026-11-17])'
    )
    params = {
        "query": query,
        "format": "json",
        "pageSize": 50,
        "resultType": "core"
    }
    all_results = []
    cursor_mark = "*"
    while len(all_results) < max_results:
        params["cursorMark"] = cursor_mark
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                print(f"  Aviso: Europe PMC retornó {r.status_code}")
                break
            data = r.json()
            if cursor_mark == "*":
                print(f"  Total registros reportados por Europe PMC: {data.get('hitCount', 0)}")
            results = data.get("resultList", {}).get("result", [])
            if not results:
                break
            for item in results:
                doi = item.get("doi", "")
                title = item.get("title", "")
                year = item.get("pubYear", "")
                journal = item.get("journalTitle") or (item.get("journalInfo", {}).get("journal", {})).get("title", "")
                author_str = item.get("authorString", "")
                abstract = item.get("abstractText", "")
                is_retracted = item.get("hasRetraction", "N") == "Y"
                full_text_urls = item.get("fullTextUrlList", {}).get("fullTextUrl", [])
                oa_url = ""
                for u in full_text_urls:
                    if u.get("availability") == "Free" or u.get("documentStyle") == "pdf":
                        oa_url = u.get("url", "")
                        break
                all_results.append({
                    "id_origen": item.get("id"),
                    "fuente": "EuropePMC",
                    "doi": doi.strip(),
                    "titulo": title.strip().rstrip("."),
                    "autores": author_str,
                    "año": year,
                    "venue": journal,
                    "resumen": abstract,
                    "is_retracted": is_retracted,
                    "oa_url": oa_url,
                    "fecha_consulta": TODAY_STR
                })
            next_cursor = data.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor_mark:
                break
            cursor_mark = next_cursor
            time.sleep(0.4)
        except Exception as e:
            print(f"  Error consultando Europe PMC: {e}")
            break

    print(f"  -> Recuperados de Europe PMC: {len(all_results)} registros.")
    json_path = os.path.join(RAW_DIR, f"europepmc_{TODAY_STR}.json")
    csv_path = os.path.join(RAW_DIR, f"europepmc_{TODAY_STR}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    pd.DataFrame(all_results).to_csv(csv_path, index=False, encoding="utf-8")
    return all_results

def buscar_semanticscholar(max_results=50):
    print("\n[4/5] Consultando Semantic Scholar API...")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    query = 'glaucoma "deep learning" fundus "domain shift"'
    params = {
        "query": query,
        "limit": 30,
        "year": "2016-2026",
        "fields": "paperId,title,year,authors,venue,abstract,externalIds,isOpenAccess,openAccessPdf"
    }
    all_results = []
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            papers = data.get("data", [])
            print(f"  Total reportado por Semantic Scholar: {data.get('total', 0)}; recibidos: {len(papers)}")
            for p in papers:
                doi = (p.get("externalIds") or {}).get("DOI", "")
                authors = [a.get("name") for a in p.get("authors", []) if a.get("name")]
                oa_url = (p.get("openAccessPdf") or {}).get("url", "")
                all_results.append({
                    "id_origen": p.get("paperId"),
                    "fuente": "SemanticScholar",
                    "doi": doi.strip(),
                    "titulo": (p.get("title") or "").strip(),
                    "autores": "; ".join(authors),
                    "año": p.get("year"),
                    "venue": p.get("venue") or "",
                    "resumen": p.get("abstract") or "",
                    "is_retracted": False,
                    "oa_url": oa_url,
                    "fecha_consulta": TODAY_STR
                })
        elif r.status_code == 429:
            print("  Aviso: Semantic Scholar público reportó límite de tasa HTTP 429 (Throttling).")
        else:
            print(f"  Aviso: Semantic Scholar retornó {r.status_code}")
    except Exception as e:
        print(f"  Error consultando Semantic Scholar: {e}")

    print(f"  -> Recuperados de Semantic Scholar: {len(all_results)} registros.")
    json_path = os.path.join(RAW_DIR, f"semanticscholar_{TODAY_STR}.json")
    csv_path = os.path.join(RAW_DIR, f"semanticscholar_{TODAY_STR}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    pd.DataFrame(all_results).to_csv(csv_path, index=False, encoding="utf-8")
    return all_results

def buscar_arxiv(max_results=50):
    print("\n[5/5] Consultando arXiv API (Preprints para rastreo a publicación formal)...")
    url = "http://export.arxiv.org/api/query"
    query = 'all:glaucoma AND (all:"deep learning" OR all:"CNN") AND all:fundus AND (all:"domain shift" OR all:"domain adaptation" OR all:"generalization")'
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    all_results = []
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            root = ET.fromstring(r.text)
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
            entries = root.findall("atom:entry", ns)
            print(f"  Entradas devueltas por arXiv: {len(entries)}")
            for e in entries:
                arxiv_id = e.find("atom:id", ns).text.strip()
                title = e.find("atom:title", ns).text.strip().replace("\n", " ")
                summary = e.find("atom:summary", ns).text.strip().replace("\n", " ")
                published = e.find("atom:published", ns).text.strip()
                year = published[:4] if published else ""
                authors = [a.find("atom:name", ns).text.strip() for a in e.findall("atom:author", ns)]
                doi_elem = e.find("arxiv:doi", ns)
                doi = doi_elem.text.strip() if doi_elem is not None else ""
                pdf_url = ""
                for link in e.findall("atom:link", ns):
                    if link.attrib.get("title") == "pdf":
                        pdf_url = link.attrib.get("href", "")
                        break
                all_results.append({
                    "id_origen": arxiv_id,
                    "fuente": "arXiv",
                    "doi": doi,
                    "titulo": title,
                    "autores": "; ".join(authors),
                    "año": year,
                    "venue": "arXiv preprint",
                    "resumen": summary,
                    "is_retracted": False,
                    "oa_url": pdf_url or arxiv_id,
                    "fecha_consulta": TODAY_STR
                })
        else:
            print(f"  Aviso: arXiv retornó código HTTP {r.status_code}")
    except Exception as e:
        print(f"  Error consultando arXiv: {e}")

    print(f"  -> Recuperados de arXiv: {len(all_results)} registros.")
    json_path = os.path.join(RAW_DIR, f"arxiv_{TODAY_STR}.json")
    csv_path = os.path.join(RAW_DIR, f"arxiv_{TODAY_STR}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    pd.DataFrame(all_results).to_csv(csv_path, index=False, encoding="utf-8")
    return all_results

def integrar_manuales():
    manual_files = glob.glob(os.path.join(MANUAL_DIR, "*.csv"))
    if not manual_files:
        print("\n[*] No se encontraron archivos de exportación manual en data/raw/manual/.")
        return []
    print(f"\n[*] Integrando {len(manual_files)} archivos manuales desde data/raw/manual/...")
    manual_records = []
    for mf in manual_files:
        try:
            df = pd.read_csv(mf)
            print(f"  Archivo cargado: {os.path.basename(mf)} con {len(df)} filas.")
            # Normalizar columnas típicas de Scopus / WoS / IEEE
            for _, row in df.iterrows():
                manual_records.append({
                    "id_origen": f"MANUAL:{os.path.basename(mf)}_{_}",
                    "fuente": "ManualExport",
                    "doi": str(row.get("DOI", row.get("doi", ""))).strip(),
                    "titulo": str(row.get("Title", row.get("titulo", ""))).strip(),
                    "autores": str(row.get("Authors", row.get("autores", ""))),
                    "año": str(row.get("Year", row.get("año", ""))),
                    "venue": str(row.get("Source title", row.get("venue", ""))),
                    "resumen": str(row.get("Abstract", row.get("resumen", ""))),
                    "is_retracted": False,
                    "oa_url": "",
                    "fecha_consulta": TODAY_STR
                })
        except Exception as e:
            print(f"  Error leyendo manual {mf}: {e}")
    return manual_records

def main():
    print(f"=== INICIANDO BÚSQUEDA PROGRAMÁTICA REPRODUCIBLE ({TODAY_STR}) ===")
    r_openalex = buscar_openalex(max_results=200)
    r_pubmed = buscar_pubmed(max_results=200)
    r_europepmc = buscar_europepmc(max_results=200)
    r_s2 = buscar_semanticscholar(max_results=50)
    r_arxiv = buscar_arxiv(max_results=50)
    r_manual = integrar_manuales()

    total_bruto = len(r_openalex) + len(r_pubmed) + len(r_europepmc) + len(r_s2) + len(r_arxiv) + len(r_manual)
    print("\n=======================================================")
    print(f"RESUMEN DE RESULTADOS CRUDOS EXTRAÍDOS:")
    print(f"  - OpenAlex:         {len(r_openalex)}")
    print(f"  - PubMed:           {len(r_pubmed)}")
    print(f"  - Europe PMC:       {len(r_europepmc)}")
    print(f"  - Semantic Scholar: {len(r_s2)}")
    print(f"  - arXiv (Preprint): {len(r_arxiv)}")
    print(f"  - Manual (Scopus/WoS/IEEE): {len(r_manual)}")
    print(f"  -> TOTAL REGISTROS CRUDOS: {total_bruto}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
