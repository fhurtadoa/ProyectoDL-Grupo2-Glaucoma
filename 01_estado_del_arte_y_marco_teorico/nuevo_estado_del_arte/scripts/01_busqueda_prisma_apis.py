"""
Fase 4 adelantada (PRISMA 2020 - Vía 1: Bases de datos y registros) + Auditoría de Moondream y R7.
Ejecuta de forma 100% reproducible y auditable las consultas booleanas en:
  1. OpenAlex API
  2. PubMed (NCBI E-utilities API)
  3. Europe PMC REST API
Y consulta la API oficial de Hugging Face para documentar de forma verificada el origen y metadatos
de `moondream/glaucoma-detection`.

Guarda:
  - busqueda_prisma/raw/*.json (respuestas crudas con timestamp)
  - busqueda_prisma/log_busqueda_prisma.json (ecuaciones exactas, fechas, conteos crudos)
  - busqueda_prisma/registros_crudas_bases.csv (todos los registros recuperados antes de deduplicar)
  - busqueda_prisma/registros_bases_deduplicados.csv (deduplicados por DOI normalizado y título normalizado)
  - busqueda_prisma/auditoria_moondream_hf.json y .md
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "busqueda_prisma"
RAW_DIR = OUT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "ProyectoDL-UNAL-Glaucoma-SLR/2.0 (mailto:jtoscanoa@unal.edu.co)"
}
CUTOFF_DATE = "2026-11-24"


def http_get_json(url: str, timeout: int = 60, retries: int = 3) -> dict | list:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return {}


def http_get_text(url: str, timeout: int = 60, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return ""


def norm_doi(doi: str | None) -> str:
    if not doi:
        return ""
    d = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "https://dx.doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d.strip()


def norm_title(title: str | None) -> str:
    if not title:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def reconstruct_openalex_abstract(inv_idx: dict | None) -> str:
    if not inv_idx or not isinstance(inv_idx, dict):
        return ""
    pos_map = {}
    for word, positions in inv_idx.items():
        for p in positions:
            pos_map[p] = word
    if not pos_map:
        return ""
    return " ".join(pos_map[i] for i in sorted(pos_map.keys()))


def search_openalex() -> tuple[list[dict], dict]:
    query = (
        '(glaucoma OR "glaucomatous optic neuropathy") AND '
        '("deep learning" OR "convolutional neural network" OR CNN OR "transfer learning" OR "foundation model" OR RETFound) AND '
        '("fundus" OR "retinal photograph" OR retinography) AND '
        '("domain shift" OR "dataset shift" OR "cross-dataset" OR "domain generalization" OR "domain adaptation" OR "external validation" OR "out-of-domain" OR "multi-center" OR "harmonization")'
    )
    filter_str = f"title_and_abstract.search:{query},from_publication_date:2016-01-01,has_doi:true"
    records = []
    raw_pages = []
    page = 1
    per_page = 200
    while True:
        params = {
            "filter": filter_str,
            "per-page": str(per_page),
            "page": str(page),
            "mailto": "jtoscanoa@unal.edu.co",
        }
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        data = http_get_json(url)
        raw_pages.append(data)
        results = data.get("results", [])
        for w in results:
            authors = ", ".join(
                a.get("author", {}).get("display_name", "")
                for a in w.get("authorships", [])[:10]
                if a.get("author", {}).get("display_name")
            )
            loc = w.get("primary_location") or {}
            src = loc.get("source") or {}
            records.append({
                "fuente_bd": "OpenAlex",
                "id_fuente": w.get("id", ""),
                "doi": norm_doi(w.get("doi", "")),
                "pmid": norm_doi((w.get("ids") or {}).get("pmid", "")).replace("https://pubmed.ncbi.nlm.nih.gov/", "").strip("/"),
                "titulo": (w.get("title") or "").strip(),
                "autores": authors,
                "anio": w.get("publication_year") or "",
                "revista_o_congreso": src.get("display_name", ""),
                "tipo_documento": w.get("type", ""),
                "acceso_abierto": bool((w.get("open_access") or {}).get("is_oa")),
                "oa_url": (w.get("open_access") or {}).get("oa_url") or "",
                "resumen": reconstruct_openalex_abstract(w.get("abstract_inverted_index")),
            })
        meta = data.get("meta", {})
        if len(results) < per_page or page * per_page >= meta.get("count", 0) or page >= 10:
            break
        page += 1
        time.sleep(0.2)

    (RAW_DIR / "openalex_raw.json").write_text(json.dumps(raw_pages, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_log = {
        "database": "OpenAlex",
        "query": query,
        "filter": filter_str,
        "retrieved_count": len(records),
    }
    return records, meta_log


def search_pubmed() -> tuple[list[dict], dict]:
    query = (
        '(glaucoma[Title/Abstract] OR "glaucomatous optic neuropathy"[Title/Abstract]) AND '
        '("deep learning"[Title/Abstract] OR "convolutional neural network"[Title/Abstract] OR "CNN"[Title/Abstract] OR '
        '"transfer learning"[Title/Abstract] OR "foundation model"[Title/Abstract] OR "RETFound"[Title/Abstract]) AND '
        '("fundus"[Title/Abstract] OR "retinal photograph"[Title/Abstract] OR "retinography"[Title/Abstract] OR "color fundus"[Title/Abstract]) AND '
        '("domain shift"[Title/Abstract] OR "dataset shift"[Title/Abstract] OR "cross-dataset"[Title/Abstract] OR '
        '"domain generalization"[Title/Abstract] OR "domain adaptation"[Title/Abstract] OR "external validation"[Title/Abstract] OR '
        '"generalization"[Title/Abstract] OR "multi-center"[Title/Abstract] OR "harmonization"[Title/Abstract]) AND '
        '("2016/01/01"[Date - Publication] : "2026/11/24"[Date - Publication])'
    )
    esearch_params = {
        "db": "pubmed",
        "term": query,
        "retmax": "1000",
        "retmode": "json",
    }
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(esearch_params)
    esearch_data = http_get_json(esearch_url)
    (RAW_DIR / "pubmed_esearch.json").write_text(json.dumps(esearch_data, ensure_ascii=False, indent=2), encoding="utf-8")
    id_list = esearch_data.get("esearchresult", {}).get("idlist", [])

    records = []
    xml_chunks = []
    batch_size = 150
    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i + batch_size]
        efetch_params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
        }
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urllib.parse.urlencode(efetch_params)
        xml_txt = http_get_text(efetch_url)
        xml_chunks.append(xml_txt)
        root = ET.fromstring(xml_txt)
        for art in root.findall(".//PubmedArticle"):
            pmid = (art.findtext(".//MedlineCitation/PMID") or "").strip()
            title = "".join(art.find(".//ArticleTitle").itertext()).strip() if art.find(".//ArticleTitle") is not None else ""
            abs_elems = art.findall(".//Abstract/AbstractText")
            abstract = " ".join("".join(el.itertext()).strip() for el in abs_elems if el is not None)
            journal = (art.findtext(".//Journal/Title") or "").strip()
            year = (
                art.findtext(".//Journal/JournalIssue/PubDate/Year")
                or art.findtext(".//Journal/JournalIssue/PubDate/MedlineDate", "")[:4]
                or ""
            )
            authors_list = []
            for au in art.findall(".//AuthorList/Author")[:10]:
                ln = au.findtext("LastName") or ""
                fn = au.findtext("ForeName") or au.findtext("Initials") or ""
                if ln:
                    authors_list.append(f"{fn} {ln}".strip())
            doi = ""
            for aid in art.findall(".//PubmedData/ArticleIdList/ArticleId"):
                if aid.attrib.get("IdType") == "doi" and aid.text:
                    doi = norm_doi(aid.text)
                    break
            pub_types = [pt.text for pt in art.findall(".//PublicationTypeList/PublicationType") if pt.text]
            records.append({
                "fuente_bd": "PubMed",
                "id_fuente": f"PMID:{pmid}",
                "doi": doi,
                "pmid": pmid,
                "titulo": title,
                "autores": ", ".join(authors_list),
                "anio": year,
                "revista_o_congreso": journal,
                "tipo_documento": "; ".join(pub_types),
                "acceso_abierto": False,
                "oa_url": "",
                "resumen": abstract,
            })
        time.sleep(0.35)

    (RAW_DIR / "pubmed_efetch.xml").write_text("\n".join(xml_chunks), encoding="utf-8")
    meta_log = {
        "database": "PubMed",
        "query": query,
        "retrieved_count": len(records),
    }
    return records, meta_log


def search_europepmc() -> tuple[list[dict], dict]:
    query = (
        '(TITLE_ABS:"glaucoma" OR TITLE_ABS:"glaucomatous optic neuropathy") AND '
        '(TITLE_ABS:"deep learning" OR TITLE_ABS:"convolutional" OR TITLE_ABS:"CNN" OR TITLE_ABS:"transfer learning" OR TITLE_ABS:"foundation model" OR TITLE_ABS:"RETFound") AND '
        '(TITLE_ABS:"fundus" OR TITLE_ABS:"retinal photograph" OR TITLE_ABS:"retinography") AND '
        '(TITLE_ABS:"domain shift" OR TITLE_ABS:"cross-dataset" OR TITLE_ABS:"generalization" OR TITLE_ABS:"external validation" OR TITLE_ABS:"domain adaptation" OR TITLE_ABS:"harmonization" OR TITLE_ABS:"multi-center") AND '
        '(SRC:MED OR SRC:PMC) AND (FIRST_PDATE:[2016-01-01 TO 2026-11-24])'
    )
    records = []
    raw_pages = []
    cursor = "*"
    while True:
        params = {
            "query": query,
            "resultType": "core",
            "pageSize": "25",
            "cursorMark": cursor,
            "format": "json",
        }
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(params)
        data = http_get_json(url)
        raw_pages.append(data)
        result_list = (data.get("resultList") or {}).get("result", [])
        for r in result_list:
            pmcid = r.get("pmcid", "")
            is_oa = r.get("isOpenAccess") == "Y"
            oa_url = f"https://europepmc.org/articles/{pmcid}" if pmcid and is_oa else ""
            records.append({
                "fuente_bd": "EuropePMC",
                "id_fuente": r.get("id", ""),
                "doi": norm_doi(r.get("doi", "")),
                "pmid": r.get("pmid", ""),
                "titulo": (r.get("title") or "").strip(),
                "autores": (r.get("authorString") or "").strip(),
                "anio": r.get("pubYear", ""),
                "revista_o_congreso": (r.get("journalInfo") or {}).get("journal", {}).get("title", "") or r.get("journalTitle", ""),
                "tipo_documento": "; ".join(r.get("pubTypeList", {}).get("pubType", [])) if isinstance(r.get("pubTypeList"), dict) else "",
                "acceso_abierto": is_oa,
                "oa_url": oa_url,
                "resumen": (r.get("abstractText") or "").strip(),
            })
        next_cursor = data.get("nextCursorMark")
        if not result_list or not next_cursor or next_cursor == cursor or len(records) >= 2000:
            break
        cursor = next_cursor
        time.sleep(0.25)

    (RAW_DIR / "europepmc_raw.json").write_text(json.dumps(raw_pages, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_log = {
        "database": "EuropePMC",
        "query": query,
        "retrieved_count": len(records),
    }
    return records, meta_log


def audit_moondream() -> dict:
    info = {"repo_id": "moondream/glaucoma-detection", "checked_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        api_data = http_get_json("https://huggingface.co/api/datasets/moondream/glaucoma-detection")
        info["exists"] = True
        info["api_metadata"] = {
            "id": api_data.get("id"),
            "author": api_data.get("author"),
            "createdAt": api_data.get("createdAt"),
            "lastModified": api_data.get("lastModified"),
            "downloads": api_data.get("downloads"),
            "likes": api_data.get("likes"),
            "tags": api_data.get("tags"),
            "cardData": api_data.get("cardData"),
            "description": api_data.get("description"),
            "citation": api_data.get("citation"),
            "siblings": [s.get("rfilename") for s in api_data.get("siblings", [])],
        }
    except Exception as e:
        info["exists"] = False
        info["api_error"] = str(e)
        # Buscar datasets de glaucoma en Hugging Face por si el ID exacto difiere
        try:
            search_hf = http_get_json("https://huggingface.co/api/datasets?search=glaucoma&limit=25")
            info["hf_glaucoma_datasets_found"] = [
                {
                    "id": d.get("id"),
                    "createdAt": d.get("createdAt"),
                    "lastModified": d.get("lastModified"),
                    "downloads": d.get("downloads"),
                    "description": (d.get("description") or "")[:300],
                }
                for d in search_hf
            ]
        except Exception as e2:
            info["hf_search_error"] = str(e2)

    if info.get("exists"):
        try:
            readme = http_get_text("https://huggingface.co/datasets/moondream/glaucoma-detection/raw/main/README.md")
            info["readme_md"] = readme
        except Exception as e:
            info["readme_error"] = str(e)

    (OUT_DIR / "auditoria_moondream_hf.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def deduplicate(all_records: list[dict]) -> tuple[list[dict], dict]:
    seen_doi: dict[str, dict] = {}
    seen_title: dict[str, dict] = {}
    unique_records: list[dict] = []
    dup_by_doi = 0
    dup_by_title = 0

    for rec in all_records:
        d = rec["doi"]
        t = norm_title(rec["titulo"])
        existing = None
        if d and d in seen_doi:
            existing = seen_doi[d]
            dup_by_doi += 1
        elif t and len(t) > 20 and t in seen_title:
            existing = seen_title[t]
            dup_by_title += 1

        if existing is not None:
            # Enriquecer registro existente con fuentes adicionales y resumen/OA si faltaban
            srcs = existing["fuentes_bd"].split("; ")
            if rec["fuente_bd"] not in srcs:
                srcs.append(rec["fuente_bd"])
                existing["fuentes_bd"] = "; ".join(srcs)
            if not existing["pmid"] and rec["pmid"]:
                existing["pmid"] = rec["pmid"]
            if not existing["doi"] and rec["doi"]:
                existing["doi"] = rec["doi"]
                seen_doi[rec["doi"]] = existing
            if len(rec["resumen"]) > len(existing["resumen"]):
                existing["resumen"] = rec["resumen"]
            if rec["acceso_abierto"] and not existing["acceso_abierto"]:
                existing["acceso_abierto"] = True
                existing["oa_url"] = rec["oa_url"]
        else:
            merged = dict(rec)
            merged["fuentes_bd"] = rec["fuente_bd"]
            unique_records.append(merged)
            if d:
                seen_doi[d] = merged
            if t and len(t) > 20:
                seen_title[t] = merged

    for idx, u in enumerate(unique_records, start=1):
        u["id_prisma_bd"] = f"BD-{idx:04d}"

    stats = {
        "total_raw": len(all_records),
        "duplicates_removed_total": len(all_records) - len(unique_records),
        "duplicates_by_doi": dup_by_doi,
        "duplicates_by_title": dup_by_title,
        "unique_for_screening": len(unique_records),
    }
    return unique_records, stats


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    oa_recs, oa_log = search_openalex()
    pm_recs, pm_log = search_pubmed()
    ep_recs, ep_log = search_europepmc()

    all_raw = oa_recs + pm_recs + ep_recs
    raw_fields = [
        "fuente_bd", "id_fuente", "doi", "pmid", "anio", "titulo",
        "autores", "revista_o_congreso", "tipo_documento", "acceso_abierto", "oa_url", "resumen"
    ]
    write_csv(OUT_DIR / "registros_crudas_bases.csv", all_raw, raw_fields)

    dedup_recs, dedup_stats = deduplicate(all_raw)
    dedup_fields = [
        "id_prisma_bd", "fuentes_bd", "doi", "pmid", "anio", "titulo",
        "autores", "revista_o_congreso", "tipo_documento", "acceso_abierto", "oa_url", "resumen"
    ]
    write_csv(OUT_DIR / "registros_bases_deduplicados.csv", dedup_recs, dedup_fields)

    hf_info = audit_moondream()

    full_log = {
        "executed_at_utc": started,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cutoff_date": CUTOFF_DATE,
        "databases": [oa_log, pm_log, ep_log],
        "deduplication": dedup_stats,
        "moondream_exists_on_hf": hf_info.get("exists"),
    }
    (OUT_DIR / "log_busqueda_prisma.json").write_text(json.dumps(full_log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(full_log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
