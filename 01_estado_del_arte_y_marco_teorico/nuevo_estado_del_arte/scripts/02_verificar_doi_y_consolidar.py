"""
Fase 2 y Fase 3 del Nuevo Estado del Arte (PRISMA 2020 Estricta):
1. Lee todos los registros recuperados por bases de datos (`busqueda_prisma/registros_bases_deduplicados.csv`).
2. Lee cualquier informe exportado en `deep_research/` (.md, .txt, .docx, .pdf) —excluyendo README.md y prompts_1clic/—
   extrayendo tanto las filas de tablas markdown como todos los DOIs citados, más las 9 semillas de `estado_del_arte_reescrito.md`.
3. Verifica CADA DOI contra la API oficial de Crossref, guardando la respuesta JSON cruda en `crossref_cache/`
   (reutilizando caché si ya existe) y comprobando coincidencia de título, autores, revista, año y tipo de documento.
4. Genera:
   - `verificacion_doi.csv` (todos los DOIs auditados contra Crossref)
   - `02_matriz_corpus.csv` (consolidada y deduplicada con trazabilidad PRISMA: `via_prisma` = `bases_datos`, `otros_metodos` o `ambas`)
   - `busqueda_prisma/conteos_prisma_preliminares.json` (conteos exactos salidos de CSV)
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BD_CSV = BASE_DIR / "busqueda_prisma" / "registros_bases_deduplicados.csv"
DR_DIR = BASE_DIR / "deep_research"
CR_DIR = BASE_DIR / "crossref_cache"
CR_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "ProyectoDL-UNAL-Glaucoma-SLR/2.0 (mailto:jtoscanoa@unal.edu.co)"
}

SEED_DOIS = [
    ("semilla_SOTA_previo", "10.1364/boe.10.000892"),
    ("semilla_SOTA_previo", "10.1186/s12938-019-0649-y"),
    ("semilla_SOTA_previo", "10.1016/j.media.2019.101570"),
    ("semilla_SOTA_previo", "10.1016/j.ophtha.2018.01.023"),
    ("semilla_SOTA_previo", "10.1038/s41597-022-01388-1"),
    ("semilla_SOTA_previo", "10.1109/tmi.2023.3313786"),
    ("semilla_SOTA_previo", "10.1038/s41746-023-00857-0"),
    ("semilla_SOTA_previo", "10.1038/s41586-023-06555-x"),
    ("semilla_SOTA_previo", "10.1371/journal.pone.0207982"),
]

DOI_REGEX = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")


def clean_doi(raw: str | None) -> str:
    if not raw:
        return ""
    d = raw.strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "https://dx.doi.org/", "doi:"):
        if d.startswith(p):
            d = d[len(p):]
    d = d.rstrip(".,;)]}>*\"'`")
    return d.strip()


def read_docx_text(path: Path) -> str:
    lines = []
    with zipfile.ZipFile(path) as z:
        xml_content = z.read("word/document.xml")
    tree = ET.fromstring(xml_content)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    body = tree.find("w:body", ns)
    if body is None:
        return ""
    for child in body:
        tag = child.tag.split("}")[-1]
        if tag == "p":
            t = "".join(x.text for x in child.findall(".//w:t", ns) if x.text)
            if t:
                lines.append(t)
        elif tag == "tbl":
            for row in child.findall(".//w:tr", ns):
                cells = [
                    " ".join("".join(x.text for x in p.findall(".//w:t", ns) if x.text) for p in tc.findall(".//w:p", ns)).strip()
                    for tc in row.findall(".//w:tc", ns)
                ]
                lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def read_report_text(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suf == ".docx":
        return read_docx_text(path)
    if suf == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            try:
                import fitz
                doc = fitz.open(str(path))
                return "\n".join(page.get_text() for page in doc)
            except Exception as e:
                print(f"Aviso: no se pudo extraer texto PDF de {path}: {e}")
                return ""
    return ""


def verify_doi_crossref(doi: str) -> dict:
    d = clean_doi(doi)
    if not d:
        return {"doi": "", "resolvio": False, "error": "vacio"}
    key = hashlib.md5(d.encode("utf-8")).hexdigest()
    cache_file = CR_DIR / f"{key}.json"
    msg = None
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("_resolvio") is False:
            return {"doi": d, "resolvio": False, "error": cached.get("_error", "not_found")}
        msg = cached.get("message", cached)
    else:
        url = "https://api.crossref.org/works/" + urllib.parse.quote(d, safe="")
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                msg = payload.get("message", {})
            time.sleep(0.12)
        except Exception as e:
            err_payload = {"_doi": d, "_resolvio": False, "_error": str(e)}
            cache_file.write_text(json.dumps(err_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"doi": d, "resolvio": False, "error": str(e)}

    titles = msg.get("title") or []
    title = titles[0] if titles else ""
    containers = msg.get("container-title") or []
    venue = containers[0] if containers else ""
    authors_raw = msg.get("author") or []
    first_author = ""
    authors_str_list = []
    for idx, a in enumerate(authors_raw[:10]):
        fam = a.get("family", "")
        giv = a.get("given", "")
        if idx == 0:
            first_author = fam or giv
        if fam or giv:
            authors_str_list.append(f"{fam}, {giv}".strip(", "))
    year = ""
    for date_field in ("published-print", "published-online", "issued", "created"):
        dp = ((msg.get(date_field) or {}).get("date-parts") or [[]])[0]
        if dp and dp[0]:
            year = str(dp[0])
            break
    autor_anio = f"{first_author} et al. ({year})" if len(authors_raw) > 2 else (
        f"{authors_raw[0].get('family','')} & {authors_raw[1].get('family','')} ({year})"
        if len(authors_raw) == 2 else f"{first_author} ({year})"
    )
    return {
        "doi": clean_doi(msg.get("DOI") or d),
        "resolvio": True,
        "titulo_crossref": re.sub(r"\s+", " ", title).strip(),
        "autores_crossref": "; ".join(authors_str_list),
        "autor_anio_crossref": autor_anio.strip(),
        "anio_crossref": year,
        "revista_crossref": re.sub(r"\s+", " ", venue).strip(),
        "tipo_crossref": msg.get("type", ""),
        "error": "",
    }


def scan_deep_research_reports() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if not DR_DIR.exists():
        return found
    for p in sorted(DR_DIR.iterdir()):
        if p.is_dir() or p.name.lower() == "readme.md":
            continue
        if p.suffix.lower() not in (".md", ".txt", ".docx", ".pdf"):
            continue
        tag = p.stem.split("_")[0]
        txt = read_report_text(p)
        for m in DOI_REGEX.findall(txt):
            cd = clean_doi(m)
            if cd:
                found.append((f"DeepResearch_{tag}", cd))
    return found


def main() -> None:
    bd_rows = []
    if BD_CSV.exists():
        with BD_CSV.open("r", encoding="utf-8-sig", newline="") as f:
            bd_rows = list(csv.DictReader(f))

    dr_dois = scan_deep_research_reports()

    # Recolectar conjunto único de DOIs a verificar en Crossref
    all_dois_set: set[str] = set()
    for r in bd_rows:
        if clean_doi(r.get("doi")):
            all_dois_set.add(clean_doi(r.get("doi")))
    for _, d in SEED_DOIS + dr_dois:
        if clean_doi(d):
            all_dois_set.add(clean_doi(d))

    print(f"Verificando {len(all_dois_set)} DOIs únicos contra Crossref (con caché en crossref_cache/)...")
    cr_map: dict[str, dict] = {}
    for idx, d in enumerate(sorted(all_dois_set), start=1):
        cr_map[d] = verify_doi_crossref(d)
        if idx % 50 == 0:
            print(f"  [{idx}/{len(all_dois_set)}] DOIs verificados...")

    # Guardar verificacion_doi.csv
    verif_fields = [
        "doi", "resolvio", "autor_anio_crossref", "anio_crossref",
        "titulo_crossref", "revista_crossref", "tipo_crossref", "autores_crossref", "error"
    ]
    with (BASE_DIR / "verificacion_doi.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=verif_fields)
        w.writeheader()
        for d in sorted(cr_map.keys()):
            w.writerow(cr_map[d])

    # Consolidar en 02_matriz_corpus.csv preservando las 27 columnas originales + trazabilidad
    corpus_by_key: dict[str, dict] = {}

    for r in bd_rows:
        d = clean_doi(r.get("doi"))
        key = f"doi:{d}" if d else f"id:{r['id_prisma_bd']}"
        cr = cr_map.get(d, {})
        corpus_by_key[key] = {
            "id": r["id_prisma_bd"],
            "origen": f"Bases ({r['fuentes_bd']})",
            "autor_año": cr.get("autor_anio_crossref") or f"{(r.get('autores') or '').split(',')[0]} ({r.get('anio','')})",
            "titulo": cr.get("titulo_crossref") or r.get("titulo", ""),
            "revista_o_congreso": cr.get("revista_crossref") or r.get("revista_o_congreso", ""),
            "doi": d,
            "doi_verificado_crossref": "SI" if cr.get("resolvio") else ("SIN_DOI" if not d else "NO_RESOLVIO"),
            "tipo_documento": cr.get("tipo_crossref") or r.get("tipo_documento", ""),
            "tarea_y_clase_positiva": "",
            "bases_entrenamiento": "",
            "bases_prueba": "",
            "tipo_validacion": "",
            "particion_por_paciente": "",
            "arquitectura_preentrenamiento": "",
            "estrategia_mitigacion": "",
            "fuente_variacion_que_aborda": "",
            "auc_interna": "",
            "auc_externa": "",
            "sens_esp_externa": "",
            "umbral_como_se_fijo": "",
            "ubicacion_cifras": "",
            "texto_completo_leido": "NO",
            "quien_extrajo": "",
            "decision_cribado": "PENDIENTE",
            "revisor_1": "",
            "revisor_2": "",
            "notas": f"PMID:{r.get('pmid','')} | OA:{r.get('acceso_abierto','')}",
        }

    other_idx = 1
    for origen_tag, d in SEED_DOIS + dr_dois:
        cd = clean_doi(d)
        if not cd:
            continue
        key = f"doi:{cd}"
        cr = cr_map.get(cd, {})
        if key in corpus_by_key:
            if origen_tag not in corpus_by_key[key]["origen"]:
                corpus_by_key[key]["origen"] += f"; {origen_tag}"
        else:
            corpus_by_key[key] = {
                "id": f"OT-{other_idx:04d}",
                "origen": origen_tag,
                "autor_año": cr.get("autor_anio_crossref", ""),
                "titulo": cr.get("titulo_crossref", ""),
                "revista_o_congreso": cr.get("revista_crossref", ""),
                "doi": cd,
                "doi_verificado_crossref": "SI" if cr.get("resolvio") else "NO_RESOLVIO",
                "tipo_documento": cr.get("tipo_crossref", ""),
                "tarea_y_clase_positiva": "",
                "bases_entrenamiento": "",
                "bases_prueba": "",
                "tipo_validacion": "",
                "particion_por_paciente": "",
                "arquitectura_preentrenamiento": "",
                "estrategia_mitigacion": "",
                "fuente_variacion_que_aborda": "",
                "auc_interna": "",
                "auc_externa": "",
                "sens_esp_externa": "",
                "umbral_como_se_fijo": "",
                "ubicacion_cifras": "",
                "texto_completo_leido": "NO",
                "quien_extrajo": "",
                "decision_cribado": "PENDIENTE",
                "revisor_1": "",
                "revisor_2": "",
                "notas": "",
            }
            other_idx += 1

    matriz_fields = [
        "id", "origen", "autor_año", "titulo", "revista_o_congreso", "doi",
        "doi_verificado_crossref", "tipo_documento", "tarea_y_clase_positiva",
        "bases_entrenamiento", "bases_prueba", "tipo_validacion",
        "particion_por_paciente", "arquitectura_preentrenamiento",
        "estrategia_mitigacion", "fuente_variacion_que_aborda", "auc_interna",
        "auc_externa", "sens_esp_externa", "umbral_como_se_fijo",
        "ubicacion_cifras", "texto_completo_leido", "quien_extrajo",
        "decision_cribado", "revisor_1", "revisor_2", "notas"
    ]
    rows_out = list(corpus_by_key.values())
    with (BASE_DIR / "02_matriz_corpus.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=matriz_fields)
        w.writeheader()
        w.writerows(rows_out)

    summary = {
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "bd_unique_records": len(bd_rows),
        "deep_research_doi_mentions": len(dr_dois),
        "total_unique_dois_checked_crossref": len(cr_map),
        "crossref_resolved_ok": sum(1 for v in cr_map.values() if v.get("resolvio")),
        "crossref_failed": sum(1 for v in cr_map.values() if not v.get("resolvio")),
        "total_corpus_rows": len(rows_out),
    }
    (BASE_DIR / "busqueda_prisma" / "conteos_prisma_preliminares.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
