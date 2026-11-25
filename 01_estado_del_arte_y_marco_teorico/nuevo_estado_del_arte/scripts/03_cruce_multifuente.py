"""
Motor de cruce multifuente para el Nuevo Estado del Arte (PRISMA 2020 estricta):
Cruza tres fuentes independientes + semillas SOTA:
  1. Bases_API: `busqueda_prisma/registros_bases_deduplicados.csv`
  2. Agentes_Antigravity: `deep_research_agentes/A1*.md` .. `A6*.md`
  3. Gemini_DeepResearch: `deep_research/*.md|.txt|.docx|.pdf` (excluye README.md, prompts_1clic/ y extracted_txt/)
  4. Semillas_SOTA: 9 DOIs base de `estado_del_arte_reescrito.md`

Capacidades avanzadas para informes PDF de Gemini Deep Research:
- Extracción espacial por celdas (`pypdf` `visitor_text`) filtrando superíndices de cita
  (`fontSize < 8.5 pt`) y uniendo celdas partidas entre saltos de página consecutivos.
- Reconstrucción de filas tabulares de Gemini y líneas de texto para contexto (`±800` chars).
- Desambiguación determinística de dígitos finales residuales verificando autor/año/título en el contexto.
- Extracción y resolución cacheada de anotaciones `/URI (...)` (`PMC`, `PMID`, `arXiv`, `MDPI`,
  `medRxiv/bioRxiv`, `Nature`, `Zenodo` y DOIs embebidos en URLs editoriales).
- Separación entre estudios primarios/revisiones candidatos (`cribado_prisma_humano.csv`) y citas
  puramente metodológicas generales sin dominio oftálmico obtenidas solo vía URI (`referencias_metodologicas_gemini.csv`).
- Generación de muestra aleatoria estratificada ciega (`cribado_muestra_ciega_revisor2.csv`, 20 %, min 40)
  y archivo de auditoría `busqueda_prisma/log_extraccion_pdf_gemini.json`.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import random
import re
import time
import unicodedata
import urllib.parse
import urllib.request
import zipfile
import zlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BD_CSV = BASE_DIR / "busqueda_prisma" / "registros_bases_deduplicados.csv"
DR_GEMINI_DIR = BASE_DIR / "deep_research"
DR_AGENTES_DIR = BASE_DIR / "deep_research_agentes"
CR_DIR = BASE_DIR / "crossref_cache"
CR_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "busqueda_prisma").mkdir(parents=True, exist_ok=True)
URI_CACHE_FILE = CR_DIR / "uri_doi_cache.json"

HEADERS = {
    "User-Agent": "ProyectoDL-UNAL-Glaucoma-SLR/2.0 (mailto:jtoscanoa@unal.edu.co)"
}

SEED_DOIS = [
    ("Semilla_SOTA", "10.1364/boe.10.000892"),
    ("Semilla_SOTA", "10.1186/s12938-019-0649-y"),
    ("Semilla_SOTA", "10.1016/j.media.2019.101570"),
    ("Semilla_SOTA", "10.1016/j.ophtha.2018.01.023"),
    ("Semilla_SOTA", "10.1038/s41597-022-01388-1"),
    ("Semilla_SOTA", "10.1109/tmi.2023.3313786"),
    ("Semilla_SOTA", "10.1038/s41746-023-00857-0"),
    ("Semilla_SOTA", "10.1038/s41586-023-06555-x"),
    ("Semilla_SOTA", "10.1371/journal.pone.0207982"),
]

DOI_REGEX = re.compile(r"\b(10\.\d{4,9}/(?!\.\.\.)[-._;()/:A-Za-z0-9]+)")

OPHTHALMIC_KEYWORDS = (
    "glaucom", "retin", "fundus", "optic", "papill", "ophthalm", "eye", "ocular",
    "disc", "cup", "macul", "refuge", "airogs", "chaksu", "acrima", "rim-one",
    "orlando", "drishti", "oct", "perimetr", "visual field", "retinograph",
    "eyepacs", "retfound", "justraigs", "ohts", "uk biobank", "brset", "grape",
    "papila", "g1020", "gon", "optic nerve", "vessel", "diabetic", "smdg",
)


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text or "")
        if not unicodedata.combining(ch)
    ).lower()


def balance_parens_cut(d: str) -> str:
    """Recorta texto pegado después de paréntesis no balanceados."""
    depth = 0
    cut_idx = len(d)
    for i, ch in enumerate(d):
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                cut_idx = i
                break
            depth -= 1
    d = d[:cut_idx]
    while "(" in d and d.count("(") > d.count(")"):
        d = d[: d.rfind("(")]
    return d


def clean_doi(raw: str | None) -> str:
    if not raw:
        return ""
    d = raw.strip().lower()
    for p in (
        "https://doi.org/",
        "http://doi.org/",
        "http://dx.doi.org/",
        "https://dx.doi.org/",
        "doi:",
    ):
        if d.startswith(p):
            d = d[len(p):]
    d = re.sub(r"[`*\"'<>]+", "", d)
    d = balance_parens_cut(d)
    d = d.rstrip(".,;)]}/:")
    # Recortar sufijos de ruta web que a veces quedan pegados al extraer URLs editoriales / PDF
    d = re.sub(r"(?:\.(?:full(?:-text|\.pdf)?|abstract|short|pdf|html|xml)|/(?:full|pdf|abstract|meta|html))$", "", d)
    if re.match(r"^10\.(?:1101|64898)/\d{4}\.\d{2}\.\d{2}\.\d+v\d+$", d):
        d = re.sub(r"v\d+$", "", d)
    d = d.strip()
    if "/" not in d:
        return ""
    prefix, suffix = d.split("/", 1)
    if (
        not prefix.startswith("10.")
        or len(suffix) < 3
        or suffix in ("arxiv", "zenodo", "figshare", "dryad")
    ):
        return ""
    return d


def norm_title(title: str | None) -> str:
    if not title:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def http_get_json(url: str, timeout: int = 25, retries: int = 2) -> dict | None:
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))
    return None


def load_uri_cache() -> dict:
    if URI_CACHE_FILE.exists():
        try:
            return json.loads(URI_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_uri_cache(cache: dict) -> None:
    URI_CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


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
            rows = child.findall(".//w:tr", ns)
            for r_idx, row in enumerate(rows):
                cells = [
                    " ".join(
                        "".join(x.text for x in p.findall(".//w:t", ns) if x.text)
                        for p in tc.findall(".//w:p", ns)
                    ).strip()
                    for tc in row.findall(".//w:tc", ns)
                ]
                lines.append("| " + " | ".join(cells) + " |")
                if r_idx == 0 and cells:
                    lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(lines)


def extract_pdf_uris(path: Path) -> list[str]:
    """Extrae anotaciones /URI (...) del PDF (crudo y streams comprimidos)."""
    data = path.read_bytes()
    raw_uris = list(re.findall(rb"/URI\s*\(([^)]+)\)", data))
    for s in re.findall(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        try:
            dec = zlib.decompress(s)
            raw_uris.extend(re.findall(rb"/URI\s*\(([^)]+)\)", dec))
        except Exception:
            pass
    seen: set[str] = set()
    clean_uris: list[str] = []
    for u_bytes in raw_uris:
        u = u_bytes.decode("latin1", "replace").replace(r"\(", "(").replace(r"\)", ")").strip()
        if u and "docs.google.com/unknown_url" not in u and u not in seen:
            seen.add(u)
            clean_uris.append(u)
    return clean_uris


def _join_cell_lines(lines: list[str]) -> str:
    """Une líneas cortas dentro de una misma celda reconstruyendo DOIs sin fusionar palabras nuevas."""
    toks: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if toks and (" " not in s) and (
            " " not in toks[-1] or re.search(r"10\.[0-9A-Za-z._/;()-]*$", toks[-1])
        ):
            # Evita pegar un apellido o sección con mayúscula después de un DOI ya completo
            if (
                re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]{5,}$", toks[-1])
                and re.match(r"^[A-Z][a-z]{2,}", s)
                and not re.match(r"^(OPTH|JAMA|ISBI|IEMBS|CBMS|TMI|BMJ|JCMA|BJO)", s)
            ):
                toks.append(s)
            else:
                toks[-1] += s
        else:
            toks.append(s)
    return " ".join(toks)


def extract_pdf_structured(path: Path) -> dict:
    """
    Extrae celdas espaciales (sin superíndices), reconstruye filas de tablas,
    genera texto plano limpio para lectura humana y recolecta DOIs candidatos con contexto.
    """
    import pypdf

    reader = pypdf.PdfReader(str(path))
    cells_by_cm4: dict[float, list[dict]] = defaultdict(list)
    page_plain_texts: list[str] = []
    page_cells_all: dict[int, list[dict]] = defaultdict(list)

    for pg_i, page in enumerate(reader.pages):
        tokens: list[tuple[float, float, float, str]] = []

        def visitor(text, cm, tm, fd, fs):
            if not text or not text.strip():
                return
            eff_fs = fs * math.hypot(cm[0], cm[1]) * math.hypot(tm[0], tm[1])
            # Filtrar superíndices de citas (~7.2 pt vs 11.0 pt cuerpo)
            if eff_fs < 8.5:
                return
            ex = round(cm[0] * tm[4] + cm[2] * tm[5] + cm[4], 1)
            ey = round(cm[1] * tm[4] + cm[3] * tm[5] + cm[5], 1)
            cm4 = round(cm[4], 1)
            tokens.append((cm4, ey, ex, text))

        page.extract_text(visitor_text=visitor)

        # Texto plano por líneas visuales completas para párrafos / bibliografía
        # Agrupamos primero por celda (cm4) para no mezclar columnas en tablas
        by_cm4: dict[float, list[tuple[float, float, str]]] = defaultdict(list)
        for cm4, ey, ex, t in tokens:
            by_cm4[cm4].append((ey, ex, t))

        page_lines_visual: list[tuple[float, float, str]] = []
        for cm4, lst in by_cm4.items():
            lst.sort(key=lambda z: (-z[0], z[1]))
            lines: list[tuple[float, str]] = []
            cur_y: float | None = None
            cur_toks: list[tuple[float, str]] = []
            for ey, ex, t in lst:
                if cur_y is None or abs(ey - cur_y) <= 3.0:
                    cur_toks.append((ex, t))
                    if cur_y is None:
                        cur_y = ey
                else:
                    cur_toks.sort(key=lambda z: z[0])
                    lines.append((cur_y, " ".join(x[1] for x in cur_toks)))
                    cur_toks = [(ex, t)]
                    cur_y = ey
            if cur_toks:
                cur_toks.sort(key=lambda z: z[0])
                lines.append((cur_y, " ".join(x[1] for x in cur_toks)))

            # Partir en celdas si salto vertical > 20.5 pt
            cur_cell = None
            prev_y = None
            for y, txt in lines:
                if cur_cell is None or (prev_y - y) > 20.5:
                    cur_cell = {
                        "pg": pg_i,
                        "cm4": cm4,
                        "y_top": y,
                        "y_bot": y,
                        "lines": [txt],
                    }
                    cells_by_cm4[cm4].append(cur_cell)
                    page_cells_all[pg_i].append(cur_cell)
                else:
                    cur_cell["y_bot"] = y
                    cur_cell["lines"].append(txt)
                prev_y = y

        # Generar lectura legible de página para extracted_txt/ y contexto
        pg_blocks = sorted(page_cells_all[pg_i], key=lambda c: (-c["y_top"], c["cm4"]))
        pg_strs = [f"[{c['cm4']:.0f}] {_join_cell_lines(c['lines'])}" for c in pg_blocks]
        page_plain_texts.append(f"=== PAGINA {pg_i + 1} ===\n" + "\n".join(pg_strs))

    # Fusionar celdas divididas entre fin de página p y tope de página p+1 en el mismo cm4
    merged_cells: list[dict] = []
    for cm4 in sorted(cells_by_cm4.keys()):
        clist = cells_by_cm4[cm4]
        col_merged: list[dict] = []
        for c in clist:
            prev_last = col_merged[-1]["lines"][-1].strip() if col_merged else ""
            prev_ends_doi = bool(
                col_merged
                and re.search(r"10\.\d{0,9}[-._;()/:A-Za-z0-9]*[-/_.]$", prev_last)
            )
            page_split_std = (
                col_merged
                and c["pg"] == col_merged[-1]["pg"] + 1
                and col_merged[-1]["y_bot"] < 135
                and c["y_top"] > 650
            )
            page_split_doi = (
                col_merged
                and c["pg"] == col_merged[-1]["pg"] + 1
                and prev_ends_doi
                and c["y_top"] > 500
            )
            if page_split_std or page_split_doi:
                col_merged[-1]["lines"].extend(c["lines"])
                col_merged[-1]["pg_end"] = c["pg"]
                col_merged[-1]["y_bot"] = c["y_bot"]
            else:
                copy_c = dict(c)
                copy_c["pg_start"] = c["pg"]
                copy_c["pg_end"] = c["pg"]
                col_merged.append(copy_c)
        for mc in col_merged:
            mc["text"] = _join_cell_lines(mc["lines"])
        merged_cells.extend(col_merged)

    # También construir representación de texto continuo en orden de página y fila
    full_text = "\n\n".join(page_plain_texts)

    # Extraer DOIs candidatos de cada celda reconstruida con ventana de contexto de fila/página
    doi_candidates: list[dict] = []
    seen_cand_pairs: set[tuple[str, int]] = set()

    # Mapa rápido por (pg, banda y_top) para capturar otras columnas de la misma fila como contexto y pista
    table_hints_rows: list[dict] = []
    cells_by_pg_row: dict[int, list[dict]] = defaultdict(list)
    for mc in merged_cells:
        cells_by_pg_row[mc["pg_start"]].append(mc)

    for mc in merged_cells:
        txt = mc["text"]
        for m in DOI_REGEX.finditer(txt):
            cd = clean_doi(m.group(1))
            if not cd:
                continue
            # Contexto: celdas vecinas de la misma página (|y_top - mc.y_top| <= 80) + páginas start/end
            neighbors = [
                nc["text"]
                for nc in cells_by_pg_row.get(mc["pg_start"], [])
                if abs(nc["y_top"] - mc["y_top"]) <= 95 or nc is mc
            ]
            if mc["pg_end"] != mc["pg_start"]:
                neighbors.extend(
                    nc["text"]
                    for nc in cells_by_pg_row.get(mc["pg_end"], [])
                    if nc["y_top"] > 620
                )
            ctx_blob = " | ".join(neighbors)
            # Buscar también en texto completo alrededor del DOI o de sus primeros caracteres
            if len(ctx_blob) < 400 and cd[:12] in full_text:
                pos = full_text.find(cd[:12])
                ctx_blob = full_text[max(0, pos - 800): min(len(full_text), pos + 800)]
            key = (cd, mc["pg_start"])
            if key not in seen_cand_pairs:
                seen_cand_pairs.add(key)
                doi_candidates.append({
                    "doi_raw": m.group(1),
                    "doi_clean": cd,
                    "page": mc["pg_start"] + 1,
                    "context": ctx_blob[:2000],
                })
            # Registrar fila tabular de contexto para cruce_comparativo_informes.csv
            row_cells = sorted(
                [nc for nc in cells_by_pg_row.get(mc["pg_start"], []) if abs(nc["y_top"] - mc["y_top"]) <= 22],
                key=lambda z: z["cm4"],
            )
            if len(row_cells) >= 3:
                row_dict = {f"Col_x{int(rc['cm4'])}": rc["text"] for rc in row_cells}
                table_hints_rows.append({
                    "dois": [cd],
                    "campos": row_dict,
                    "page": mc["pg_start"] + 1,
                })

    # Respaldo adicional: reconstrucción sobre texto corrido de líneas cortas (por si algo estuvo fuera de tabla)
    raw_txt = "\n".join((p.extract_text() or "") for p in reader.pages)
    raw_lines = raw_txt.splitlines()
    rebuilt_parts: list[str] = []
    for ln in raw_lines:
        s = ln.strip()
        if not s:
            rebuilt_parts.append(" ")
            continue
        if (
            rebuilt_parts
            and len(s) <= 28
            and " " not in s
            and re.fullmatch(r"[-._;()/:A-Za-z0-9]+", s)
            and re.search(r"10\.[0-9A-Za-z._/;()-]*$", rebuilt_parts[-1])
        ):
            rebuilt_parts[-1] += s
        else:
            rebuilt_parts.append(" " + s)
    rebuilt_blob = "".join(rebuilt_parts)
    for m in DOI_REGEX.finditer(rebuilt_blob):
        cd = clean_doi(m.group(1))
        if cd and not any(c["doi_clean"] == cd for c in doi_candidates):
            pos = m.start()
            ctx = rebuilt_blob[max(0, pos - 800): min(len(rebuilt_blob), pos + 800)]
            doi_candidates.append({
                "doi_raw": m.group(1),
                "doi_clean": cd,
                "page": 0,
                "context": ctx,
            })

    return {
        "full_text": full_text,
        "doi_candidates": doi_candidates,
        "table_rows": table_hints_rows,
        "uris": extract_pdf_uris(path),
        "n_pages": len(reader.pages),
    }


def context_matches_metadata(verif: dict, context_text: str, pdf_uris_blob: str = "") -> bool:
    """Comprueba que autor, año o palabra distintiva del título coincida con la ventana del PDF."""
    if not verif.get("resolvio"):
        return False
    ctx = strip_accents(context_text + " " + pdf_uris_blob)
    year = (verif.get("anio_crossref") or "").strip()
    if len(year) == 4 and year in ctx:
        return True
    autores = strip_accents(verif.get("autores_crossref") or "")
    first_fam = autores.split(",")[0].split(";")[0].strip()
    if len(first_fam) >= 3 and re.search(r"\b" + re.escape(first_fam) + r"\b", ctx):
        return True
    titulo = strip_accents(verif.get("titulo_crossref") or "")
    for w in re.findall(r"[a-z0-9]{6,}", titulo):
        if w not in {"glaucoma", "detection", "learning", "fundus", "images", "retinal", "classification"}:
            if w in ctx:
                return True
    return False


def query_crossref(d: str) -> tuple[bool, dict, str]:
    key = hashlib.md5(d.encode("utf-8")).hexdigest()
    cache_file = CR_DIR / f"{key}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("_resolvio") is False:
            return False, {}, cached.get("_error", "not_found")
        return True, cached.get("message", cached), ""

    url = "https://api.crossref.org/works/" + urllib.parse.quote(d, safe="")
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(0.08)
            return True, payload.get("message", {}), ""
    except Exception as e:
        err_payload = {"_doi": d, "_resolvio": False, "_error": str(e)}
        cache_file.write_text(json.dumps(err_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return False, {}, str(e)


def query_datacite(d: str) -> tuple[bool, dict, str]:
    key = hashlib.md5(d.encode("utf-8")).hexdigest()
    cache_file = CR_DIR / f"datacite_{key}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("_resolvio") is False:
            return False, {}, cached.get("_error", "not_found")
        return True, (cached.get("data") or {}).get("attributes", {}), ""

    url = "https://api.datacite.org/dois/" + urllib.parse.quote(d, safe="")
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(0.08)
            return True, (payload.get("data") or {}).get("attributes", {}), ""
    except Exception as e:
        err_payload = {"_doi": d, "_resolvio": False, "_error": str(e)}
        cache_file.write_text(json.dumps(err_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return False, {}, str(e)


def clean_jats_abstract(raw_abs: str | None) -> str:
    if not raw_abs:
        return ""
    t = re.sub(r"<[^>]+>", " ", raw_abs)
    return re.sub(r"\s+", " ", t).strip()


def verify_doi_deterministic(doi: str) -> dict:
    d = clean_doi(doi)
    if not d:
        return {"doi": "", "resolvio": False, "registrador": "SIN_DOI", "error": "vacio"}

    ok_cr, msg, err_cr = query_crossref(d)
    if ok_cr:
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
        if len(authors_raw) > 2:
            autor_anio = f"{first_author} et al. ({year})"
        elif len(authors_raw) == 2:
            autor_anio = f"{authors_raw[0].get('family','')} & {authors_raw[1].get('family','')} ({year})"
        else:
            autor_anio = f"{first_author} ({year})"
        return {
            "doi": clean_doi(msg.get("DOI") or d),
            "resolvio": True,
            "registrador": "Crossref",
            "autor_anio_crossref": autor_anio.strip(),
            "anio_crossref": year,
            "titulo_crossref": re.sub(r"\s+", " ", title).strip(),
            "revista_crossref": re.sub(r"\s+", " ", venue).strip(),
            "tipo_crossref": msg.get("type", ""),
            "autores_crossref": "; ".join(authors_str_list),
            "abstract_crossref": clean_jats_abstract(msg.get("abstract")),
            "error": "",
        }

    ok_dc, attrs, err_dc = query_datacite(d)
    if ok_dc:
        titles = attrs.get("titles") or []
        title = titles[0].get("title", "") if titles and isinstance(titles[0], dict) else ""
        creators = attrs.get("creators") or []
        authors_str_list = []
        first_author = ""
        for idx, c in enumerate(creators[:10]):
            name = c.get("familyName") or c.get("name") or ""
            giv = c.get("givenName") or ""
            if idx == 0:
                first_author = name
            if name or giv:
                authors_str_list.append(f"{name}, {giv}".strip(", "))
        year = str(attrs.get("publicationYear") or "")
        if len(creators) > 2:
            autor_anio = f"{first_author} et al. ({year})"
        elif len(creators) == 2:
            n0 = creators[0].get("familyName") or creators[0].get("name") or ""
            n1 = creators[1].get("familyName") or creators[1].get("name") or ""
            autor_anio = f"{n0} & {n1} ({year})"
        else:
            autor_anio = f"{first_author} ({year})"
        pub = attrs.get("publisher") or ""
        rtype = (attrs.get("types") or {}).get("resourceTypeGeneral") or "preprint_or_dataset"
        descs = attrs.get("descriptions") or []
        dc_abs = ""
        if descs and isinstance(descs[0], dict):
            dc_abs = clean_jats_abstract(descs[0].get("description", ""))
        return {
            "doi": clean_doi(attrs.get("doi") or d),
            "resolvio": True,
            "registrador": "DataCite",
            "autor_anio_crossref": autor_anio.strip(),
            "anio_crossref": year,
            "titulo_crossref": re.sub(r"\s+", " ", title).strip(),
            "revista_crossref": pub,
            "tipo_crossref": f"DataCite:{rtype}",
            "autores_crossref": "; ".join(authors_str_list),
            "abstract_crossref": dc_abs,
            "error": "",
        }

    return {
        "doi": d,
        "resolvio": False,
        "registrador": "NO_RESOLVIO",
        "autor_anio_crossref": "",
        "anio_crossref": "",
        "titulo_crossref": "",
        "revista_crossref": "",
        "tipo_crossref": "",
        "autores_crossref": "",
        "abstract_crossref": "",
        "error": f"Crossref: {err_cr} | DataCite: {err_dc}",
    }


def resolve_uri_batch(uris: list[str]) -> dict[str, dict]:
    """
    Resuelve URIs (PMC, PubMed, medRxiv, MDPI, arXiv, Zenodo, Nature, editoriales con DOI)
    a DOIs canónicos usando caché en disco y APIs oficiales.
    """
    cache = load_uri_cache()
    missing_pmc: list[str] = []
    missing_pmid: list[str] = []
    pending_uris: list[str] = []

    for u in uris:
        if u in cache and cache[u].get("doi"):
            continue
        if u in cache and cache[u].get("checked") and cache[u].get("doi") == "":
            continue
        pending_uris.append(u)
        m_pmc = re.search(r"/(PMC\d+)", u, re.I)
        if m_pmc and ("ncbi.nlm.nih.gov" in u or "europepmc.org" in u):
            pmcid = m_pmc.group(1).upper()
            if pmcid not in missing_pmc:
                missing_pmc.append(pmcid)
        m_pm = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", u, re.I)
        if m_pm:
            pmid = m_pm.group(1)
            if pmid not in missing_pmid:
                missing_pmid.append(pmid)

    # 1. Lotes NCBI idconv para PMCIDs y PMIDs + esummary respaldo
    id_to_doi: dict[str, str] = {}
    for i in range(0, len(missing_pmc), 50):
        batch = missing_pmc[i: i + 50]
        url = (
            "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids="
            + ",".join(batch)
            + "&idtype=pmcid&format=json&tool=unal_slr&email=jtoscanoa@unal.edu.co"
        )
        res = http_get_json(url, timeout=25)
        if res:
            for rec in res.get("records", []):
                if rec.get("pmcid") and rec.get("doi"):
                    id_to_doi[rec["pmcid"].upper()] = clean_doi(rec["doi"])
        time.sleep(0.15)

    for i in range(0, len(missing_pmid), 50):
        batch = missing_pmid[i: i + 50]
        url = (
            "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids="
            + ",".join(batch)
            + "&idtype=pmid&format=json&tool=unal_slr&email=jtoscanoa@unal.edu.co"
        )
        res = http_get_json(url, timeout=25)
        unresolved_pmids = set(batch)
        if res:
            for rec in res.get("records", []):
                pm = str(rec.get("pmid") or "")
                if pm and rec.get("doi"):
                    id_to_doi[f"PMID:{pm}"] = clean_doi(rec["doi"])
                    unresolved_pmids.discard(pm)
        # Fallback NCBI esummary para PMIDs que no están en PMC
        if unresolved_pmids:
            es_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id="
                + ",".join(sorted(unresolved_pmids))
                + "&retmode=json"
            )
            es = http_get_json(es_url, timeout=25)
            if es:
                rs = es.get("result", {})
                for pm in unresolved_pmids:
                    item = rs.get(pm) or {}
                    for aid in item.get("articleids", []):
                        if aid.get("idtype") == "doi" and aid.get("value"):
                            id_to_doi[f"PMID:{pm}"] = clean_doi(aid["value"])
                            break
        time.sleep(0.20)

    # Fallback Europe PMC para PMCIDs que no resolvieron en idconv
    for pmcid in missing_pmc:
        if pmcid not in id_to_doi:
            ep_url = (
                f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
                f"query=PMCID:{pmcid}&format=json&resultType=lite"
            )
            ep = http_get_json(ep_url, timeout=20, retries=1)
            if ep:
                for rec in (ep.get("resultList") or {}).get("result", []):
                    if rec.get("doi"):
                        id_to_doi[pmcid] = clean_doi(rec["doi"])
                        break
            time.sleep(0.1)

    # 2. Resolver cada URI pendiente individualmente
    for u in pending_uris:
        resolved_doi = ""
        method = "unresolved"
        m_pmc = re.search(r"/(PMC\d+)", u, re.I)
        m_pm = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", u, re.I)
        m_arxiv = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", u, re.I)
        m_mdpi = re.search(r"mdpi\.com/(\d{4}-\d{3}[\dXx])/(\d+)/(\d+)/(\d+)", u, re.I)
        m_zen = re.search(r"zenodo\.org/records?/(\d+)", u, re.I)
        m_nat = re.search(r"nature\.com/articles/([a-z0-9\-]+)", u, re.I)
        m_rsq = re.search(r"researchsquare\.com/article/(rs-\d+)", u, re.I)

        if m_pmc and ("ncbi.nlm.nih.gov" in u or "europepmc.org" in u):
            pmcid = m_pmc.group(1).upper()
            resolved_doi = id_to_doi.get(pmcid, "")
            method = f"NCBI_PMCID:{pmcid}" if resolved_doi else "PMCID_no_doi"
        elif m_pm:
            pmid = m_pm.group(1)
            resolved_doi = id_to_doi.get(f"PMID:{pmid}", "")
            method = f"NCBI_PMID:{pmid}" if resolved_doi else "PMID_no_doi"
        elif "doi.org/" in u.lower() or "10." in u:
            # Extraer DOI embebido (doi.org, medrxiv, plos, frontiers, spie, bmc, springer, etc.)
            u_clean = re.sub(r"\?.*$", "", u)
            m_doi = re.search(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", u_clean)
            if m_doi:
                cand = clean_doi(m_doi.group(1))
                # Quitar /full o /abstract o /meta al final si aparece
                cand = re.sub(r"/(?:full|meta|pdf|html)$", "", cand)
                if cand:
                    resolved_doi = cand
                    method = "URL_embedded_DOI"
        elif m_mdpi:
            issn, vol, iss, art = m_mdpi.groups()
            cr_url = (
                f"https://api.crossref.org/works?filter=issn:{urllib.parse.quote(issn)}"
                f"&rows=5&query.bibliographic={vol}+{iss}+{art}"
            )
            cr_res = http_get_json(cr_url, timeout=20)
            if cr_res:
                items = (cr_res.get("message") or {}).get("items", [])
                for it in items:
                    if str(it.get("volume", "")) == vol and str(it.get("issue", "")) == iss:
                        it_doi = clean_doi(it.get("DOI"))
                        it_art = str(it.get("article-number") or "")
                        if (
                            it_art == art
                            or it_doi.endswith(f"{vol}{iss.zfill(2)}{art.zfill(4)}")
                            or it_doi.endswith(f"{vol}{iss.zfill(2)}{art}")
                            or (it.get("URL") or "").rstrip("/").endswith(f"/{vol}/{iss}/{art}")
                        ):
                            resolved_doi = it_doi
                            method = "Crossref_MDPI_ISSN"
                            break
            time.sleep(0.1)
        elif m_arxiv:
            aid = m_arxiv.group(1)
            oa_url = f"https://api.openalex.org/works/https://doi.org/10.48550/arxiv.{aid}"
            oa = http_get_json(oa_url, timeout=20)
            pub_doi = ""
            if oa:
                cand_doi = clean_doi(oa.get("doi"))
                pl_type = (((oa.get("primary_location") or {}).get("source") or {}).get("type") or "")
                # Usar versión publicada si es artículo o congreso real y no el propio 10.48550/arxiv ni 10.21227 dataport
                if cand_doi and not cand_doi.startswith("10.48550/") and not cand_doi.startswith("10.21227/"):
                    if oa.get("type") in ("article", "proceedings-article", "book-chapter") or pl_type in ("journal", "conference"):
                        pub_doi = cand_doi
            resolved_doi = pub_doi or f"10.48550/arxiv.{aid}"
            method = "OpenAlex_arXiv_published" if pub_doi else "arXiv_DataCite"
            time.sleep(0.1)
        elif m_zen:
            resolved_doi = f"10.5281/zenodo.{m_zen.group(1)}"
            method = "Zenodo_URI"
        elif m_nat:
            resolved_doi = f"10.1038/{m_nat.group(1).lower()}"
            method = "Nature_URI"
        elif m_rsq:
            resolved_doi = f"10.21203/rs.3.{m_rsq.group(1).lower()}/v1"
            method = "ResearchSquare_URI"

        cache[u] = {
            "doi": clean_doi(resolved_doi),
            "method": method,
            "checked": True,
        }

    save_uri_cache(cache)
    return {u: cache.get(u, {"doi": "", "method": "none"}) for u in uris}


def is_ophthalmic_study(verif: dict) -> bool:
    blob = strip_accents((verif.get("titulo_crossref") or "") + " " + (verif.get("revista_crossref") or ""))
    return any(kw in blob for kw in OPHTHALMIC_KEYWORDS)


def parse_markdown_tables(text: str, source_label: str) -> list[dict]:
    """Extrae filas de tablas Markdown conservando encabezados originales y DOIs."""
    lines = text.splitlines()
    parsed_rows = []
    i = 0
    while i < len(lines) - 1:
        line = lines[i].strip()
        next_line = lines[i + 1].strip()
        if line.startswith("|") and "|" in line[1:] and re.match(r"^\|[\s:\-|]+\|$", next_line):
            headers = [h.strip() for h in line.strip("|").split("|")]
            i += 2
            while i < len(lines):
                row_line = lines[i].strip()
                if not row_line.startswith("|"):
                    break
                cells = [c.strip() for c in row_line.strip("|").split("|")]
                if len(cells) < len(headers):
                    cells.extend([""] * (len(headers) - len(cells)))
                row_dict = {headers[idx]: cells[idx] for idx in range(len(headers))}
                row_dois = []
                for v in row_dict.values():
                    for m in DOI_REGEX.findall(v):
                        cd = clean_doi(m)
                        if cd and cd not in row_dois:
                            row_dois.append(cd)
                parsed_rows.append({
                    "fuente_informe": source_label,
                    "dois": row_dois,
                    "campos": row_dict,
                })
                i += 1
        else:
            i += 1
    return parsed_rows


def map_table_row_to_fields(row_campos: dict, prefix_tag: str) -> dict:
    """Mapea columnas heterogéneas de tablas R1-R6 / A1-A6 a campos estándar."""
    mapped: dict[str, str] = {}
    for k, v in row_campos.items():
        if not v or v.strip() in ("-", "—", "N/A", "no reportado", "No reportado"):
            continue
        kl = k.lower()
        val = f"[{prefix_tag}: verificar en texto completo] {v.strip()}"
        if "tarea" in kl or "clase positiva" in kl or "definición de etiqueta" in kl or "esquema" in kl:
            mapped.setdefault("tarea_y_clase_positiva", val)
        elif "bases de entrenamiento" in kl or "entrenamiento" in kl:
            mapped.setdefault("bases_entrenamiento", val)
        elif "bases de prueba" in kl or "no vistas" in kl:
            mapped.setdefault("bases_prueba", val)
        elif "tipo de validación" in kl or "validación" in kl or "diseño" in kl:
            mapped.setdefault("tipo_validacion", val)
        elif "partición por paciente" in kl:
            mapped.setdefault("particion_por_paciente", val)
        elif "arquitectura" in kl or "preentrenamiento" in kl:
            mapped.setdefault("arquitectura_preentrenamiento", val)
        elif "estrategia" in kl or "método de corrección" in kl:
            mapped.setdefault("estrategia_mitigacion", val)
        elif "variación que ataca" in kl or "factor estudiado" in kl or "dominio clínico" in kl:
            mapped.setdefault("fuente_variacion_que_aborda", val)
        elif "auc interna" in kl:
            mapped.setdefault("auc_interna", val)
        elif "auc externa" in kl or "mejora reportada" in kl or "magnitud del efecto" in kl or "severidad" in kl:
            mapped.setdefault("auc_externa", val)
        elif "sensibilidad" in kl or "especificidad" in kl or "qué pasó" in kl:
            mapped.setdefault("sens_esp_externa", val)
        elif "umbral" in kl or "fijó" in kl:
            mapped.setdefault("umbral_como_se_fijo", val)
        elif "ubicación" in kl:
            mapped.setdefault("ubicacion_cifras", val)
    return mapped


def scan_folder_reports(folder: Path, prefix_group: str) -> dict:
    dois_found: list[tuple[str, str]] = []
    table_rows: list[dict] = []
    files_read: list[str] = []
    pdf_audit: dict[str, dict] = {}
    uri_only_records: dict[str, dict] = {}  # doi -> {'verif': ..., 'informes': set(), 'urls': set()}

    if not folder.exists():
        return {
            "dois_found": dois_found,
            "table_rows": table_rows,
            "files_read": files_read,
            "pdf_audit": pdf_audit,
            "uri_only_records": uri_only_records,
        }

    extracted_txt_dir = folder / "extracted_txt"

    for p in sorted(folder.iterdir()):
        # Ignora directorios (como prompts_1clic/ y extracted_txt/) para no contar doble
        if p.is_dir() or p.name.lower() == "readme.md":
            continue
        suf = p.suffix.lower()
        if suf not in (".md", ".txt", ".docx", ".pdf"):
            continue

        tag = p.stem.split("_")[0]
        label = f"{prefix_group}_{tag}"
        files_read.append(p.name)

        if suf in (".md", ".txt", ".docx"):
            txt = p.read_text(encoding="utf-8", errors="replace") if suf != ".docx" else read_docx_text(p)
            if not txt.strip():
                continue
            for m in DOI_REGEX.findall(txt):
                cd = clean_doi(m)
                if cd:
                    dois_found.append((label, cd))
            table_rows.extend(parse_markdown_tables(txt, label))
            continue

        # Procesamiento dual específico para PDFs de Gemini
        pdf_res = extract_pdf_structured(p)
        if extracted_txt_dir.exists() or prefix_group == "Gemini":
            extracted_txt_dir.mkdir(parents=True, exist_ok=True)
            (extracted_txt_dir / f"{p.stem}.txt").write_text(pdf_res["full_text"], encoding="utf-8")

        uris_blob = " ".join(pdf_res["uris"])
        resolved_uris_map = resolve_uri_batch(pdf_res["uris"])

        pdf_text_dois: set[str] = set()
        superscript_fixed: list[dict] = []
        truncated_or_failed: list[dict] = []

        for cand in pdf_res["doi_candidates"]:
            d = cand["doi_clean"]
            v = verify_doi_deterministic(d)
            if v["resolvio"]:
                pdf_text_dois.add(v["doi"])
                continue

            # Intentar quitar superíndice pegado (1..3 dígitos) validando contra contexto local ±800
            # Evitar recortar IDs numéricos de repositorios (figshare, zenodo, kaggle, dvn)
            matched_sub = None
            is_repo_id = any(r in d for r in ("figshare.", "zenodo.", "kaggle/", "dvn/"))
            if re.search(r"\d{1,3}$", d) and not is_repo_id:
                for trim in (1, 2, 3):
                    if len(d) - trim < 8:
                        continue
                    sub = clean_doi(d[:-trim])
                    if not sub or "/" not in sub:
                        continue
                    v_sub = verify_doi_deterministic(sub)
                    if v_sub["resolvio"] and context_matches_metadata(v_sub, cand["context"], ""):
                        matched_sub = v_sub["doi"]
                        superscript_fixed.append({
                            "raw_extracted": d,
                            "resolved_doi": matched_sub,
                            "trimmed_digits": trim,
                            "page": cand["page"],
                        })
                        break
            if matched_sub:
                pdf_text_dois.add(matched_sub)
            else:
                suffix_part = d.split("/", 1)[1] if "/" in d else d
                is_trunc = len(suffix_part) <= 6 or not re.search(r"[0-9a-z]{3,}", suffix_part)
                truncated_or_failed.append({
                    "doi_candidate": d,
                    "page": cand["page"],
                    "probable_truncation": is_trunc,
                })

        for d in sorted(pdf_text_dois):
            dois_found.append((label, d))

        for tr in pdf_res["table_rows"]:
            resolved_row_dois = [d for d in tr["dois"] if d in pdf_text_dois]
            if resolved_row_dois:
                table_rows.append({
                    "fuente_informe": label,
                    "dois": resolved_row_dois,
                    "campos": tr["campos"],
                })

        # Analizar DOIs resueltos únicamente vía anotaciones /URI (...)
        uri_resolved_total = 0
        uri_ophthalmic_added = 0
        uri_method_only = 0
        for u, info in resolved_uris_map.items():
            udoi = info.get("doi")
            if not udoi:
                continue
            uri_resolved_total += 1
            if udoi in pdf_text_dois:
                continue
            ver_u = verify_doi_deterministic(udoi)
            if not ver_u["resolvio"]:
                continue
            canon_doi = ver_u["doi"]
            if is_ophthalmic_study(ver_u):
                dois_found.append((label, canon_doi))
                pdf_text_dois.add(canon_doi)
                uri_ophthalmic_added += 1
            else:
                uri_method_only += 1
                rec = uri_only_records.setdefault(
                    canon_doi,
                    {"verif": ver_u, "informes": set(), "urls": set()},
                )
                rec["informes"].add(tag)
                rec["urls"].add(u)

        pdf_audit[p.name] = {
            "pages": pdf_res["n_pages"],
            "uris_unicas_pdf": len(pdf_res["uris"]),
            "uris_resueltas_a_doi": uri_resolved_total,
            "dois_desde_texto_y_tablas": len(pdf_text_dois) - uri_ophthalmic_added,
            "dois_anadidos_por_uri_oftalmica": uri_ophthalmic_added,
            "citas_uri_metodologicas_no_oftalmicas": uri_method_only,
            "superindices_desambiguados": superscript_fixed,
            "fragmentos_sin_resolver": truncated_or_failed,
        }

    return {
        "dois_found": dois_found,
        "table_rows": table_rows,
        "files_read": files_read,
        "pdf_audit": pdf_audit,
        "uri_only_records": uri_only_records,
    }


def build_reading_hint(ent: dict, doi_status: str, verif: dict, n_vias: int) -> str:
    hints: list[str] = []
    if n_vias >= 3:
        hints.append("Alta convergencia triple (Bases+Agentes+Gemini)")
    elif n_vias == 2:
        hints.append("Convergencia doble")
    if doi_status == "SI_DATACITE":
        hints.append("Preprint/dataset DataCite: verificar criterio E4 (publicación revisada por pares)")
    titulo_l = (verif.get("titulo_crossref") or "").lower()
    if any(w in titulo_l for w in ("review", "meta-analysis", "survey", "systematic")):
        hints.append("Posible revisión/survey: verificar para exclusión primaria o snowballing")
    if any(w in titulo_l for w in ("dataset", "database", "challenge", "benchmark")):
        hints.append("Posible paper de base de datos/desafío (contexto R2 / E3)")
    return " | ".join(hints)


def main() -> None:
    bd_rows = []
    if BD_CSV.exists():
        with BD_CSV.open("r", encoding="utf-8-sig", newline="") as f:
            bd_rows = list(csv.DictReader(f))

    ag_scan = scan_folder_reports(DR_AGENTES_DIR, "Agente")
    gm_scan = scan_folder_reports(DR_GEMINI_DIR, "Gemini")

    ag_dois, ag_tables, ag_files = ag_scan["dois_found"], ag_scan["table_rows"], ag_scan["files_read"]
    gm_dois, gm_tables, gm_files = gm_scan["dois_found"], gm_scan["table_rows"], gm_scan["files_read"]

    # Guardar auditoría específica de extracción PDF de Gemini
    if gm_scan["pdf_audit"]:
        (BASE_DIR / "busqueda_prisma" / "log_extraccion_pdf_gemini.json").write_text(
            json.dumps(gm_scan["pdf_audit"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Guardar referencias puramente metodológicas citadas vía URI sin término oftálmico
    meth_rows = []
    for mdoi, minfo in sorted(gm_scan["uri_only_records"].items()):
        v = minfo["verif"]
        meth_rows.append({
            "doi": mdoi,
            "registrador": v.get("registrador", ""),
            "anio": v.get("anio_crossref", ""),
            "autor_anio": v.get("autor_anio_crossref", ""),
            "titulo": v.get("titulo_crossref", ""),
            "revista_o_congreso": v.get("revista_crossref", ""),
            "informes_gemini": ", ".join(sorted(minfo["informes"])),
            "url_origen": "; ".join(sorted(minfo["urls"])[:3]),
        })
    meth_fields = [
        "doi", "registrador", "anio", "autor_anio", "titulo",
        "revista_o_congreso", "informes_gemini", "url_origen",
    ]
    with (BASE_DIR / "referencias_metodologicas_gemini.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=meth_fields)
        w.writeheader()
        w.writerows(meth_rows)

    # Mapa de pistas por DOI
    hints_by_doi: dict[str, list[dict]] = defaultdict(list)
    for row in ag_tables:
        for d in row["dois"]:
            hints_by_doi[d].append({"grupo": "Agente", "fuente": row["fuente_informe"], "campos": row["campos"]})
    for row in gm_tables:
        for d in row["dois"]:
            hints_by_doi[d].append({"grupo": "Gemini", "fuente": row["fuente_informe"], "campos": row["campos"]})

    # Recolectar todos los DOIs únicos
    all_dois_set: set[str] = set()
    for r in bd_rows:
        cd = clean_doi(r.get("doi"))
        if cd:
            all_dois_set.add(cd)
    for _, d in SEED_DOIS + ag_dois + gm_dois:
        cd = clean_doi(d)
        if cd:
            all_dois_set.add(cd)

    print(f"Archivos leídos -> Agentes: {len(ag_files)} {ag_files} | Gemini: {len(gm_files)} {gm_files}")
    print(f"Verificando {len(all_dois_set)} DOIs únicos (Crossref + respaldo DataCite)...")
    cr_map: dict[str, dict] = {}
    for idx, d in enumerate(sorted(all_dois_set), start=1):
        cr_map[d] = verify_doi_deterministic(d)
        if idx % 50 == 0:
            print(f"  [{idx}/{len(all_dois_set)}] DOIs verificados...")

    verif_fields = [
        "doi", "resolvio", "registrador", "autor_anio_crossref", "anio_crossref",
        "titulo_crossref", "revista_crossref", "tipo_crossref", "autores_crossref", "error"
    ]
    with (BASE_DIR / "verificacion_doi.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=verif_fields)
        w.writeheader()
        for d in sorted(cr_map.keys()):
            row_out = {k: cr_map[d].get(k, "") for k in verif_fields}
            w.writerow(row_out)

    # Consolidar orígenes por clave
    origins_by_key: dict[str, dict] = {}

    def ensure_entry(key: str, doi_val: str) -> dict:
        if key not in origins_by_key:
            origins_by_key[key] = {
                "doi": doi_val,
                "bd_row": None,
                "bases_list": [],
                "agentes_list": [],
                "gemini_list": [],
                "semilla": False,
            }
        return origins_by_key[key]

    for r in bd_rows:
        d = clean_doi(r.get("doi"))
        key = f"doi:{d}" if d else f"id:{r['id_prisma_bd']}"
        ent = ensure_entry(key, d)
        ent["bd_row"] = r
        for b in (r.get("fuentes_bd") or "").split(";"):
            bs = b.strip()
            if bs and bs not in ent["bases_list"]:
                ent["bases_list"].append(bs)

    for tag, d in SEED_DOIS:
        cd = clean_doi(d)
        if not cd:
            continue
        ent = ensure_entry(f"doi:{cd}", cd)
        ent["semilla"] = True

    for tag, d in ag_dois:
        cd = clean_doi(d)
        if not cd:
            continue
        ent = ensure_entry(f"doi:{cd}", cd)
        short_tag = tag.replace("Agente_", "")
        if short_tag not in ent["agentes_list"]:
            ent["agentes_list"].append(short_tag)

    for tag, d in gm_dois:
        cd = clean_doi(d)
        if not cd:
            continue
        ent = ensure_entry(f"doi:{cd}", cd)
        short_tag = tag.replace("Gemini_", "")
        if short_tag not in ent["gemini_list"]:
            ent["gemini_list"].append(short_tag)

    matriz_rows = []
    cribado_rows = []
    ot_counter = 1

    def via_prisma_of(ent: dict) -> str:
        has_bd = bool(ent["bases_list"])
        has_other = bool(ent["agentes_list"] or ent["gemini_list"] or ent["semilla"])
        if has_bd and has_other:
            return "ambas"
        if has_bd:
            return "bases_datos"
        return "otros_metodos"

    def origen_str_of(ent: dict) -> str:
        parts = []
        if ent["bases_list"]:
            parts.append(f"Bases ({'; '.join(ent['bases_list'])})")
        if ent["agentes_list"]:
            parts.append(f"Agentes ({', '.join(sorted(ent['agentes_list']))})")
        if ent["gemini_list"]:
            parts.append(f"Gemini ({', '.join(sorted(ent['gemini_list']))})")
        if ent["semilla"]:
            parts.append("Semilla_SOTA")
        return "; ".join(parts)

    sorted_keys = sorted(
        origins_by_key.keys(),
        key=lambda k: (
            -(
                (1 if origins_by_key[k]["bases_list"] else 0)
                + (1 if origins_by_key[k]["agentes_list"] else 0)
                + (1 if origins_by_key[k]["gemini_list"] else 0)
                + (1 if origins_by_key[k]["semilla"] else 0)
            ),
            -(len(origins_by_key[k]["agentes_list"]) + len(origins_by_key[k]["gemini_list"])),
            k,
        ),
    )

    for key in sorted_keys:
        ent = origins_by_key[key]
        d = ent["doi"]
        r = ent["bd_row"]
        cr = cr_map.get(d, {})
        via = via_prisma_of(ent)
        origen_str = origen_str_of(ent)

        if r is not None:
            rec_id = r["id_prisma_bd"]
        else:
            rec_id = f"OT-{ot_counter:04d}"
            ot_counter += 1

        if not d:
            doi_status = "SIN_DOI"
            registrador = "SIN_DOI"
        elif cr.get("resolvio"):
            registrador = cr.get("registrador", "Crossref")
            doi_status = f"SI_{registrador.upper()}"
        else:
            registrador = "NO_RESOLVIO"
            doi_status = "NO_RESOLVIO"

        autor_anio = (
            cr.get("autor_anio_crossref")
            or (f"{(r.get('autores') or '').split(',')[0]} ({r.get('anio','')})" if r else "")
        )
        titulo = cr.get("titulo_crossref") or (r.get("titulo", "") if r else "")
        venue = cr.get("revista_crossref") or (r.get("revista_o_congreso", "") if r else "")
        tipo_doc = cr.get("tipo_crossref") or (r.get("tipo_documento", "") if r else "")
        anio = cr.get("anio_crossref") or (r.get("anio", "") if r else "")
        resumen = (r.get("resumen", "") if r else "") or cr.get("abstract_crossref", "")

        mapped_hints: dict[str, str] = {}
        for h in hints_by_doi.get(d, []):
            ptag = "PISTA_GEMINI" if h["grupo"] == "Gemini" else "PISTA_AGENTE"
            m = map_table_row_to_fields(h["campos"], f"{ptag} ({h['fuente']})")
            for fld, val in m.items():
                if fld not in mapped_hints:
                    mapped_hints[fld] = val

        decision_init = "DESCARTADO_DOI_NO_RESOLVIO" if doi_status == "NO_RESOLVIO" else "PENDIENTE"
        notas = []
        if r:
            notas.append(f"PMID:{r.get('pmid','')} | OA:{r.get('acceso_abierto','')}")
        if doi_status == "SI_DATACITE":
            notas.append("Registrado en DataCite (preprint/repositorio; verificar si hay versión revisada por pares)")
        if doi_status == "NO_RESOLVIO":
            notas.append(f"DOI no resuelto en Crossref ni DataCite: {cr.get('error','')}")

        matriz_rows.append({
            "id": rec_id,
            "origen": origen_str,
            "via_prisma": via,
            "autor_año": autor_anio,
            "titulo": titulo,
            "revista_o_congreso": venue,
            "doi": d,
            "doi_verificado_crossref": doi_status,
            "tipo_documento": tipo_doc,
            "tarea_y_clase_positiva": mapped_hints.get("tarea_y_clase_positiva", ""),
            "bases_entrenamiento": mapped_hints.get("bases_entrenamiento", ""),
            "bases_prueba": mapped_hints.get("bases_prueba", ""),
            "tipo_validacion": mapped_hints.get("tipo_validacion", ""),
            "particion_por_paciente": mapped_hints.get("particion_por_paciente", ""),
            "arquitectura_preentrenamiento": mapped_hints.get("arquitectura_preentrenamiento", ""),
            "estrategia_mitigacion": mapped_hints.get("estrategia_mitigacion", ""),
            "fuente_variacion_que_aborda": mapped_hints.get("fuente_variacion_que_aborda", ""),
            "auc_interna": mapped_hints.get("auc_interna", ""),
            "auc_externa": mapped_hints.get("auc_externa", ""),
            "sens_esp_externa": mapped_hints.get("sens_esp_externa", ""),
            "umbral_como_se_fijo": mapped_hints.get("umbral_como_se_fijo", ""),
            "ubicacion_cifras": mapped_hints.get("ubicacion_cifras", ""),
            "texto_completo_leido": "NO",
            "quien_extrajo": "",
            "decision_cribado": decision_init,
            "revisor_1": "",
            "revisor_2": "",
            "notas": " | ".join(notas),
        })

        if doi_status != "NO_RESOLVIO":
            n_vias = (
                (1 if ent["bases_list"] else 0)
                + (1 if ent["agentes_list"] else 0)
                + (1 if ent["gemini_list"] else 0)
                + (1 if ent["semilla"] else 0)
            )
            cribado_rows.append({
                "id": rec_id,
                "via_prisma": via,
                "n_vias_convergentes": n_vias,
                "origen": origen_str,
                "anio": anio,
                "autor_año": autor_anio,
                "titulo": titulo,
                "revista_o_congreso": venue,
                "doi": d,
                "registrador_doi": registrador,
                "tipo_documento": tipo_doc,
                "acceso_abierto": r.get("acceso_abierto", "") if r else "",
                "oa_url": r.get("oa_url", "") if r else "",
                "resumen": (resumen or "")[:1800],
                "notas_apoyo_lectura": build_reading_hint(ent, doi_status, cr, n_vias),
                "revisor_1_decision": "",
                "revisor_1_criterio": "",
                "revisor_2_decision": "",
                "revisor_2_criterio": "",
                "consenso_final": "",
                "notas_cribado": "",
            })

    matriz_fields = [
        "id", "origen", "via_prisma", "autor_año", "titulo", "revista_o_congreso", "doi",
        "doi_verificado_crossref", "tipo_documento", "tarea_y_clase_positiva",
        "bases_entrenamiento", "bases_prueba", "tipo_validacion",
        "particion_por_paciente", "arquitectura_preentrenamiento",
        "estrategia_mitigacion", "fuente_variacion_que_aborda", "auc_interna",
        "auc_externa", "sens_esp_externa", "umbral_como_se_fijo",
        "ubicacion_cifras", "texto_completo_leido", "quien_extrajo",
        "decision_cribado", "revisor_1", "revisor_2", "notas"
    ]
    with (BASE_DIR / "02_matriz_corpus.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=matriz_fields)
        w.writeheader()
        w.writerows(matriz_rows)

    cribado_fields = [
        "id", "via_prisma", "n_vias_convergentes", "origen", "anio", "autor_año",
        "titulo", "revista_o_congreso", "doi", "registrador_doi", "tipo_documento",
        "acceso_abierto", "oa_url", "resumen", "notas_apoyo_lectura",
        "revisor_1_decision", "revisor_1_criterio",
        "revisor_2_decision", "revisor_2_criterio",
        "consenso_final", "notas_cribado"
    ]
    with (BASE_DIR / "cribado_prisma_humano.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cribado_fields)
        w.writeheader()
        w.writerows(cribado_rows)

    # Muestra ciega estratificada 20 % (min 40) para segundo revisor humano (kappa real)
    rng = random.Random(42)
    target_n = max(40, math.ceil(0.20 * len(cribado_rows))) if cribado_rows else 0
    by_strata: dict[str, list[dict]] = defaultdict(list)
    for row in cribado_rows:
        y_bin = "2024_2026" if str(row.get("anio", "")) >= "2024" else "2016_2023"
        by_strata[f"{row['via_prisma']}_{y_bin}"].append(row)

    blind_sample: list[dict] = []
    if cribado_rows:
        allocated = 0
        strata_keys = sorted(by_strata.keys())
        for idx, sk in enumerate(strata_keys):
            pool = list(by_strata[sk])
            rng.shuffle(pool)
            if idx == len(strata_keys) - 1:
                take = max(1, target_n - allocated)
            else:
                take = max(1, round(target_n * len(pool) / len(cribado_rows)))
            take = min(take, len(pool))
            blind_sample.extend(pool[:take])
            allocated += take
        if len(blind_sample) < target_n:
            chosen_ids = {x["id"] for x in blind_sample}
            rem = [r for r in cribado_rows if r["id"] not in chosen_ids]
            rng.shuffle(rem)
            blind_sample.extend(rem[: target_n - len(blind_sample)])
        rng.shuffle(blind_sample)

    blind_rows = [
        {
            "id": row["id"],
            "anio": row["anio"],
            "autor_año": row["autor_año"],
            "titulo": row["titulo"],
            "revista_o_congreso": row["revista_o_congreso"],
            "doi": row["doi"],
            "resumen": row["resumen"],
            "revisor_2_decision": "",
            "revisor_2_criterio": "",
            "notas_revisor_2": "",
        }
        for row in blind_sample
    ]
    blind_fields = [
        "id", "anio", "autor_año", "titulo", "revista_o_congreso", "doi", "resumen",
        "revisor_2_decision", "revisor_2_criterio", "notas_revisor_2",
    ]
    with (BASE_DIR / "cribado_muestra_ciega_revisor2.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=blind_fields)
        w.writeheader()
        w.writerows(blind_rows)

    # Cruce comparativo lado a lado para DOIs mencionados en >1 informe o en Gemini
    comp_rows = []
    for d, items in sorted(hints_by_doi.items()):
        if len(items) > 1 or any(x["grupo"] == "Gemini" for x in items):
            cr = cr_map.get(d, {})
            for item in items:
                comp_rows.append({
                    "doi": d,
                    "resolvio_doi": cr.get("resolvio", False),
                    "registrador": cr.get("registrador", ""),
                    "autor_anio_ref": cr.get("autor_anio_crossref", ""),
                    "grupo_fuente": item["grupo"],
                    "informe": item["fuente"],
                    "fila_extraida_json": json.dumps(item["campos"], ensure_ascii=False),
                })

    comp_fields = ["doi", "resolvio_doi", "registrador", "autor_anio_ref", "grupo_fuente", "informe", "fila_extraida_json"]
    with (BASE_DIR / "cruce_comparativo_informes.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=comp_fields)
        w.writeheader()
        w.writerows(comp_rows)

    bd_set = {k for k, e in origins_by_key.items() if e["bases_list"]}
    ag_set = {k for k, e in origins_by_key.items() if e["agentes_list"]}
    gm_set = {k for k, e in origins_by_key.items() if e["gemini_list"]}
    sem_set = {k for k, e in origins_by_key.items() if e["semilla"]}
    invalid_dois = [r for r in matriz_rows if r["doi_verificado_crossref"] == "NO_RESOLVIO"]
    datacite_rows = [r for r in matriz_rows if r["doi_verificado_crossref"] == "SI_DATACITE"]
    crossref_rows = [r for r in matriz_rows if r["doi_verificado_crossref"] == "SI_CROSSREF"]

    summary = {
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "archivos_agentes_leidos": ag_files,
        "archivos_gemini_leidos": gm_files,
        "via1_bases_unicos": len(bd_set),
        "agentes_antigravity_unicos": len(ag_set),
        "gemini_deep_research_unicos": len(gm_set),
        "semillas_sota": len(sem_set),
        "referencias_metodologicas_gemini_separadas": len(meth_rows),
        "solapamiento": {
            "bases_y_agentes": len(bd_set & ag_set),
            "bases_y_gemini": len(bd_set & gm_set),
            "agentes_y_gemini": len(ag_set & gm_set),
            "tres_vias_bases_agentes_gemini": len(bd_set & ag_set & gm_set),
            "solo_agentes": len(ag_set - bd_set - gm_set - sem_set),
            "solo_gemini": len(gm_set - bd_set - ag_set - sem_set),
            "solo_bases": len(bd_set - ag_set - gm_set - sem_set),
        },
        "verificacion_doi": {
            "total_unicos": len(cr_map),
            "si_crossref": len(crossref_rows),
            "si_datacite_preprint_o_dataset": len(datacite_rows),
            "no_resolvio_descartados": len(invalid_dois),
        },
        "total_matriz_corpus": len(matriz_rows),
        "total_cribable_sin_doi_invalido": len(cribado_rows),
        "muestra_ciega_revisor2": len(blind_rows),
    }
    (BASE_DIR / "busqueda_prisma" / "conteos_prisma_preliminares.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    top_conv = [r for r in cribado_rows if r["n_vias_convergentes"] >= 2][:40]
    md = [
        "# Resumen del Cruce Multifuente PRISMA 2020",
        "",
        f"**Actualizado (UTC):** {summary['updated_at_utc']}",
        f"**Informes Agentes Antigravity (`deep_research_agentes/`):** {len(ag_files)} ({', '.join(ag_files) if ag_files else 'ninguno'})",
        f"**Informes Gemini Web Deep Research (`deep_research/`):** {len(gm_files)} ({', '.join(gm_files) if gm_files else 'ninguno'})",
        "",
        "## 1. Conteos y Solapamiento entre Fuentes",
        "",
        "| Fuente / Intersección | Registros únicos |",
        "| --- | ---: |",
        f"| Vía 1: Bases indexadas deduplicadas (OpenAlex + PubMed + Europe PMC) | {summary['via1_bases_unicos']} |",
        f"| Vía 2a: Agentes Antigravity (`A1`–`A6`) | {summary['agentes_antigravity_unicos']} |",
        f"| Vía 2b: Gemini Deep Research (`R2`–`R6`) | {summary['gemini_deep_research_unicos']} |",
        f"| Semillas SOTA previo | {summary['semillas_sota']} |",
        f"| Intersección Bases ∩ Agentes | {summary['solapamiento']['bases_y_agentes']} |",
        f"| Intersección Bases ∩ Gemini | {summary['solapamiento']['bases_y_gemini']} |",
        f"| Intersección Agentes ∩ Gemini | {summary['solapamiento']['agentes_y_gemini']} |",
        f"| Convergencia triple (Bases ∩ Agentes ∩ Gemini) | {summary['solapamiento']['tres_vias_bases_agentes_gemini']} |",
        f"| Exclusivos en Gemini respecto al resto del corpus | {summary['solapamiento']['solo_gemini']} |",
        f"| Exclusivos en Agentes respecto al resto del corpus | {summary['solapamiento']['solo_agentes']} |",
        f"| Exclusivos en Bases API respecto al resto del corpus | {summary['solapamiento']['solo_bases']} |",
        f"| **Total corpus consolidado en `02_matriz_corpus.csv`** | **{summary['total_matriz_corpus']}** |",
        f"| **Corpus cribable válido en `cribado_prisma_humano.csv`** | **{summary['total_cribable_sin_doi_invalido']}** |",
        f"| Referencias metodológicas generales separadas vía URI (`referencias_metodologicas_gemini.csv`) | {summary['referencias_metodologicas_gemini_separadas']} |",
        f"| Muestra ciega estratificada para Revisor 2 (`cribado_muestra_ciega_revisor2.csv`, 20 %) | {summary['muestra_ciega_revisor2']} |",
        "",
        "## 2. Auditoría Determinística de DOI (Crossref + DataCite)",
        "",
        f"- **Resueltos en Crossref (artículos/actas):** {summary['verificacion_doi']['si_crossref']}",
        f"- **Resueltos en DataCite (preprints arXiv / repositorios Zenodo, revisables para criterio E4):** {summary['verificacion_doi']['si_datacite_preprint_o_dataset']}",
        f"- **No resueltos (`NO_RESOLVIO`, bloqueados de `cribado_prisma_humano.csv`):** {summary['verificacion_doi']['no_resolvio_descartados']}",
        "",
    ]
    if invalid_dois:
        md.append("### DOIs descartados por no resolver en Crossref ni DataCite")
        for r in invalid_dois:
            md.append(f"- `{r['doi']}` (origen: {r['origen']})")
        md.append("")

    md.append("## 3. Estudios con mayor convergencia entre vías (muestra superior)")
    md.append("")
    md.append("| ID | Vías | Origen | Año | Autor | DOI | Registrador |")
    md.append("| --- | ---: | --- | --- | --- | --- | --- |")
    for r in top_conv:
        md.append(
            f"| {r['id']} | {r['n_vias_convergentes']} | {r['origen']} | {r['anio']} | "
            f"{r['autor_año']} | `{r['doi']}` | {r['registrador_doi']} |"
        )
    md.append("")
    (BASE_DIR / "cruce_informacion_resumen.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
