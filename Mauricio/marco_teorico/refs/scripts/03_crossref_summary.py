# -*- coding: utf-8 -*-
"""Resume los campos bibliograficos de cada JSON de Crossref en refs/crossref."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in sorted((ROOT / "crossref").glob("*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))["message"]
    if "items" in d:
        print(f"## {p.stem} (busqueda)")
        for it in d["items"]:
            print("   -", it.get("DOI"), "|", (it.get("title") or [""])[0][:110], "|", (it.get("container-title") or [""])[0][:60],
                  "|", (it.get("issued", {}).get("date-parts") or [[None]])[0][0], "|", ", ".join(a.get("family", "") for a in it.get("author", [])[:4]))
        continue
    au = "; ".join(f"{a.get('family','')}, {a.get('given','')}" for a in d.get("author", []))
    yr = lambda k: (d.get(k, {}).get("date-parts") or [[None]])[0]
    print(f"## {p.stem}\n DOI: {d.get('DOI')}\n type: {d.get('type')}\n title: {(d.get('title') or [''])[0]}\n authors: {au}\n"
          f" container: {(d.get('container-title') or [''])} | short: {d.get('short-container-title')}\n vol: {d.get('volume')} issue: {d.get('issue')} pages: {d.get('page')} art: {d.get('article-number')}\n"
          f" issued: {yr('issued')} | print: {yr('published-print')} | online: {yr('published-online')}\n publisher: {d.get('publisher')} | event: {(d.get('event') or {}).get('name')}\n"
          f" update-to/relation: {d.get('update-to')} {list((d.get('relation') or {}).keys())}\n")
