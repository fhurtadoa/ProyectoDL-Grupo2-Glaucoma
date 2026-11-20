# -*- coding: utf-8 -*-
"""Convierte HTML de PMC/USPSTF/PMLR/IJCAI y XML de NCBI/EPMC a texto limpio en refs/txt/."""
import html, json, re, xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TXT, HTML, XML = ROOT / "txt", ROOT / "html", ROOT / "xml"
TXT.mkdir(exist_ok=True)

def clean_html_article(raw, selector="pmc"):
    if selector == "pmc":
        m = re.search(r'<section[^>]+aria-label="Article content".*?</article>|<div[^>]+class="[^"]*jig-ncbiinpagenav[^"]*".*|<main[^>]*>.*?</main>', raw, re.S | re.I)
        frag = m.group(0) if m else raw
    elif selector == "uspstf":
        m = re.search(r'<main[^>]*>.*?</main>', raw, re.S | re.I)
        frag = m.group(0) if m else raw
    else:
        frag = raw
    frag = re.sub(r'<(script|style|noscript)[^>]*>.*?</\1>', ' ', frag, flags=re.S | re.I)
    frag = re.sub(r'</(p|div|section|h[1-6]|li|tr|br)\s*>', '\n', frag, flags=re.I)
    frag = re.sub(r'<[^>]+>', ' ', frag)
    frag = html.unescape(frag)
    lines = [re.sub(r'\s+', ' ', ln).strip() for ln in frag.splitlines()]
    return "\n".join(ln for ln in lines if ln)

for name, kind in [
    ("01_weinreb_pmc.html", "pmc"),
    ("03_ting_pmc.html", "pmc"),
    ("17_vickers_pmc.html", "pmc"),
    ("19_ab_pv_pmc.html", "pmc"),
    ("20_ab_ss_pmc.html", "pmc"),
    ("22_uspstf2022.html", "uspstf"),
    ("15_guo2017_pmlr.html", "other"),
]:
    p = HTML / name
    if p.exists():
        txt = clean_html_article(p.read_text(encoding="utf-8", errors="replace"), kind)
        out = TXT / (name.replace(".html", ".txt"))
        out.write_text(txt, encoding="utf-8")
        print(out.name, len(txt), "chars")

# Parse 14_elkan2001 entry in IJCAI 2001-2 HTML
ij = HTML / "14_elkan2001_ijcai_2001-2.html"
if ij.exists():
    raw = ij.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'.{0,300}Foundations of Cost-Sensitive Learning.{0,500}', raw, re.S | re.I)
    if m:
        print("IJCAI snippet:", re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', html.unescape(m.group(0)))))

# Parse 25_abramoff2010_ncbi.xml
xp = XML / "25_abramoff2010_ncbi.xml"
if xp.exists():
    root = ET.fromstring(xp.read_text(encoding="utf-8", errors="replace"))
    body_txt = " ".join("".join(el.itertext()) for el in root.findall(".//body"))
    abs_txt = " ".join("".join(el.itertext()) for el in root.findall(".//abstract"))
    out = TXT / "25_abramoff2010_xml.txt"
    out.write_text(re.sub(r'\s+', ' ', abs_txt + "\n\n" + body_txt), encoding="utf-8")
    print(out.name, out.stat().st_size, "bytes")
