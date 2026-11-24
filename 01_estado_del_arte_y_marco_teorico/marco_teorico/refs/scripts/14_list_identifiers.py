import json, xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path(r"C:\Users\Mau\Downloads\Proyecto Final DL\ProyectoDL-Grupo2-Glaucoma\Mauricio\marco_teorico\refs")
CR = BASE / "crossref"
keys = [
    "02_tham2014", "05_fumero2011", "11_metz1978", "12_youden1950",
    "16_saerens2002", "23_chou2022", "24_prum2016_ppp",
    "19_altmanbland1994_pv", "20_altmanbland1994_ss",
]
pmids = {}
r = ET.parse(BASE / "xml" / "pubmed_paywalled.xml").getroot()
for art in r.findall(".//PubmedArticle"):
    pmid = art.findtext(".//MedlineCitation/PMID")
    for aid in art.findall(".//ArticleId"):
        if aid.attrib.get("IdType") == "doi":
            pmids[aid.text.lower()] = pmid

for k in keys:
    d = json.loads((CR / f"{k}.json").read_text(encoding="utf-8"))["message"]
    doi = d.get("DOI", "")
    au = ", ".join(a.get("family", "") for a in d.get("author", [])[:3])
    if len(d.get("author", [])) > 3:
        au += " et al."
    tit = " ".join((d.get("title") or [""])[0].split())
    cont = (d.get("container-title") or [""])[0]
    yr = (d.get("issued", {}).get("date-parts") or [[None]])[0][0]
    issn = d.get("ISSN", [])
    isbn = d.get("ISBN", [])
    pmid = pmids.get(doi.lower(), "8038641" if "102" in doi else ("8019315" if "1552" in doi else "N/A"))
    print(f"{k}: {au} ({yr}) | {tit} | {cont} {d.get('volume')}({d.get('issue')}):{d.get('page')} | DOI={doi} | PMID={pmid} | ISSN={issn} | ISBN={isbn}")
