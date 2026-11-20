# -*- coding: utf-8 -*-
"""F1 - Descarga metadatos Crossref/EPMC y PDFs abiertos para secciones 2.1-2.5."""
import json, subprocess, urllib.parse
from pathlib import Path

ROOT = Path(r"C:\Users\Mau\Downloads\Proyecto Final DL\ProyectoDL-Grupo2-Glaucoma\Mauricio\marco_teorico\refs")
CR, PDF, HTML, XML = ROOT / "crossref", ROOT / "pdf", ROOT / "html", ROOT / "xml"

def curl_get(url, out_path=None, timeout=25):
    cmd = ["curl.exe", "-sL", "-m", str(timeout), "-A", "Mozilla/5.0", url]
    if out_path:
        cmd = ["curl.exe", "-sL", "-m", str(timeout), "-A", "Mozilla/5.0", "-o", str(out_path), url]
        subprocess.run(cmd, check=False)
        return out_path.exists() and out_path.stat().st_size > 0
    r = subprocess.run(cmd, capture_output=True)
    return r.stdout

# (key, doi_or_none, crossref_query_or_none, open_pdf_url_or_none)
ITEMS = [
    ("30_spaeth2002_ddls", None, "The disc damage likelihood scale reproducibility of a new method Spaeth 2002", None),
    ("31_harizman2006_isnt", "10.1001/archopht.124.11.1579", None, None),
    ("32_jonas2017_lancet", "10.1016/S0140-6736(17)31469-1", None, None),
    ("33_reinhard2001", None, "Color transfer between images Reinhard Ashikhmin Gooch Shirley 2001", None),
    ("34_lecun2015", None, "Deep learning LeCun Bengio Hinton Nature 2015", None),
    ("35_vaswani2017", None, "Attention is all you need Vaswani Shazeer Parmar 2017", "https://arxiv.org/pdf/1706.03762"),
    ("36_dosovitskiy2021_vit", None, "An Image is Worth 16x16 Words Transformers for Image Recognition at Scale Dosovitskiy", "https://arxiv.org/pdf/2010.11929"),
    ("37_yosinski2014", None, "How transferable are features in deep neural networks Yosinski Clune Bengio Lipson", "https://arxiv.org/pdf/1411.1792"),
    ("38_loshchilov2017_sgdr", None, "SGDR Stochastic Gradient Descent with Warm Restarts Loshchilov Hutter", "https://arxiv.org/pdf/1608.03983"),
    ("39_loshchilov2019_adamw", None, "Decoupled Weight Decay Regularization Loshchilov Hutter", "https://arxiv.org/pdf/1711.05101"),
    ("40_deng2009_imagenet", None, "ImageNet A large-scale hierarchical image database Deng Dong Socher Li", None),
    ("41_bendavid2010", "10.1007/s10994-009-5152-4", None, "https://link.springer.com/content/pdf/10.1007/s10994-009-5152-4.pdf"),
    ("42_zhou2023_dg", "10.1109/TPAMI.2022.3195549", None, "https://arxiv.org/pdf/2103.02503"),
    ("43_wang2018_da", None, "Deep visual domain adaptation A survey Wang Deng Neurocomputing 2018", "https://arxiv.org/pdf/1802.10301"),
    ("44_shorten2019_da", None, "A survey on Image Data Augmentation for Deep Learning Shorten Khoshgoftaar 2019", None),
    ("45_zhang2018_mixup", None, "mixup Beyond Empirical Risk Minimization Zhang Cisse Dauphin Lopez-Paz", "https://arxiv.org/pdf/1710.09412"),
    ("46_nakkiran2020_dd", None, "Deep Double Descent Where Bigger Models and More Data Hurt Nakkiran", "https://arxiv.org/pdf/1912.02292"),
    ("47_power2022_grokking", None, "Grokking Generalization Beyond Overfitting on Small Algorithmic Datasets Power", "https://arxiv.org/pdf/2201.02177"),
    ("48_morenotorres2012", None, "A unifying view on dataset shift in classification Moreno-Torres Raeder Alaiz-Rodriguez", None),
    ("49_shimodaira2000", None, "Improving predictive inference under covariate shift by weighting the log-likelihood function Shimodaira 2000", None),
    ("50_collins2015_tripod", "10.1136/bmj.g7594", None, None),
    ("51_gulrajani2021_domainbed", None, "In Search of Lost Domain Generalization Gulrajani Lopez-Paz", "https://arxiv.org/pdf/2007.01434"),
    ("52_lipton2018_labelshift", None, "Detecting and Correcting for Label Shift with Black Box Predictors Lipton Wang Smola", "https://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf"),
    ("53_cohen1968_kappa", None, "Weighted kappa Nominal scale agreement with provision for scaled disagreement Cohen 1968", None),
    ("54_delong1988", None, "Comparing the areas under two or more correlated receiver operating characteristic curves DeLong 1988", None),
    ("55_hanley1982", None, "The meaning and use of the area under a receiver operating characteristic ROC curve Hanley McNeil 1982", None),
]

res = {}
for key, doi, q, pdf_url in ITEMS:
    info = {"doi": doi, "pdf": False, "pmc": None}
    if doi:
        raw = curl_get("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/()<>:;.-_"))
        if raw and raw.startswith(b"{"):
            (CR / f"{key}.json").write_bytes(raw)
    elif q:
        raw = curl_get("https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(q))
        if raw and raw.startswith(b"{"):
            (CR / f"{key}_search.json").write_bytes(raw)
            try:
                items = json.loads(raw)["message"]["items"]
                if items:
                    info["top_cr"] = {
                        "DOI": items[0].get("DOI"),
                        "title": (items[0].get("title") or [""])[0][:90],
                        "container": (items[0].get("container-title") or [""])[0][:50],
                        "year": (items[0].get("issued", {}).get("date-parts") or [[None]])[0][0],
                        "page": items[0].get("page") or items[0].get("article-number"),
                    }
                    doi = items[0].get("DOI")
            except Exception:
                pass
    # Check EPMC for PMCID / OA PDF
    if doi:
        ep = curl_get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=" + urllib.parse.quote(f'DOI:"{doi}"') + "&format=json&resultType=lite")
        try:
            r0 = json.loads(ep).get("resultList", {}).get("result", [{}])[0]
            if r0.get("pmcid"):
                info["pmc"] = r0["pmcid"]
                curl_get(f"https://pmc.ncbi.nlm.nih.gov/articles/{r0['pmcid']}/", HTML / f"{key}_pmc.html")
        except Exception:
            pass
    # Special check for Spaeth 2002 PMC1447547 and Shorten 2019 SpringerOpen
    if key == "30_spaeth2002_ddls":
        info["pmc"] = "PMC1447547"
        curl_get("https://pmc.ncbi.nlm.nih.gov/articles/PMC1447547/", HTML / f"{key}_pmc.html")
    if key == "44_shorten2019_da" and info.get("top_cr", {}).get("DOI"):
        d44 = info["top_cr"]["DOI"]
        pdf_url = f"https://journalofbigdata.springeropen.com/counter/pdf/{d44}.pdf"
    if pdf_url:
        pout = PDF / f"{key}.pdf"
        curl_get(pdf_url, pout, timeout=35)
        if pout.exists():
            head = pout.read_bytes()[:6]
            if head.startswith(b"%PDF-"):
                info["pdf"] = True
            else:
                pout.unlink()
    res[key] = info
    print(key, "| pdf:", info["pdf"], "| pmc:", info["pmc"], "| cr:", info.get("top_cr") or doi)

(ROOT / "fetch_f1_log.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print("DONE F1 FETCH")
