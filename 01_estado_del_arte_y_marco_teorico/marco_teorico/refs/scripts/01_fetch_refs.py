# -*- coding: utf-8 -*-
"""F1b - Descarga de metadatos (Crossref, Europe PMC) y PDF de acceso abierto.
Solo stdlib. No intenta saltar ningun control de acceso: si un servidor
devuelve HTML (desafio anti-bot, muro de pago), se registra y se omite.
Salidas: refs/crossref/<key>.json, refs/pdf/<key>.pdf, refs/xml/<key>.xml,
refs/fetch_log.json
"""
import json, re, time, urllib.request, urllib.parse, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # marco_teorico/refs
CR, PDF, XML, HTML = ROOT / "crossref", ROOT / "pdf", ROOT / "xml", ROOT / "html"
for d in (CR, PDF, XML, HTML):
    d.mkdir(parents=True, exist_ok=True)
UA = "Mozilla/5.0 (compatible; marco-teorico-verificacion/1.0; academic use)"

def get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.geturl(), r.headers.get("Content-Type", ""), r.read()

# key: (doi or None, [candidate pdf urls], [html pages to save])
T = {
 # ---- Tarea 1 / 2 ----
 "01_weinreb2014": ("10.1001/jama.2014.3192", [], []),
 "02_tham2014": ("10.1016/j.ophtha.2014.05.013", ["https://www.aaojournal.org/article/S0161-6420(14)00433-3/pdf"], []),
 "03_ting2017": ("10.1001/jama.2017.18152", [], []),
 "04_diazpinto2019": ("10.1186/s12938-019-0649-y", ["https://biomedical-engineering-online.biomedcentral.com/counter/pdf/10.1186/s12938-019-0649-y.pdf"], []),
 "05_fumero2011": ("10.1109/CBMS.2011.5999143", [], []),
 "06a_doi_8840759": ("10.1155/2020/8840759", [], []),
 "06b_rimonedl_ias2020": ("10.5566/ias.2346", ["IAS"], []),
 "07_convnext2022": ("10.1109/CVPR52688.2022.01167", ["https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf", "https://arxiv.org/pdf/2201.03545"], []),
 "08_resnet2016": ("10.1109/CVPR.2016.90", ["https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf", "https://arxiv.org/pdf/1512.03385"], []),
 "09_swin2021": ("10.1109/ICCV48922.2021.00986", ["https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf", "https://arxiv.org/pdf/2103.14030"], []),
 # ---- Tarea 3 (seccion 3.5) ----
 "10_fawcett2006": ("10.1016/j.patrec.2005.10.010", [], []),
 "11_metz1978": ("10.1016/S0001-2998(78)80014-2", [], []),
 "12_youden1950": ("10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3", [], []),
 "13_habibzadeh2016": ("10.11613/BM.2016.034", [], []),
 "14_elkan2001": (None, ["IJCAI", "https://cseweb.ucsd.edu/~elkan/rescale.pdf"], []),
 "15_guo2017": (None, ["https://proceedings.mlr.press/v70/guo17a/guo17a.pdf"], []),
 "16_saerens2002": ("10.1162/089976602753284446", [], []),
 "17_vickers2006": ("10.1177/0272989X06295361", [], []),
 "18_wilson1968": (None, ["https://iris.who.int/bitstream/handle/10665/37650/WHO_PHP_34.pdf", "https://apps.who.int/iris/bitstream/handle/10665/37650/WHO_PHP_34.pdf"], []),
 "19_altmanbland1994_pv": ("10.1136/bmj.309.6947.102", [], []),
 "20_altmanbland1994_ss": ("10.1136/bmj.308.6943.1552", [], []),
 "21_trevethan2017": ("10.3389/fpubh.2017.00307", [], []),
 # ---- Tamizaje de glaucoma (tarea 2b) ----
 "22_uspstf2022": ("10.1001/jama.2022.7013", [], ["https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/primary-open-angle-glaucoma-screening"]),
 "23_chou2022": ("10.1001/jama.2022.6290", [], []),
 "24_prum2016_ppp": ("10.1016/j.ophtha.2015.10.053", ["https://www.aaojournal.org/article/S0161-6420(15)01276-2/pdf"], []),
 # ---- Fuentes alternativas tarea 2 (d, e, f) ----
 "25_abramoff2010": ("10.1109/RBME.2010.2084567", [], []),
 "26_castro2020": ("10.1038/s41467-020-17478-w", ["https://www.nature.com/articles/s41467-020-17478-w.pdf"], []),
 "27_geirhos2020": ("10.1038/s42256-020-00257-z", ["https://arxiv.org/pdf/2004.07780"], []),
 "28_zech2018": ("10.1371/journal.pmed.1002683", ["https://journals.plos.org/plosmedicine/article/file?id=10.1371/journal.pmed.1002683&type=printable"], []),
}

QUERIES = {  # busquedas bibliograficas en Crossref (para localizar DOI reales)
 "q_rimonedl": "RIM-ONE DL unified retinal image database assessing glaucoma deep learning",
 "q_elkan": "Elkan foundations of cost-sensitive learning",
 "q_guo": "On calibration of modern neural networks Guo Pleiss Sun Weinberger",
 "q_wilson": "Wilson Jungner principles and practice of screening for disease",
}

log = {}
def save_pdf(key, url):
    try:
        st, final, ct, data = get(url)
        if data[:5] == b"%PDF-":
            (PDF / f"{key}.pdf").write_bytes(data)
            return {"url": url, "final": final, "status": "PDF_OK", "bytes": len(data)}
        return {"url": url, "final": final, "status": f"NO_PDF ({ct})", "head": data[:200].decode("latin1", "replace")}
    except Exception as e:
        return {"url": url, "status": f"ERROR {type(e).__name__}: {e}"}

for key, (doi, urls, pages) in T.items():
    ent = {"doi": doi, "attempts": []}
    if doi:
        try:
            st, _, _, data = get("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/()<>:;.-_"))
            (CR / f"{key}.json").write_bytes(data)
            ent["crossref"] = "OK"
        except Exception as e:
            ent["crossref"] = f"ERROR {e}"
        try:
            q = urllib.parse.quote(f'DOI:"{doi}"')
            st, _, _, data = get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}&format=json&resultType=lite")
            res = json.loads(data).get("resultList", {}).get("result", [])
            if res:
                r0 = res[0]
                ent["epmc"] = {k: r0.get(k) for k in ("pmid", "pmcid", "isOpenAccess", "inEPMC", "title")}
                pmc = r0.get("pmcid")
                if pmc:
                    urls = urls + [f"https://europepmc.org/articles/{pmc}?pdf=render"]
                    try:
                        st, _, _, x = get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML")
                        (XML / f"{key}.xml").write_bytes(x); ent["xml"] = "OK"
                    except Exception as e:
                        ent["xml"] = f"ERROR {e}"
        except Exception as e:
            ent["epmc"] = f"ERROR {e}"
    for u in urls:
        if u == "IAS":
            try:
                _, _, _, h = get("https://www.ias-iss.org/ojs/IAS/article/view/2346")
                (HTML / f"{key}_landing.html").write_bytes(h)
                links = sorted(set(re.findall(rb'href="([^"]*article/(?:view|download)/2346/[^"]+)"', h)))
                ent["ias_links"] = [l.decode() for l in links]
                for l in links:
                    l2 = l.decode().replace("/article/view/", "/article/download/")
                    a = save_pdf(key, l2); ent["attempts"].append(a)
                    if a["status"] == "PDF_OK": break
            except Exception as e:
                ent["attempts"].append({"url": "IAS landing", "status": f"ERROR {e}"})
        elif u == "IJCAI":
            try:
                _, _, _, h = get("https://www.ijcai.org/proceedings/2001-2")
                m = re.search(rb'Foundations of Cost-Sensitive Learning.{0,800}?href="([^"]+\.pdf)"', h, re.S | re.I) or \
                    re.search(rb'href="([^"]+\.pdf)"[^<]{0,400}?</a>[^<]*<[^>]*>[^<]*Foundations of Cost', h, re.S | re.I)
                ent["ijcai_match"] = m.group(1).decode() if m else None
                (HTML / f"{key}_ijcai_2001-2.html").write_bytes(h)
                if m:
                    href = urllib.parse.urljoin("https://www.ijcai.org/proceedings/2001-2", m.group(1).decode())
                    ent["attempts"].append(save_pdf(key, href))
            except Exception as e:
                ent["attempts"].append({"url": "IJCAI listing", "status": f"ERROR {e}"})
        else:
            a = save_pdf(key, u); ent["attempts"].append(a)
        if (PDF / f"{key}.pdf").exists():
            break
        time.sleep(0.5)
    for p in pages:
        try:
            _, _, ct, h = get(p); (HTML / f"{key}.html").write_bytes(h); ent["html"] = f"OK {ct}"
        except Exception as e:
            ent["html"] = f"ERROR {e}"
    ent["pdf"] = (PDF / f"{key}.pdf").exists()
    log[key] = ent
    print(key, "| crossref:", ent.get("crossref"), "| pdf:", ent["pdf"], "| epmc:", (ent.get("epmc") or {}).get("pmcid") if isinstance(ent.get("epmc"), dict) else ent.get("epmc"))
    time.sleep(0.3)

for k, q in QUERIES.items():
    try:
        _, _, _, data = get("https://api.crossref.org/works?rows=5&query.bibliographic=" + urllib.parse.quote(q))
        (CR / f"{k}.json").write_bytes(data); log[k] = "OK"
    except Exception as e:
        log[k] = f"ERROR {e}"

(ROOT / "fetch_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
print("DONE")
