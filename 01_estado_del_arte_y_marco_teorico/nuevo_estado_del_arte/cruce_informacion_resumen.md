# Resumen del Cruce Multifuente PRISMA 2020

**Actualizado (UTC):** 2026-11-24T23:52:54.608856+00:00
**Informes Agentes Antigravity (`deep_research_agentes/`):** 6 (A1_validacion_externa.md, A2_catalogo_bases.md, A3_variacion_imagen.md, A4_variacion_etiqueta_estadios.md, A5_umbral_calibracion_prevalencia.md, A6_estrategias_mitigacion.md)
**Informes Gemini Web Deep Research (`deep_research/`):** 6 (R1_validacion_externa.pdf, R2_catalogo_bases.pdf, R3_variacion_imagen.pdf, R4_variacion_etiqueta_estadios.pdf, R5_umbral_calibracion_prevalencia.pdf, R6_estrategias_mitigacion.pdf)

## 1. Conteos y Solapamiento entre Fuentes

| Fuente / Intersección | Registros únicos |
| --- | ---: |
| Vía 1: Bases indexadas deduplicadas (OpenAlex + PubMed + Europe PMC) | 131 |
| Vía 2a: Agentes Antigravity (`A1`–`A6`) | 117 |
| Vía 2b: Gemini Deep Research (`R2`–`R6`) | 227 |
| Semillas SOTA previo | 9 |
| Intersección Bases ∩ Agentes | 34 |
| Intersección Bases ∩ Gemini | 24 |
| Intersección Agentes ∩ Gemini | 65 |
| Convergencia triple (Bases ∩ Agentes ∩ Gemini) | 19 |
| Exclusivos en Gemini respecto al resto del corpus | 157 |
| Exclusivos en Agentes respecto al resto del corpus | 36 |
| Exclusivos en Bases API respecto al resto del corpus | 92 |
| **Total corpus consolidado en `02_matriz_corpus.csv`** | **371** |
| **Corpus cribable válido en `cribado_prisma_humano.csv`** | **370** |
| Referencias metodológicas generales separadas vía URI (`referencias_metodologicas_gemini.csv`) | 8 |
| Muestra ciega estratificada para Revisor 2 (`cribado_muestra_ciega_revisor2.csv`, 20 %) | 74 |

## 2. Auditoría Determinística de DOI (Crossref + DataCite)

- **Resueltos en Crossref (artículos/actas):** 336
- **Resueltos en DataCite (preprints arXiv / repositorios Zenodo, revisables para criterio E4):** 34
- **No resueltos (`NO_RESOLVIO`, bloqueados de `cribado_prisma_humano.csv`):** 1

### DOIs descartados por no resolver en Crossref ni DataCite
- `10.29303/jtika.v8i1.518` (origen: Bases (OpenAlex))

## 3. Estudios con mayor convergencia entre vías (muestra superior)

| ID | Vías | Origen | Año | Autor | DOI | Registrador |
| --- | ---: | --- | --- | --- | --- | --- |
| OT-0001 | 3 | Agentes (A1, A2, A3, A4, A5); Gemini (R1, R2, R3, R4, R5, R6); Semilla_SOTA | 2024 | de Vente et al. (2024) | `10.1109/tmi.2023.3313786` | Crossref |
| OT-0002 | 3 | Agentes (A1, A3, A4, A5); Gemini (R1, R2, R3, R4, R5, R6); Semilla_SOTA | 2023 | Hemelings et al. (2023) | `10.1038/s41746-023-00857-0` | Crossref |
| OT-0003 | 3 | Agentes (A1, A2); Gemini (R1, R2, R3, R4, R5, R6); Semilla_SOTA | 2019 | Diaz-Pinto et al. (2019) | `10.1186/s12938-019-0649-y` | Crossref |
| BD-0022 | 3 | Bases (OpenAlex); Agentes (A3, A6); Gemini (R1, R2, R3, R4, R5, R6) | 2026 | Rahman et al. (2026) | `10.48550/arxiv.2611.01992` | DataCite |
| OT-0004 | 3 | Agentes (A1, A2); Gemini (R1, R2, R3, R4); Semilla_SOTA | 2020 | Orlando et al. (2020) | `10.1016/j.media.2019.101570` | Crossref |
| BD-0009 | 3 | Bases (OpenAlex; PubMed; EuropePMC); Agentes (A1, A3, A4, A6); Gemini (R5, R6) | 2026 | van Meyel et al. (2026) | `10.1016/j.ophtha.2026.10.030` | Crossref |
| BD-0001 | 3 | Bases (OpenAlex; PubMed; EuropePMC); Agentes (A1, A3, A4, A6); Gemini (R5, R6) | 2026 | Bolo et al. (2026) | `10.1016/j.xops.2025.101008` | Crossref |
| OT-0005 | 3 | Agentes (A1, A4, A5); Gemini (R1, R4); Semilla_SOTA | 2018 | Li et al. (2018) | `10.1016/j.ophtha.2018.01.023` | Crossref |
| BD-0118 | 3 | Bases (PubMed; EuropePMC); Agentes (A1, A4); Gemini (R1, R4, R6) | 2023 | Fan et al. (2023) | `10.1016/j.xops.2022.100233` | Crossref |
| BD-0002 | 3 | Bases (OpenAlex; PubMed; EuropePMC); Agentes (A1); Gemini (R1, R3, R4, R5) | 2023 | Qian et al. (2023) | `10.1038/s41433-023-02622-9` | Crossref |
| BD-0005 | 3 | Bases (OpenAlex; PubMed; EuropePMC); Agentes (A1, A3, A6); Gemini (R4, R6) | 2026 | Rashidisabet et al. (2026) | `10.1167/tvst.15.9.3` | Crossref |
| BD-0021 | 3 | Bases (OpenAlex; PubMed; EuropePMC); Agentes (A1, A3, A6); Gemini (R5, R6) | 2026 | Alhasson et al. (2026) | `10.3390/jcm15166418` | Crossref |
| BD-0124 | 3 | Bases (PubMed; EuropePMC); Agentes (A1); Gemini (R1, R4, R5) | 2019 | Liu et al. (2019) | `10.1001/jamaophthalmol.2019.3501` | Crossref |
| BD-0053 | 3 | Bases (OpenAlex; PubMed; EuropePMC); Agentes (A1, A6); Gemini (R1, R6) | 2026 | Hou et al. (2026) | `10.1016/j.xops.2025.100923` | Crossref |
| BD-0040 | 3 | Bases (OpenAlex; PubMed; EuropePMC); Agentes (A1); Gemini (R1, R2, R6) | 2025 | Chaurasia et al. (2025) | `10.1038/s41433-024-03388-4` | Crossref |
| BD-0013 | 3 | Bases (OpenAlex; PubMed; EuropePMC); Agentes (A1, A4); Gemini (R1, R5) | 2024 | Jan et al. (2024) | `10.3390/bioengineering11111139` | Crossref |
| BD-0031 | 3 | Bases (OpenAlex; PubMed; EuropePMC); Agentes (A1); Gemini (R1, R4, R5) | 2025 | Ling et al. (2025) | `10.3390/biomedicines13020420` | Crossref |
| BD-0010 | 3 | Bases (OpenAlex; PubMed; EuropePMC); Agentes (A1); Gemini (R1, R5) | 2025 | Milad et al. (2025) | `10.1016/j.xops.2025.100721` | Crossref |
| OT-0006 | 3 | Agentes (A1, A6); Gemini (R6); Semilla_SOTA | 2023 | Zhou et al. (2023) | `10.1038/s41586-023-06555-x` | Crossref |
| OT-0007 | 3 | Agentes (A2); Gemini (R2, R4); Semilla_SOTA | 2022 | Kovalyk et al. (2022) | `10.1038/s41597-022-01388-1` | Crossref |
| BD-0106 | 3 | Bases (PubMed; EuropePMC); Agentes (A1); Gemini (R1, R2) | 2026 | Wu et al. (2026) | `10.1038/s41746-026-02752-w` | Crossref |
| OT-0008 | 3 | Agentes (A4); Gemini (R2, R4); Semilla_SOTA | 2018 | Ahn et al. (2018) | `10.1371/journal.pone.0207982` | Crossref |
| BD-0120 | 3 | Bases (PubMed; EuropePMC); Agentes (A1); Gemini (R5, R6) | 2022 | Ko et al. (2022) | `10.3390/biomedicines10061314` | Crossref |
| BD-0008 | 3 | Bases (OpenAlex; PubMed; EuropePMC); Agentes (A1); Gemini (R1) | 2024 | Madadi et al. (2024) | `10.1016/j.bspc.2024.106061` | Crossref |
| BD-0049 | 3 | Bases (OpenAlex; PubMed; EuropePMC); Agentes (A6); Gemini (R6) | 2024 | Qiu et al. (2024) | `10.1056/aioa2300221` | Crossref |
| BD-0076 | 3 | Bases (OpenAlex); Agentes (A6); Gemini (R2) | 2025 | Li et al. (2025) | `10.1101/2025.04.22.25326210` | Crossref |
| BD-0028 | 3 | Bases (OpenAlex); Agentes (A6); Gemini (R6) | 2026 | Shi et al. (2026) | `10.64898/2026.03.02.26347127` | Crossref |
| OT-0009 | 2 | Agentes (A1, A2); Gemini (R1, R2, R3, R4, R6) | 2020 | Fumero Batista et al. (2020) | `10.5566/ias.2346` | Crossref |
| OT-0010 | 2 | Agentes (A1, A4); Gemini (R1, R3, R4, R5) | 2019 | Phene et al. (2019) | `10.1016/j.ophtha.2019.07.024` | Crossref |
| OT-0011 | 2 | Agentes (A2, A4); Gemini (R1, R2, R4, R6) | 2026 | Madadi et al. (2026) | `10.1109/tmi.2025.3596874` | Crossref |
| OT-0012 | 2 | Agentes (A1, A5); Gemini (R1, R5) | 2017 | Ting et al. (2017) | `10.1001/jama.2017.18152` | Crossref |
| OT-0013 | 2 | Agentes (A3, A5); Gemini (R2, R4) | 2023 | Lemij et al. (2023) | `10.1016/j.xops.2023.100300` | Crossref |
| OT-0014 | 2 | Agentes (A3); Gemini (R3, R5, R6) | 2021 | Hemelings et al. (2021) | `10.1038/s41598-021-99605-1` | Crossref |
| OT-0015 | 2 | Agentes (A2); Gemini (R1, R2, R3) | 2022 | Fang et al. (2022) | `10.48550/arxiv.2202.08994` | DataCite |
| OT-0016 | 2 | Agentes (A4); Gemini (R4, R6) | 2019 | Medeiros et al. (2019) | `10.1016/j.ophtha.2018.12.033` | Crossref |
| OT-0017 | 2 | Agentes (A2); Gemini (R2, R3) | 2023 | Kumar et al. (2023) | `10.1038/s41597-023-01943-4` | Crossref |
| OT-0018 | 2 | Agentes (A4); Gemini (R4, R5) | 2018 | Christopher et al. (2018) | `10.1038/s41598-018-35044-9` | Crossref |
| OT-0019 | 2 | Agentes (A2); Gemini (R2, R4) | 2010 | Zhuo Zhang et al. (2010) | `10.1109/iembs.2010.5626137` | Crossref |
| OT-0020 | 2 | Agentes (A3, A5); Gemini (R5) | 2018 | Zech et al. (2018) | `10.1371/journal.pmed.1002683` | Crossref |
| BD-0034 | 2 | Bases (OpenAlex); Gemini (R1, R2, R4) | 2025 | Remyes et al. (2025) | `10.3390/info16060432` | Crossref |
