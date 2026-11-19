#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador del Slide Deck Oficial y Guión de Exposición (11 Diapositivas):
Evaluación de la generalización de clasificadores profundos de glaucoma
ante heterogeneidad de adquisición de datos en retinografías.

Facultad de Minas — Universidad Nacional de Colombia
"""

import os, sys, base64, subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SLIDES_DIR = BASE_DIR / "slides"
SLIDES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = BASE_DIR / "figures"

HTML_PATH = SLIDES_DIR / "SLIDES_PRESENTACION_GLAUCOMA_DYNA.html"
PDF_PATH = SLIDES_DIR / "SLIDES_PRESENTACION_GLAUCOMA_DYNA.pdf"
SCRIPT_MD_PATH = SLIDES_DIR / "SLIDES_GUIÓN_EXPOSICION.md"

def get_b64(fname):
    p = FIGURES_DIR / fname
    if p.exists():
        ext = p.suffix.lower().replace(".", "")
        mime = "image/png" if ext == "png" else "image/jpeg"
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    print(f"[WARN] No se encontró figura: {fname}")
    return ""

def generate_slides():
    print("[*] 1. Codificando figuras en alta resolución para diapositivas...")
    img_domain_shift = get_b64("fig1_domain_shift_comparison.png")
    img_web_filter = get_b64("fig20_web_images_filtering_pipeline.png")
    img_backbones = get_b64("fig30_backbone_benchmark_multicriteria_tradeoff.png")
    img_confusion_3x3 = get_b64("fig33_confusion_matrices_2_vs_3_classes.png")
    img_data_aug_strip = get_b64("fig_data_augmentation_4_panel_strip.png")
    img_generalization = get_b64("fig21_cross_device_generalization_master.png")

    template = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Presentación Oficial: Generalización de Clasificadores de Glaucoma — UNAL</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css"/>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '$', right: '$', display: false}
    ]});"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

  :root {
    --bg-dark: #0f172a;
    --card-dark: #1e293b;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --accent: #38bdf8;
    --accent-green: #34d399;
    --accent-red: #f87171;
    --accent-gold: #fbbf24;
    --border-color: #334155;
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    font-family: 'Inter', sans-serif;
    background: #000;
    color: var(--text-main);
    overflow: hidden;
    height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .slide-container {
    width: 100vw;
    height: 100vh;
    position: relative;
  }

  .slide {
    display: none;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    padding: 35px 55px;
    background: radial-gradient(circle at 10% 20%, #1e293b 0%, #0f172a 90%);
    opacity: 0;
    transition: opacity 0.3s ease-in-out;
    flex-direction: column;
    justify-content: space-between;
  }

  .slide.active {
    display: flex;
    opacity: 1;
  }

  .slide-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 12px;
  }

  .slide-tag {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--accent);
    background: rgba(56, 189, 248, 0.12);
    padding: 4px 12px;
    border-radius: 999px;
    border: 1px solid rgba(56, 189, 248, 0.25);
  }

  .slide-institution {
    font-size: 0.82rem;
    color: var(--text-muted);
    font-weight: 500;
  }

  .slide-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 16px 0;
  }

  .slide-title {
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 8px;
    background: linear-gradient(to right, #ffffff, #cbd5e1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .slide-subtitle {
    font-size: 1.1rem;
    color: var(--accent);
    margin-bottom: 20px;
    font-weight: 500;
  }

  .grid-2 {
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 32px;
    align-items: center;
    height: 100%;
  }

  .grid-2-eq {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: stretch;
    height: 100%;
  }

  .grid-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 20px;
  }

  .card {
    background: var(--card-dark);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.4);
  }

  .card-highlight {
    background: rgba(56, 189, 248, 0.05);
    border-color: rgba(56, 189, 248, 0.4);
  }

  .card h3 {
    font-size: 1.15rem;
    color: var(--accent);
    margin-bottom: 8px;
    font-weight: 700;
  }

  .card p, .card li {
    font-size: 0.94rem;
    color: #e2e8f0;
    line-height: 1.45;
  }

  ul.bullets {
    list-style: none;
    padding-left: 0;
  }

  ul.bullets li {
    margin-bottom: 10px;
    position: relative;
    padding-left: 22px;
    font-size: 0.96rem;
    line-height: 1.45;
  }

  ul.bullets li::before {
    content: "▹";
    position: absolute;
    left: 0;
    color: var(--accent);
    font-weight: bold;
    font-size: 1.1rem;
    top: -1px;
  }

  .slide-img-box {
    text-align: center;
    background: #000;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 8px;
    overflow: hidden;
  }

  .slide-img-box img {
    max-width: 100%;
    max-height: 450px;
    object-fit: contain;
    border-radius: 6px;
  }

  .stat-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.85rem;
    margin-right: 4px;
  }
  .stat-green { background: rgba(52, 211, 153, 0.2); color: #34d399; border: 1px solid #34d399; }
  .stat-red { background: rgba(248, 113, 113, 0.2); color: #f87171; border: 1px solid #f87171; }
  .stat-gold { background: rgba(251, 191, 36, 0.2); color: #fbbf24; border: 1px solid #fbbf24; }

  .slide-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid var(--border-color);
    padding-top: 12px;
    font-size: 0.85rem;
    color: var(--text-muted);
  }

  .nav-controls {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .nav-btn {
    background: var(--card-dark);
    border: 1px solid var(--border-color);
    color: var(--text-main);
    padding: 5px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
  }

  .slide-title-hero {
    text-align: center;
    padding: 30px 60px;
  }
  .slide-title-hero h1 {
    font-size: 2.5rem;
    font-weight: 800;
    line-height: 1.25;
    margin-bottom: 16px;
    background: linear-gradient(135deg, #ffffff 0%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .slide-title-hero h2 {
    font-size: 1.25rem;
    color: #94a3b8;
    font-weight: 400;
    margin-bottom: 30px;
  }
  .meta-box {
    display: inline-flex;
    gap: 25px;
    background: var(--card-dark);
    padding: 14px 28px;
    border-radius: 12px;
    border: 1px solid var(--border-color);
  }

  @media print {
    body { background: #fff; color: #000; }
    .slide {
      display: flex !important;
      opacity: 1 !important;
      position: relative !important;
      page-break-after: always;
      height: 100vh;
      background: #fff !important;
      color: #000 !important;
    }
    .nav-controls { display: none; }
  }
</style>
</head>
<body>

<div class="slide-container">

  <!-- SLIDE 1: PORTADA -->
  <div class="slide active" id="slide-1">
    <div class="slide-header">
      <span class="slide-tag">Proyecto Final — Deep Learning</span>
      <span class="slide-institution">Facultad de Minas — Universidad Nacional de Colombia</span>
    </div>
    <div class="slide-content slide-title-hero">
      <h1>Evaluación de la generalización de clasificadores profundos de glaucoma ante heterogeneidad de adquisición de datos en retinografías</h1>
      <h2>Un estudio experimental centrado en datos: de la ilusión del AUC mono-centro al control de TFP/TFN, estadificación de tres estadíos y aumento de escala</h2>
      <div class="meta-box">
        <div><strong>Asignatura:</strong> Deep Learning (2026-2)</div>
        <div><strong>Equipo:</strong> Grupo 2</div>
        <div><strong>Formato Editorial:</strong> Revista DYNA</div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Navegación: Flechas ⬅ ➡ o Barra Espaciadora | Tecla 'F' para Pantalla Completa</span>
      <span class="slide-counter">Diapositiva 1 / 11</span>
    </div>
  </div>

  <!-- SLIDE 2: PROBLEMÁTICA CLÍNICA Y DOMAIN SHIFT -->
  <div class="slide" id="slide-2">
    <div class="slide-header">
      <span class="slide-tag">1. Motivación y Problemática</span>
      <span class="slide-institution">El Espejismo de las Métricas Mono-Centro</span>
    </div>
    <div class="slide-content">
      <div class="grid-2">
        <div>
          <h2 class="slide-title">El Colapso del Desempeño por Cambio de Dominio Óptico</h2>
          <p class="slide-subtitle">Modelos con ROC-AUC > 0.98 fracasan al trasladarse a cámaras hospitalarias externas.</p>
          <ul class="bullets">
            <li><strong>El Espejismo Monocéntrico:</strong> Las redes convolucionales aprenden atajos del fondo y calibración del retinógrafo de origen en lugar de biomarcadores biológicos puros.</li>
            <li><strong>Colapso Inter-Cámara Demostrado:</strong> Un modelo entrenado en <em>Canon CR-DGi</em> (ACRIMA) alcanza $0.998$ internamente, pero se desploma a $0.528$ en <em>EyePACS</em> (equivalente al azar).</li>
            <li><strong>Factores Físicos Determinantes:</strong>
              <ul style="margin-left:20px; margin-top:4px;">
                <li>Disparidad angular: $35^\circ$ macro peripapilar vs. $45^\circ$ panorámico macular.</li>
                <li>Variación de temperatura cromática y balance de blancos entre fabricantes.</li>
              </ul>
            </li>
          </ul>
        </div>
        <div class="slide-img-box">
          <img src="@@IMG_DOMAIN_SHIFT@@" alt="Domain Shift en Retinografías"/>
          <p style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">Figura: Variación de encuadre y respuesta espectral inter-fabricante.</p>
        </div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Módulo Introductorio</span>
      <span class="slide-counter">Diapositiva 2 / 11</span>
    </div>
  </div>

  <!-- SLIDE 3: DELIMITACIÓN Y ENFOQUE SANITARIO -->
  <div class="slide" id="slide-3">
    <div class="slide-header">
      <span class="slide-tag">2. Delimitación Metodológica</span>
      <span class="slide-institution">Objetos de Estudio y Enfoque en TFP vs. TFN</span>
    </div>
    <div class="slide-content">
      <h2 class="slide-title">Delimitación Rigurosa y Criterio Clínico Operativo</h2>
      <p class="slide-subtitle">Sustitución del AUC abstracto por el impacto asistencial y deontológico en tamizaje.</p>
      <div class="grid-3" style="margin-top:10px;">
        <div class="card">
          <h3>Objeto Material</h3>
          <p>Clasificadores profundos de visión artificial (CNNs convolucionales y Vision Transformers) aplicados a retinografías digitales de fondo de ojo para detección binaria y estadificación ordinal.</p>
        </div>
        <div class="card">
          <h3>Objeto Formal</h3>
          <p>La capacidad de generalización inter-dispositivo ante disparidad de encuadre ($35^\circ$ vs. $45^\circ$), condiciones de sensor y la eficacia de estrategias centradas en datos (aumento de escala y multifuente).</p>
        </div>
        <div class="card card-highlight">
          <h3>Tradeoff Clínico: TFP vs. TFN</h3>
          <p><span class="stat-badge stat-red">Minimizar TFN</span> <strong>Mandato Clínico:</strong> Evitar omitir pacientes enfermos que progresan a la ceguera irreversible.</p>
          <p style="margin-top:8px;"><span class="stat-badge stat-green">Minimizar TFP</span> <strong>Sostenibilidad:</strong> Evitar saturar consultas de oftalmología especializada con falsas alarmas.</p>
        </div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Fundamentación Metodológica</span>
      <span class="slide-counter">Diapositiva 3 / 11</span>
    </div>
  </div>

  <!-- SLIDE 4: ESTADO DEL ARTE Y SÍNTESIS DE LITERATURA -->
  <div class="slide" id="slide-4">
    <div class="slide-header">
      <span class="slide-tag">3. Estado del Arte</span>
      <span class="slide-institution">Revisión Sistemática de Literatura (PRISMA 2020)</span>
    </div>
    <div class="slide-content">
      <div class="grid-2-eq">
        <div class="card">
          <h3>Rigor Metodológico de la RSL</h3>
          <ul class="bullets">
            <li><strong>Búsqueda programática en 4 APIs científicas:</strong> OpenAlex ($n=150$), Europe PMC ($n=62$), PubMed ($n=54$) y arXiv ($n=29$). Total: 295 brutos, 221 únicos.</li>
            <li><strong>Verificación bibliométrica:</strong> 190 DOIs validados en Crossref API (0 retracciones).</li>
            <li><strong>Concordancia inter-evaluador:</strong> Kappa de Cohen $\kappa = 0.6484$ (acuerdo sustancial con $82.35\%$ observado).</li>
            <li><strong>Calidad metodológica:</strong> Media QA1–QA9 de $8.33/9.00$ sobre 27 variables clínico-técnicas.</li>
          </ul>
        </div>
        <div class="card card-highlight">
          <h3>Hallazgos y Vacíos Críticos en la Literatura</h3>
          <ul class="bullets">
            <li><strong>Caída promedio de transferencia:</strong> Pérdida media inter-dominio de casi 10 puntos de AUC en la literatura (hasta $0.2310$ en pruebas ciegas).</li>
            <li><strong>Factor de confusión primordial:</strong> Disparidad angular del campo visual ($30^\circ-35^\circ$ papilar vs. $45^\circ$ panorámico macular).</li>
            <li><strong>El Vacío Identificado:</strong> Más del $90\%$ de los artículos se limitan a tareas binarias triviales y omiten la estadificación anatómica de severidad.</li>
          </ul>
        </div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Revisión Sistemática de Literatura</span>
      <span class="slide-counter">Diapositiva 4 / 11</span>
    </div>
  </div>

  <!-- SLIDE 5: ECOSISTEMA DE DATOS Y FILTRO WEB -->
  <div class="slide" id="slide-5">
    <div class="slide-header">
      <span class="slide-tag">4. Ecosistema de Datos</span>
      <span class="slide-institution">Cohortes Clínicas e Ingesta Web Inteligente</span>
    </div>
    <div class="slide-content">
      <div class="grid-2">
        <div>
          <h2 class="slide-title">Curaduría de Cohortes y Filtro Web</h2>
          <p class="slide-subtitle">Depuración automática de ruido y diagramas en datos recolectados in-the-wild.</p>
          <ul class="bullets">
            <li><strong>Bases Hospitalarias:</strong> ACRIMA ($n=705$, Canon $35^\circ$), RIM-ONE DL ($n=447$, Kowa 3D), EyePACS ($n=200$, Centervue $45^\circ$) y Moondream ($n=800$, estadificada).</li>
            <li><strong>Adquisición Web (Google Images):</strong> Raspado de retinografías abiertas.</li>
            <li><strong>Discriminador Inteligente de Calidad Retiniana:</strong>
              <ul style="margin-left:20px; margin-top:4px;">
                <li>Verificación de máscara esférica pupilar.</li>
                <li>Perfil fotométrico y absorción de hemoglobina en canal verde.</li>
                <li>Densidad de gradiente papilar (Laplaciano) para purgar diagramas planos.</li>
              </ul>
            </li>
            <li><strong>Eficacia:</strong> <span class="stat-badge stat-green">100% de Precisión</span> en descarte del $47.1\%$ de ruido sin falsos rechazos en ojos reales.</li>
          </ul>
        </div>
        <div class="slide-img-box">
          <img src="@@IMG_WEB_FILTER@@" alt="Filtro Inteligente de Calidad Web"/>
          <p style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">Figura: Pipeline del discriminador de validez de fondo de ojo.</p>
        </div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Preprocesamiento y Calidad</span>
      <span class="slide-counter">Diapositiva 5 / 11</span>
    </div>
  </div>

  <!-- SLIDE 6: BENCHMARK DE BACKBONES Y TRADEOFF TFP/TFN -->
  <div class="slide" id="slide-6">
    <div class="slide-header">
      <span class="slide-tag">5. Selección del Modelo Base</span>
      <span class="slide-institution">Benchmark Multicriterio de 10 Arquitecturas (ImageNet Transfer)</span>
    </div>
    <div class="slide-content">
      <div class="grid-2">
        <div>
          <h2 class="slide-title">Benchmark y Selección del Backbone</h2>
          <p class="slide-subtitle">Evaluación de CNNs vs. Vision Transformers bajo latencia y TFP/TFN.</p>
          <ul class="bullets">
            <li><strong>ConvNeXt-Tiny (Ganador Absoluto):</strong>
              <br/><span class="stat-badge stat-green">AUC 0.9686</span> <span class="stat-badge stat-green">TFN 6.13%</span> <span class="stat-badge stat-gold">Latencia 8.58 ms</span>
              <br/>Kernel $7\times 7$, retiene sesgo inductivo 2D y lidera en sensibilidad clínica.
            </li>
            <li><strong>ResNet-18 (Eficiencia Máxima en el Borde):</strong>
              <br/><span class="stat-badge stat-green">AUC 0.9569</span> <span class="stat-badge stat-gold">Latencia 5.86 ms</span> <span class="stat-badge stat-gold">11.18 M Params</span>
              <br/>Columna vertebral ideal para despliegues portátiles y bajo consumo.
            </li>
            <li><strong>Colapso de ViT-B/16 (Plain Transformer):</strong>
              <br/><span class="stat-badge stat-red">AUC 0.9022</span> <span class="stat-badge stat-red">TFN 17.18%</span> <span class="stat-badge stat-red">24.60 ms</span>
              <br/>Al carecer de sesgo local, sufre severamente en conjuntos biomédicos finitos ($N \approx 2,000$).</li>
          </ul>
        </div>
        <div class="slide-img-box">
          <img src="@@IMG_BACKBONES@@" alt="Tradeoff Multicriterio de Backbones"/>
          <p style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">Figura: Tradeoff Latencia vs. ROC-AUC vs. Parámetros.</p>
        </div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Transfer Learning y Backbones</span>
      <span class="slide-counter">Diapositiva 6 / 11</span>
    </div>
  </div>

  <!-- SLIDE 7: PARADIGMAS DE DECISIÓN: 2 VS 3 CLASES -->
  <div class="slide" id="slide-7">
    <div class="slide-header">
      <span class="slide-tag">6. Paradigmas de Clasificación</span>
      <span class="slide-institution">Detección Binaria vs. Estadificación Trinaria Ordinal</span>
    </div>
    <div class="slide-content">
      <div class="grid-2">
        <div>
          <h2 class="slide-title">¿Por qué la Clasificación Binaria es Insuficiente?</h2>
          <p class="slide-subtitle">El glaucoma no es dicotómico; es una neuropatía progresiva continua.</p>
          <ul class="bullets">
            <li><strong>Falso Confort Binario:</strong> El modelo binario reporta $86\%$ de exactitud, pero su TFN en glaucoma temprano es inaceptable ($12\%$) al concentrarse en excavaciones terminales.</li>
            <li><strong>Formulación Trinaria (Sano, Incipiente, Avanzado):</strong>
              Permite descomponer y penalizar los errores según su severidad:
            </li>
            <li><span class="stat-badge stat-gold">Errores Adyacentes (Leves)</span> <strong>Sano ↔ Incipiente:</strong> Incertidumbre diagnóstica tolerable entre especialistas.</li>
            <li><span class="stat-badge stat-red">Errores Catastróficos (Severos)</span> <strong>Sano ↔ Avanzado:</strong> Omisión crítica de un ojo severamente dañado.</li>
          </ul>
        </div>
        <div class="slide-img-box">
          <img src="@@IMG_CONFUSION_3X3@@" alt="Matrices de Confusión 2 vs 3 clases"/>
          <p style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">Figura: Matriz 3x3 que desvela la distribución real de errores.</p>
        </div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Estadificación Clínica</span>
      <span class="slide-counter">Diapositiva 7 / 11</span>
    </div>
  </div>

  <!-- SLIDE 8: DATA AUGMENTATION MALIGNO -->
  <div class="slide" id="slide-8">
    <div class="slide-header">
      <span class="slide-tag">7. Estrategias Centradas en Datos</span>
      <span class="slide-institution">Malignant Scale Augmentation e Invarianza de Escala</span>
    </div>
    <div class="slide-content">
      <div style="text-align:center; margin-bottom:10px;">
        <h2 class="slide-title" style="font-size:1.8rem; margin-bottom:2px;">Malignant Scale Augmentation: Rompiendo el Atajo de Escala</h2>
        <p class="slide-subtitle" style="margin-bottom:10px;">Operador estocástico para forzar atención biológica y resistencia inter-cámara.</p>
      </div>
      <div class="slide-img-box" style="margin-bottom:14px;">
        <img src="@@IMG_DATA_AUG_STRIP@@" alt="Tira de 4 paneles Data Augmentation" style="max-height:200px;"/>
      </div>
      <div class="grid-3">
        <div class="card">
          <h3>Zoom-In Papilar (1.50x)</h3>
          <p>Forza a los filtros convolucionales a enfocarse en la regla ISNT y muescas polares sin depender de la mácula.</p>
        </div>
        <div class="card">
          <h3>Zoom-Out Severo (0.45x)</h3>
          <p>Incrusta la papila en lienzo sintético coroideo con máscara pupilar, induciendo resistencia ante cámaras panorámicas ($45^\circ$).</p>
        </div>
        <div class="card card-highlight">
          <h3>Impacto Cuantitativo</h3>
          <p><span class="stat-badge stat-green">Errores Catastróficos: 20% ➔ 9.76%</span> (Reducción de más del $51\%$).</p>
          <p style="margin-top:4px;"><span class="stat-badge stat-green">Acuerdo Cuadrático: QWK = 0.8163</span> (Concordancia casi perfecta).</p>
        </div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Data-Centric AI</span>
      <span class="slide-counter">Diapositiva 8 / 11</span>
    </div>
  </div>

  <!-- SLIDE 9: MATRIZ DE GENERALIZACIÓN Y ANÁLISIS FORENSE -->
  <div class="slide" id="slide-9">
    <div class="slide-header">
      <span class="slide-tag">8. Resultados de Generalización</span>
      <span class="slide-institution">Matriz Cruzada Inter-Dispositivo y Federación Multi-Sensor</span>
    </div>
    <div class="slide-content">
      <div class="grid-2">
        <div>
          <h2 class="slide-title">Superando el Colapso de Generalización</h2>
          <p class="slide-subtitle">La combinación de entrenamiento multifuente y aumento de escala neutraliza el domain shift.</p>
          <ul class="bullets">
            <li><strong>Entrenamiento Monomodal (ACRIMA):</strong> Se desploma a $0.5283$ en EyePACS (ciego a imágenes $45^\circ$).</li>
            <li><strong>Entrenamiento Bi-céntrico (ACRIMA + RIM-ONE):</strong> Recupera la varianza espectral ($0.7400$ en EyePACS).</li>
            <li><strong>Mega-Cohorte Federada + Malignant:</strong>
              <br/><span class="stat-badge stat-green">ACRIMA: AUC 0.9950</span>
              <br/><span class="stat-badge stat-green">RIM-ONE: AUC 0.9710</span>
              <br/><span class="stat-badge stat-green">EyePACS: AUC 0.9400 (Especificidad: 100%)</span>
            </li>
            <li><strong>Conclusión:</strong> El aumento de datos bien fundamentado cierra la brecha geométrica sin necesidad de complejas redes adversarias (GANs).</li>
          </ul>
        </div>
        <div class="slide-img-box">
          <img src="@@IMG_GENERALIZATION@@" alt="Matriz Maestra de Generalización"/>
          <p style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">Figura: Matriz de transferencia cruzada y retención inter-dispositivo.</p>
        </div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Evaluación Inter-Cámara</span>
      <span class="slide-counter">Diapositiva 9 / 11</span>
    </div>
  </div>

  <!-- SLIDE 10: DEMOSTRACIÓN EN VIVO (LIVE DEMO) -->
  <div class="slide" id="slide-10">
    <div class="slide-header">
      <span class="slide-tag">9. Validación Experimental</span>
      <span class="slide-institution">Demostración en Tiempo Real de Clasificación</span>
    </div>
    <div class="slide-content" style="text-align:center;">
      <div style="max-width:800px; margin:0 auto;">
        <h2 class="slide-title" style="font-size:2.8rem; margin-bottom:12px;">DEMOSTRACIÓN EN VIVO</h2>
        <p class="slide-subtitle" style="font-size:1.3rem;">Inferencia en Tiempo Real y Diagnóstico de Glaucoma</p>
        
        <div class="card card-highlight" style="padding:28px; margin-top:16px; text-align:left;">
          <h3 style="font-size:1.35rem; color:var(--accent); margin-bottom:12px;">Protocolo de Prueba en Vivo ante el Jurado:</h3>
          <ul class="bullets" style="font-size:1.02rem;">
            <li><strong>Carga interactiva de retinografías:</strong> Prueba sobre imágenes de diferentes fuentes (hospitalarias $35^\circ$, cribado $45^\circ$ e imágenes in-the-wild de la web).</li>
            <li><strong>Inferencia instantánea (< 15 ms en GPU / < 50 ms en CPU):</strong> Ejecución con los pesos congelados de <code>best_fine_tuned_convnext_tiny_glaucoma.pth</code>.</li>
            <li><strong>Resultados desplegados en pantalla:</strong>
              <div style="margin-top:10px; display:flex; gap:10px; flex-wrap:wrap;">
                <span class="stat-badge stat-green">Predicción Binaria (Normal vs. Glaucoma)</span>
                <span class="stat-badge stat-gold">Estadificación (Sano / Incipiente / Avanzado)</span>
                <span class="stat-badge stat-red">Probabilidad Softmax & Alerta de Calidad</span>
              </div>
            </li>
          </ul>
        </div>
        <p style="margin-top:20px; font-style:italic; color:#94a3b8; font-size:1.05rem;">
          [ Transición para ejecutar la Live Demo interactiva ]
        </p>
      </div>
    </div>
    <div class="slide-footer">
      <span>Momento Cumbre de la Sustentación</span>
      <span class="slide-counter">Diapositiva 10 / 11</span>
    </div>
  </div>

  <!-- SLIDE 11: CONCLUSIONES Y CIERRE -->
  <div class="slide" id="slide-11">
    <div class="slide-header">
      <span class="slide-tag">10. Conclusiones</span>
      <span class="slide-institution">Recomendaciones de Ingeniería Clínica y Cierre</span>
    </div>
    <div class="slide-content">
      <h2 class="slide-title">Conclusiones y Recomendaciones de Ingeniería</h2>
      <p class="slide-subtitle">Lecciones metodológicas para futuros sistemas de cribado asistencial.</p>
      <div class="grid-2-eq" style="margin-top:10px;">
        <div class="card">
          <h3>Hallazgos Metodológicos Clave</h3>
          <ul class="bullets">
            <li><strong>El espejismo del AUC mono-centro:</strong> Las métricas en un único hospital no son transferibles; la disparidad angular ($35^\circ$ vs $45^\circ$) es el mayor destructor de rendimiento.</li>
            <li><strong>ResNet y ConvNeXt superan a ViT:</strong> En regímenes biomédicos de tamaño moderado ($N \approx 2,000$), el sesgo inductivo convolucional es superior a la atención irrestricta.</li>
          </ul>
        </div>
        <div class="card card-highlight">
          <h3>Aportes en Ingeniería de Datos</h3>
          <ul class="bullets">
            <li><strong>La estadificación de 3 clases es indispensable:</strong> Tratar el glaucoma como variable binaria oculta el daño inicial; el esquema trinario modela la incertidumbre y reduce a la mitad los fallos graves.</li>
            <li><strong>Data-Centric AI sobre complejidad de modelos:</strong> El aumento estocástico de escala (<em>Malignant Scale</em>) resolvió la brecha panorámica sin necesidad de costosas arquitecturas adversarias.</li>
          </ul>
        </div>
      </div>
      <div style="text-align:center; margin-top:20px;">
        <p style="font-size:1.3rem; font-weight:700; color:var(--accent);">¡Muchas gracias por su atención!</p>
        <p style="font-size:0.95rem; color:var(--text-muted); margin-top:4px;">Espacio abierto para preguntas y discusión académica.</p>
      </div>
    </div>
    <div class="slide-footer">
      <span>Cierre y Preguntas (Q&A)</span>
      <span class="slide-counter">Diapositiva 11 / 11</span>
    </div>
  </div>

</div>

<script>
  let currentSlide = 1;
  const totalSlides = 11;

  function showSlide(n) {
    const slides = document.querySelectorAll('.slide');
    if (n > totalSlides) currentSlide = 1;
    if (n < 1) currentSlide = totalSlides;

    slides.forEach((s, idx) => {
      s.classList.remove('active');
      if (idx === currentSlide - 1) {
        s.classList.add('active');
      }
    });
  }

  function nextSlide() {
    currentSlide++;
    showSlide(currentSlide);
  }

  function prevSlide() {
    currentSlide--;
    showSlide(currentSlide);
  }

  document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
      nextSlide();
    } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
      prevSlide();
    } else if (e.key === 'f' || e.key === 'F') {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
      } else {
        document.exitFullscreen();
      }
    }
  });
</script>
</body>
</html>
"""

    # Reemplazar imágenes
    rendered_html = template.replace("@@IMG_DOMAIN_SHIFT@@", img_domain_shift) \
                            .replace("@@IMG_WEB_FILTER@@", img_web_filter) \
                            .replace("@@IMG_BACKBONES@@", img_backbones) \
                            .replace("@@IMG_CONFUSION_3X3@@", img_confusion_3x3) \
                            .replace("@@IMG_DATA_AUG_STRIP@@", img_data_aug_strip) \
                            .replace("@@IMG_GENERALIZATION@@", img_generalization)

    print(f"[*] 2. Guardando presentación interactiva en: {HTML_PATH.name}")
    HTML_PATH.write_text(rendered_html, encoding="utf-8")

    # Guión de Sustentación
    script_md = """# Guión Oficial de Exposición (11 Diapositivas / 12–15 Minutos)
## Evaluación de la generalización de clasificadores profundos de glaucoma ante heterogeneidad de adquisición de datos en retinografías

**Materia:** Deep Learning — Facultad de Minas, Universidad Nacional de Colombia  
**Tiempo estimado total:** 12 a 14 minutos + 3 a 5 minutos de preguntas.  

---

### Diapositiva 1: Portada y Presentación del Proyecto (1:00 min)
* **Qué decir:**
  * "Buenas tardes profesor y compañeros. Hoy les presentamos nuestra investigación titulada: *Evaluación de la generalización de clasificadores profundos de glaucoma ante heterogeneidad de adquisición de datos en retinografías*."
  * "El objetivo central de este proyecto no fue simplemente entrenar un modelo para sacar un 98% en un dataset cerrado, sino responder a la pregunta que la literatura suele evadir: ¿qué pasa cuando ese modelo se lleva a un hospital real con otra cámara, otra calibración de color o un ángulo de visión distinto?"

---

### Diapositiva 2: La Problemática Clínica y el Espejismo Monocéntrico (1:15 min)
* **Qué decir:**
  * "El glaucoma es la principal causa de ceguera irreversible en el mundo y es una enfermedad silenciosa: cuando el paciente nota pérdida de visión, ya ha perdido más del 40% de sus fibras nerviosas."
  * "Al entrenar redes convolucionales en bases de datos públicas como ACRIMA, los modelos reportan métricas excepcionales, pero descubrimos que esto es un **espejismo monocéntrico**."
  * "Cuando evaluamos ese mismo modelo en una cohorte externa como EyePACS, el AUC colapsa a 0.528, que es literalmente tirar una moneda al aire. ¿La razón? El cambio de dominio óptico: cámaras de sobremesa con recorte focal a 35° frente a cámaras panorámicas de campo abierto a 45°."

---

### Diapositiva 3: Delimitación Metodológica y Enfoque en TFP vs. TFN (1:15 min)
* **Qué decir:**
  * "Para abordar esto con rigor científico, delimitamos nuestro objeto material y formal."
  * "Y adoptamos un mandato clínico fundamental: en salud pública, el ROC-AUC general es engañoso. Lo que verdaderamente importa es el balance entre la **Tasa de Falsos Negativos (TFN)** y la **Tasa de Falsos Positivos (TFP)**."
  * "Minimizar la TFN es prioritario para no enviar a casa a un paciente enfermo que terminará ciego. Minimizar la TFP es indispensable para no colapsar el sistema de salud con falsas alarmas. En este estudio evaluamos ambos frentes."

---

### Diapositiva 4: Estado del Arte y Revisión Sistemática PRISMA 2020 (1:15 min)
* **Qué decir:**
  * "Construimos un estado del arte formal mediante una Revisión Sistemática de Literatura bajo directrices PRISMA 2020 y Kitchenham."
  * "Buscamos de manera programática en 4 APIs científicas (OpenAlex, Europe PMC, PubMed y arXiv), validamos 190 artículos en Crossref y logramos un acuerdo sustancial de Kappa inter-evaluador de 0.648."
  * "La literatura reveló dos vacíos mayúsculos: una pérdida promedio inter-dominio de casi 10 puntos de AUC, y el hecho de que más del 90% de los estudios tratan el glaucoma como una variable binaria trivial, ignorando los estadíos tempranos."

---

### Diapositiva 5: Ecosistema de Datos y Filtro Web Inteligente (1:15 min)
* **Qué decir:**
  * "Consolidamos cuatro cohortes internacionales (ACRIMA de España, RIM-ONE DL de Canarias con fotos estereoscópicas, EyePACS de Estados Unidos y Moondream con anotación ordinal)."
  * "Además, exploramos la recolección de imágenes web in-the-wild desde Google Images. Como internet está lleno de diagramas y esquemas anatómicos no utilizables, diseñamos un discriminador inteligente de calidad de fondo de ojo basado en la firma fotométrica del canal verde y gradientes peripapilares."
  * "El filtro depuró el 47% de ruido con un 100% de precisión, demostrando cómo preprocesar datos abiertos con rigor."

---

### Diapositiva 6: Benchmark de Backbones y Tradeoff TFP vs. TFN (1:15 min)
* **Qué decir:**
  * "Evaluamos diez arquitecturas representativas con Transfer Learning desde ImageNet."
  * "El modelo ganador fue **ConvNeXt-Tiny**, logrando un AUC de 0.9686 y la menor tasa de falsos negativos del benchmark (TFN = 6.13%)."
  * "Por su parte, **ResNet-18** demostró ser la red más rápida y eficiente (menos de 6 milisegundos de inferencia)."
  * "Un hallazgo teórico clave: los Vision Transformers puros como ViT-B/16 sufrieron severamente en este tamaño de datos (AUC 0.9022 y TFN de 17%), confirmando que el sesgo inductivo 2D de las convoluciones es superior cuando no se cuenta con millones de imágenes médicas."

---

### Diapositiva 7: Paradigmas de Decisión: 2 vs. 3 Clases (1:15 min)
* **Qué decir:**
  * "Demostramos que clasificar glaucoma como 'Sano vs. Enfermo' es un error metodológico: la red aprende a buscar ojos con excavaciones gigantescas y falla en los casos incipientes."
  * "Implementamos una estadificación en tres clases: Sano, Incipiente y Avanzado."
  * "Esto nos permitió categorizar los fallos: los errores adyacentes (Sano confundido con Incipiente) representan la incertidumbre humana normal; pero los **errores catastróficos** (un paciente avanzado catalogado como sano) son inaceptables."

---

### Diapositiva 8: Data Augmentation Maligno e Invarianza de Escala (1:30 min)
* **Qué decir:**
  * "Para resolver el colapso geométrico entre cámaras de 35° y 45°, diseñamos *Malignant Scale Augmentation*."
  * "En la tira de imágenes pueden ver la intervención: combinamos Zoom-In focal para obligar a la red a mirar la regla ISNT en la copa, Zoom-Out severo incrustando la papila en un lienzo sintético con máscara pupilar para simular fotos de campo completo, y perturbaciones fotométricas para emular diferencias de sensor."
  * "El resultado fue contundente: **redujimos los errores catastróficos a más de la mitad (del 20% al 9.76%)** y logramos un acuerdo ponderado casi perfecto de QWK = 0.8163."

---

### Diapositiva 9: Matriz de Generalización y Validación Cruzada (1:15 min)
* **Qué decir:**
  * "Al integrar el aumento de escala y la mega-cohorte federada, rompimos el colapso inter-cámara."
  * "El rendimiento en EyePACS —que inicialmente era de 0.528— escaló hasta un AUC de 0.9400 con un 100% de especificidad."
  * "Esto demuestra que con una estrategia centrada en los datos no hace falta recurrir a arquitecturas adversarias complejas ni a modelos pesados para lograr una generalización robusta."

---

### Diapositiva 10: DEMOSTRACIÓN EN VIVO (2:00 - 2:30 min)
* **Qué decir y hacer:**
  * "Como la mejor prueba de robustez es la verificación en tiempo real, pasamos ahora a la demostración en vivo de nuestro clasificador."
  * *(Hacer la transición a la Live Demo: cargar una retinografía, mostrar la predicción instantánea binaria y de estadificación de 3 clases en pantalla con sus probabilidades Softmax).*

---

### Diapositiva 11: Conclusiones y Cierre (1:00 min)
* **Qué decir:**
  * "En conclusión: 1) Los modelos binarios tradicionales son frágiles en entornos reales; 2) Las CNNs modernas como ConvNeXt y ResNet siguen siendo superiores a los Transformers en volúmenes biomédicos moderados; 3) La estadificación de 3 clases mitiga los fallos críticos de tamizaje; y 4) Las intervenciones centradas en los datos son la clave para un despliegue clínico fiable."
  * "Muchas gracias. Quedamos atentos a sus preguntas y comentarios."
"""
    print(f"[*] 3. Guardando guión de sustentación en: {SCRIPT_MD_PATH.name}")
    SCRIPT_MD_PATH.write_text(script_md, encoding="utf-8")

    # Compilar a PDF vía Chromium
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    ]
    browser = None
    for c in candidates:
        if Path(c).exists():
            browser = c
            break

    if browser:
        print(f"[*] 4. Compilando PDF de diapositivas vía {Path(browser).name}...")
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            f"--print-to-pdf={PDF_PATH}",
            str(HTML_PATH)
        ]
        subprocess.run(cmd, check=True)
        if PDF_PATH.exists():
            print(f"[EXITO] Diapositivas PDF compiladas: {PDF_PATH.name} ({PDF_PATH.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    generate_slides()
