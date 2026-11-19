#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de Presentación Oficial — Formato Híbrido Facultad de Minas (UNAL)
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

def generate_hybrid_slides():
    print("[*] 1. Codificando figuras en alta resolución para diapositivas...")
    img_domain_shift = get_b64("fig1_domain_shift_comparison.png")
    img_backbones = get_b64("fig30_backbone_benchmark_multicriteria_tradeoff.png")
    img_confusion_3x3 = get_b64("fig33_confusion_matrices_2_vs_3_classes.png")
    img_data_aug_strip = get_b64("fig_data_augmentation_4_panel_strip.png")
    img_generalization = get_b64("fig21_cross_device_generalization_master.png")

    template = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Evaluación de la Generalización de Clasificadores de Glaucoma — Facultad de Minas UNAL</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css"/>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '$', right: '$', display: false}
    ]});"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Merriweather:ital,wght@0,300;0,400;0,700;1,300&display=swap');

  :root {
    --verde-minas-deep: #004D25;
    --verde-minas-medio: #165B33;
    --verde-minas-claro: #2E7D47;
    --dorado-unal: #D4A017;
    --dorado-suave: #F3E5AB;
    --gris-fondo: #F8FAF8;
    --gris-borde: #E2E8E2;
    --texto-oscuro: #1E293B;
    --texto-tenue: #64748B;
    --rojo-alerta: #DC2626;
    --verde-exito: #16A34A;
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    font-family: 'Montserrat', sans-serif;
    background: #111;
    color: var(--texto-oscuro);
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
    padding: 30px 50px;
    opacity: 0;
    transition: opacity 0.3s ease-in-out;
    flex-direction: column;
    justify-content: space-between;
  }

  .slide.active {
    display: flex;
    opacity: 1;
  }

  /* ESTILO A: SLIDES VERDES INSTITUCIONALES (Portada, Live Demo, Cierre) */
  .slide-minas-verde {
    background: radial-gradient(circle at 20% 20%, #165B33 0%, #004D25 70%, #003319 100%);
    color: #FFFFFF;
  }

  .slide-minas-verde .slide-header {
    border-bottom: 1.5px solid rgba(212, 160, 23, 0.4);
  }

  .slide-minas-verde .slide-tag {
    background: rgba(212, 160, 23, 0.18);
    color: var(--dorado-suave);
    border: 1px solid var(--dorado-unal);
  }

  .slide-minas-verde .slide-institution {
    color: #E2E8F0;
  }

  .slide-minas-verde .slide-footer {
    border-top: 1.5px solid rgba(212, 160, 23, 0.4);
    color: #CBD5E1;
  }

  /* ESTILO B: SLIDES CLARAS TÉCNICAS (Contenido analítico, figuras y tablas) */
  .slide-minas-clara {
    background: var(--gris-fondo);
    color: var(--texto-oscuro);
  }

  .slide-minas-clara .slide-header {
    border-bottom: 2.5px solid var(--verde-minas-deep);
  }

  .slide-minas-clara .slide-tag {
    background: rgba(0, 77, 37, 0.1);
    color: var(--verde-minas-deep);
    border: 1px solid var(--verde-minas-deep);
  }

  .slide-minas-clara .slide-institution {
    color: var(--texto-tenue);
    font-weight: 600;
  }

  .slide-minas-clara .slide-footer {
    border-top: 1.5px solid var(--gris-borde);
    color: var(--texto-tenue);
  }

  /* Componentes Comunes */
  .slide-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
  }

  .slide-tag {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 4px 14px;
    border-radius: 999px;
  }

  .slide-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 14px 0;
  }

  .slide-title-clara {
    font-size: 2.05rem;
    font-weight: 800;
    color: var(--verde-minas-deep);
    line-height: 1.2;
    margin-bottom: 4px;
  }

  .slide-subtitle-clara {
    font-size: 1.05rem;
    color: var(--verde-minas-claro);
    font-weight: 600;
    margin-bottom: 18px;
  }

  .slide-title-verde {
    font-size: 2.5rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.25;
    margin-bottom: 14px;
  }

  .slide-subtitle-verde {
    font-size: 1.2rem;
    color: var(--dorado-suave);
    font-weight: 500;
    margin-bottom: 24px;
    line-height: 1.4;
  }

  /* Grids */
  .grid-2 {
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 30px;
    align-items: center;
    height: 100%;
  }

  .grid-2-eq {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: stretch;
  }

  .grid-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 20px;
  }

  /* Tarjetas */
  .card-clara {
    background: #FFFFFF;
    border: 1px solid var(--gris-borde);
    border-top: 4px solid var(--verde-minas-deep);
    border-radius: 8px;
    padding: 18px;
    box-shadow: 0 4px 12px rgba(0, 77, 37, 0.06);
  }

  .card-clara h3 {
    font-size: 1.15rem;
    color: var(--verde-minas-deep);
    margin-bottom: 8px;
    font-weight: 700;
  }

  .card-clara p, .card-clara li {
    font-size: 0.92rem;
    color: var(--texto-oscuro);
    line-height: 1.45;
  }

  .card-verde {
    background: rgba(0, 51, 25, 0.45);
    border: 1px solid rgba(212, 160, 23, 0.35);
    border-radius: 10px;
    padding: 22px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
  }

  .card-verde h3 {
    font-size: 1.2rem;
    color: var(--dorado-unal);
    margin-bottom: 10px;
    font-weight: 700;
  }

  .card-verde p, .card-verde li {
    font-size: 0.95rem;
    color: #F8FAFC;
    line-height: 1.5;
  }

  ul.bullets-clara {
    list-style: none;
    padding-left: 0;
  }

  ul.bullets-clara li {
    margin-bottom: 10px;
    position: relative;
    padding-left: 20px;
    font-size: 0.94rem;
    line-height: 1.45;
  }

  ul.bullets-clara li::before {
    content: "■";
    position: absolute;
    left: 0;
    color: var(--verde-minas-claro);
    font-size: 0.8rem;
    top: 2px;
  }

  ul.bullets-verde {
    list-style: none;
    padding-left: 0;
  }

  ul.bullets-verde li {
    margin-bottom: 12px;
    position: relative;
    padding-left: 22px;
    font-size: 1.0rem;
    line-height: 1.5;
  }

  ul.bullets-verde li::before {
    content: "◆";
    position: absolute;
    left: 0;
    color: var(--dorado-unal);
    font-size: 0.9rem;
    top: 1px;
  }

  .slide-img-box-clara {
    text-align: center;
    background: #FFFFFF;
    border: 1px solid var(--gris-borde);
    border-radius: 8px;
    padding: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
  }

  .slide-img-box-clara img {
    max-width: 100%;
    max-height: 440px;
    object-fit: contain;
    border-radius: 4px;
  }

  .badge-clara {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.84rem;
    margin-right: 4px;
  }
  .badge-green { background: #DCFCE7; color: #15803D; border: 1px solid #86EFAC; }
  .badge-red { background: #FEE2E2; color: #B91C1C; border: 1px solid #FCA5A5; }
  .badge-gold { background: #FEF3C7; color: #B45309; border: 1px solid #FCD34D; }

  .slide-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 12px;
    font-size: 0.82rem;
  }

  .meta-tag-minas {
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(212, 160, 23, 0.5);
    padding: 10px 24px;
    border-radius: 8px;
    display: inline-flex;
    gap: 24px;
    font-size: 0.95rem;
  }

  @media print {
    body { background: #fff; }
    .slide {
      display: flex !important;
      opacity: 1 !important;
      position: relative !important;
      page-break-after: always;
      height: 100vh;
    }
  }
</style>
</head>
<body>

<div class="slide-container">

  <!-- SLIDE 1: PORTADA [FONDO VERDE MINAS OFICIAL] -->
  <div class="slide slide-minas-verde active" id="slide-1">
    <div class="slide-header">
      <span class="slide-tag">Proyecto Final — Deep Learning</span>
      <span class="slide-institution">Facultad de Minas — Universidad Nacional de Colombia</span>
    </div>
    <div class="slide-content" style="text-align: center; padding: 20px 40px;">
      <h1 class="slide-title-verde">Evaluación de la generalización de clasificadores profundos de glaucoma ante heterogeneidad de adquisición de datos en retinografías</h1>
      <p class="slide-subtitle-verde">De la ilusión del AUC mono-centro al control de TFP/TFN, estadificación en tres niveles y aumento de escala</p>
      <div class="meta-tag-minas">
        <div><strong>Asignatura:</strong> Deep Learning (2026-2)</div>
        <div><strong>Equipo:</strong> Grupo 2</div>
        <div><strong>Formato de Publicación:</strong> Revista DYNA</div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Navegación: Flechas ⬅ ➡ o Barra Espaciadora | Tecla 'F' para Pantalla Completa</span>
      <span>Diapositiva 1 / 11</span>
    </div>
  </div>

  <!-- SLIDE 2: PROBLEMÁTICA Y DOMAIN SHIFT [FONDO CLARO] -->
  <div class="slide slide-minas-clara" id="slide-2">
    <div class="slide-header">
      <span class="slide-tag">1. Motivación y Problemática</span>
      <span class="slide-institution">Facultad de Minas — Universidad Nacional de Colombia</span>
    </div>
    <div class="slide-content">
      <div class="grid-2">
        <div>
          <h2 class="slide-title-clara">El Espejismo de las Métricas Monocéntricas</h2>
          <p class="slide-subtitle-clara">Modelos con ROC-AUC > 0.98 colapsan a nivel aleatorio al cambiar de cámara.</p>
          <ul class="bullets-clara">
            <li><strong>La Trampa del Entorno Cerrado:</strong> En retinografía, las redes convolucionales memorizan fácilmente firmas de fondo y balance de blancos del sensor de origen en vez de patología pura.</li>
            <li><strong>Desplome Crítico Demostrado:</strong> Un clasificador entrenado en <em>Canon CR-DGi</em> (ACRIMA) logra un AUC casi perfecto ($0.998$), pero se derrumba a <strong>$0.528$ en EyePACS</strong> (puro azar).</li>
            <li><strong>Causas Físicas del Domain Shift:</strong>
              <ul style="margin-left: 20px; margin-top: 4px;">
                <li>Disparidad de encuadre: $35^\circ$ macro peripapilar vs. $45^\circ$ panorámico macular.</li>
                <li>Diferencias espectrales y calibración de color entre fabricantes.</li>
              </ul>
            </li>
          </ul>
        </div>
        <div class="slide-img-box-clara">
          <img src="@@IMG_DOMAIN_SHIFT@@" alt="Domain Shift en Retinografías"/>
          <p style="font-size:0.78rem; color:var(--texto-tenue); margin-top:4px;">Figura: Variación geométrica de encuadre y respuesta cromática.</p>
        </div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Contexto Clínico y Justificación</span>
      <span>Diapositiva 2 / 11</span>
    </div>
  </div>

  <!-- SLIDE 3: DELIMITACIÓN Y CRITERIO CLÍNICO TFP/TFN [FONDO CLARO] -->
  <div class="slide slide-minas-clara" id="slide-3">
    <div class="slide-header">
      <span class="slide-tag">2. Delimitación Metodológica</span>
      <span class="slide-institution">Facultad de Minas — Universidad Nacional de Colombia</span>
    </div>
    <div class="slide-content">
      <h2 class="slide-title-clara">Delimitación Rigurosa y Criterio Clínico Operativo</h2>
      <p class="slide-subtitle-clara">Por qué en medicina la Tasa de Falsos Negativos y Positivos supera al ROC-AUC abstracto.</p>
      <div class="grid-3" style="margin-top: 8px;">
        <div class="card-clara">
          <h3>Objeto Material</h3>
          <p>Clasificadores profundos de visión artificial (arquitecturas convolucionales CNNs y Vision Transformers) aplicados a retinografías digitales de fondo de ojo para detección binaria y estadificación ordinal.</p>
        </div>
        <div class="card-clara">
          <h3>Objeto Formal</h3>
          <p>La evaluación sistemática de la pérdida de generalización inter-dispositivo ante variaciones de encuadre ($35^\circ$ vs $45^\circ$), condiciones de sensor y la eficacia de estrategias centradas en datos.</p>
        </div>
        <div class="card-clara" style="border-top-color: var(--dorado-unal);">
          <h3>Tradeoff Clínico: TFP vs. TFN</h3>
          <p><span class="badge-clara badge-red">Minimizar TFN</span> <strong>Mandato Clínico:</strong> Evitar omitir pacientes enfermos que progresan asintomáticamente hacia la ceguera irreversible.</p>
          <p style="margin-top: 8px;"><span class="badge-clara badge-green">Minimizar TFP</span> <strong>Sostenibilidad:</strong> Evitar colapsar la atención especializada de segundo y tercer nivel con falsas alarmas.</p>
        </div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Enfoque Sanitario y Deontológico</span>
      <span>Diapositiva 3 / 11</span>
    </div>
  </div>

  <!-- SLIDE 4: ESTADO DEL ARTE Y VACÍOS [FONDO CLARO] -->
  <div class="slide slide-minas-clara" id="slide-4">
    <div class="slide-header">
      <span class="slide-tag">3. Estado del Arte</span>
      <span class="slide-institution">Facultad de Minas — Universidad Nacional de Colombia</span>
    </div>
    <div class="slide-content">
      <div class="grid-2-eq">
        <div class="card-clara">
          <h3>Revisión Sistemática de Literatura (PRISMA 2020)</h3>
          <ul class="bullets-clara">
            <li><strong>Búsqueda exhaustiva en 4 APIs científicas:</strong> OpenAlex ($n=150$), Europe PMC ($n=62$), PubMed ($n=54$) y arXiv ($n=29$). Total: 295 brutos, 221 únicos tras deduplicación.</li>
            <li><strong>Validación rigurosa:</strong> 190 DOIs verificados en Crossref API (0 artículos retractados).</li>
            <li><strong>Concordancia inter-evaluador:</strong> Kappa de Cohen $\kappa = 0.6484$ (acuerdo sustancial con $82.35\%$ observado).</li>
            <li><strong>Puntaje metodológico:</strong> Media QA1–QA9 de $8.33/9.00$ sobre 27 variables extraídas.</li>
          </ul>
        </div>
        <div class="card-clara" style="border-top-color: var(--dorado-unal);">
          <h3>Hallazgos y Vacíos Críticos de la Literatura</h3>
          <ul class="bullets-clara">
            <li><strong>Caída promedio de transferencia:</strong> Pérdida media inter-dominio de casi 10 puntos de AUC en la literatura internacional ($\Delta\text{AUC} \approx 0.10$).</li>
            <li><strong>El Destructor de Rendimiento Primordial:</strong> La disparidad angular del campo de visión ($30^\circ-35^\circ$ papilar vs. $45^\circ$ panorámico macular).</li>
            <li><strong>El Vacío Identificado:</strong> Más del $90\%$ de los artículos científicos se limitan a tareas binarias simples, ignorando por completo la estadificación anatómica por severidad.</li>
          </ul>
        </div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Síntesis Bibliográfica Rigurosa</span>
      <span>Diapositiva 4 / 11</span>
    </div>
  </div>

  <!-- SLIDE 5: ECOSISTEMA DE DATOS HOSPITALARIOS [FONDO CLARO] -->
  <div class="slide slide-minas-clara" id="slide-5">
    <div class="slide-header">
      <span class="slide-tag">4. Ecosistema de Datos</span>
      <span class="slide-institution">Facultad de Minas — Universidad Nacional de Colombia</span>
    </div>
    <div class="slide-content">
      <h2 class="slide-title-clara">Cohortes Clínicas y Heterogeneidad Instrumental</h2>
      <p class="slide-subtitle-clara">Las diferencias técnicas y epidemiológicas entre bases explican la falla de transferencia.</p>
      <div class="grid-3" style="margin-top: 10px;">
        <div class="card-clara">
          <h3>ACRIMA (España)</h3>
          <p><strong>Cámara:</strong> Canon CR-DGi digital.<br/>
          <strong>Encuadre:</strong> $35^\circ$ macro centrado en papila.<br/>
          <strong>Muestra:</strong> $N = 705$.<br/>
          <strong>Sesgo Clave:</strong> $58.6\%$ de casos con glaucoma avanzado/tardío consolidado.</p>
        </div>
        <div class="card-clara">
          <h3>RIM-ONE DL (Canarias)</h3>
          <p><strong>Cámara:</strong> Kowa WX 3D estereoscópica.<br/>
          <strong>Encuadre:</strong> $35^\circ$ con lente polarizada.<br/>
          <strong>Muestra:</strong> $N = 447$.<br/>
          <strong>Sesgo Clave:</strong> $56.8\%$ de casos tempranos incipientes (detección precoz).</p>
        </div>
        <div class="card-clara">
          <h3>EyePACS (EE.UU.)</h3>
          <p><strong>Cámara:</strong> Centervue DRS / Topcon TRC.<br/>
          <strong>Encuadre:</strong> $45^\circ$ panorámico macular.<br/>
          <strong>Muestra:</strong> $N = 200$.<br/>
          <strong>Rol:</strong> Prueba de fuego de campo abierto (disco ocupa $< 6\%$ del lienzo).</p>
        </div>
      </div>
      <div class="card-clara" style="margin-top: 14px; border-top-color: var(--verde-minas-claro); padding: 12px 18px;">
        <p style="font-size: 0.88rem; color: var(--texto-tenue); margin: 0;">
          <strong>Nota Metodológica sobre Datos In-The-Wild:</strong> Evaluamos la recolección web (Google Images) con un filtro discriminador de calidad (purga de diagramas planos con 100% de precisión). Sin embargo, confirmamos que el ruido y la falta de metadatos clínicos de internet hacen inviable su uso directo para diagnóstico médico, justificando concentrar los esfuerzos en cohortes clínicas auditadas.
        </p>
      </div>
    </div>
    <div class="slide-footer">
      <span>Diversidad de Sensores y Cámaras</span>
      <span>Diapositiva 5 / 11</span>
    </div>
  </div>

  <!-- SLIDE 6: BENCHMARK DE BACKBONES [FONDO CLARO] -->
  <div class="slide slide-minas-clara" id="slide-6">
    <div class="slide-header">
      <span class="slide-tag">5. Selección del Modelo Base</span>
      <span class="slide-institution">Facultad de Minas — Universidad Nacional de Colombia</span>
    </div>
    <div class="slide-content">
      <div class="grid-2">
        <div>
          <h2 class="slide-title-clara">Benchmark de 10 Arquitecturas Base</h2>
          <p class="slide-subtitle-clara">Transfer Learning desde ImageNet: CNNs vs. Vision Transformers.</p>
          <ul class="bullets-clara">
            <li><strong>ConvNeXt-Tiny (Ganador Clínico):</strong>
              <br/><span class="badge-clara badge-green">AUC 0.9686</span> <span class="badge-clara badge-green">TFN 6.13%</span> <span class="badge-clara badge-gold">Latencia 8.58 ms</span>
              <br/>Kernel grande de $7\times 7$, retiene sesgo inductivo 2D y alcanza la menor tasa de omisión de enfermos.
            </li>
            <li><strong>ResNet-18 (Eficiencia Máxima):</strong>
              <br/><span class="badge-clara badge-green">AUC 0.9569</span> <span class="badge-clara badge-gold">Latencia 5.86 ms</span> <span class="badge-clara badge-gold">11.18 M Params</span>
              <br/>Columna vertebral óptima para dispositivos portátiles y procesamiento rápido.
            </li>
            <li><strong>Colapso de ViT-B/16 (Plain Transformer):</strong>
              <br/><span class="badge-clara badge-red">AUC 0.9022</span> <span class="badge-clara badge-red">TFN 17.18%</span> <span class="badge-clara badge-red">24.60 ms</span>
              <br/>Al carecer de sesgo local, sufre fuertemente en regímenes de datos médicos moderados ($N \approx 2,000$).
            </li>
          </ul>
        </div>
        <div class="slide-img-box-clara">
          <img src="@@IMG_BACKBONES@@" alt="Benchmark Multicriterio de Backbones"/>
          <p style="font-size:0.78rem; color:var(--texto-tenue); margin-top:4px;">Figura: Tradeoff Latencia vs. ROC-AUC vs. Parámetros.</p>
        </div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Evaluación y Tradeoff TFP vs. TFN</span>
      <span>Diapositiva 6 / 11</span>
    </div>
  </div>

  <!-- SLIDE 7: 2 VS 3 CLASES [FONDO CLARO] -->
  <div class="slide slide-minas-clara" id="slide-7">
    <div class="slide-header">
      <span class="slide-tag">6. Paradigmas de Clasificación</span>
      <span class="slide-institution">Facultad de Minas — Universidad Nacional de Colombia</span>
    </div>
    <div class="slide-content">
      <div class="grid-2">
        <div>
          <h2 class="slide-title-clara">¿Por qué Binario es Insuficiente?</h2>
          <p class="slide-subtitle-clara">El glaucoma es un espectro continuo; modelar 3 clases reduce fallos críticos.</p>
          <ul class="bullets-clara">
            <li><strong>Falso Confort Binario:</strong> El modelo binario reporta $86\%$ de exactitud, pero su TFN en glaucoma temprano es inaceptable ($12\%$) porque se acomoda a clasificar ojos terminales.</li>
            <li><strong>Formulación Trinaria Ordinal:</strong> $\mathcal{Y} = \{{Sano, Incipiente, Avanzado\}}$.</li>
            <li><strong>Desglose Formal de Errores:</strong>
              <ul style="margin-left: 20px; margin-top: 4px;">
                <li><span class="badge-clara badge-gold">Errores Adyacentes (Leves)</span> <strong>Sano ↔ Incipiente:</strong> Margen de incertidumbre normal entre oftalmólogos expertos.</li>
                <li><span class="badge-clara badge-red">Errores Catastróficos (Severos)</span> <strong>Sano ↔ Avanzado:</strong> Omisión letal de un paciente severo dado de alta como sano.</li>
              </ul>
            </li>
          </ul>
        </div>
        <div class="slide-img-box-clara">
          <img src="@@IMG_CONFUSION_3X3@@" alt="Matrices de Confusión 2 vs 3 clases"/>
          <p style="font-size:0.78rem; color:var(--texto-tenue); margin-top:4px;">Figura: Matriz 3x3 que desvela la distribución real de errores.</p>
        </div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Estadificación Clínica y Severidad</span>
      <span>Diapositiva 7 / 11</span>
    </div>
  </div>

  <!-- SLIDE 8: DATA AUGMENTATION MALIGNO [FONDO CLARO] -->
  <div class="slide slide-minas-clara" id="slide-8">
    <div class="slide-header">
      <span class="slide-tag">7. Estrategias Centradas en Datos</span>
      <span class="slide-institution">Facultad de Minas — Universidad Nacional de Colombia</span>
    </div>
    <div class="slide-content">
      <div style="text-align: center; margin-bottom: 8px;">
        <h2 class="slide-title-clara" style="font-size: 1.85rem; margin-bottom: 2px;">Malignant Scale Augmentation: Rompiendo el Atajo de Escala</h2>
        <p class="slide-subtitle-clara" style="margin-bottom: 8px;">Operador estocástico para forzar atención biológica e invarianza inter-cámara.</p>
      </div>
      <div class="slide-img-box-clara" style="margin-bottom: 12px;">
        <img src="@@IMG_DATA_AUG_STRIP@@" alt="Tira de 4 paneles Data Augmentation" style="max-height: 185px;"/>
      </div>
      <div class="grid-3">
        <div class="card-clara">
          <h3>Zoom-In Papilar (1.50x)</h3>
          <p>Forza a los filtros a enfocarse en el anillo neurorretiniano y la regla ISNT sin depender del fondo periférico.</p>
        </div>
        <div class="card-clara">
          <h3>Zoom-Out Severo (0.45x)</h3>
          <p>Incrusta la papila en lienzo coroideo sintético con máscara pupilar, induciendo resistencia ante cámaras $45^\circ$.</p>
        </div>
        <div class="card-clara" style="border-top-color: var(--dorado-unal);">
          <h3>Impacto Cuantitativo</h3>
          <p><span class="badge-clara badge-green">Errores Catastróficos: 20% ➔ 9.76%</span> (Reducción de más del $51\%$).</p>
          <p style="margin-top: 4px;"><span class="badge-clara badge-green">Acuerdo Cuadrático: QWK = 0.8163</span> (Concordancia casi perfecta).</p>
        </div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Data-Centric AI y Geometría</span>
      <span>Diapositiva 8 / 11</span>
    </div>
  </div>

  <!-- SLIDE 9: MATRIZ DE GENERALIZACIÓN [FONDO CLARO] -->
  <div class="slide slide-minas-clara" id="slide-9">
    <div class="slide-header">
      <span class="slide-tag">8. Validación Cruzada</span>
      <span class="slide-institution">Facultad de Minas — Universidad Nacional de Colombia</span>
    </div>
    <div class="slide-content">
      <div class="grid-2">
        <div>
          <h2 class="slide-title-clara">Superando el Colapso de Generalización</h2>
          <p class="slide-subtitle-clara">La combinación de datos federados y aumento de escala neutraliza el domain shift.</p>
          <ul class="bullets-clara">
            <li><strong>Entrenamiento Monomodal (ACRIMA):</strong> Se desploma a $0.5283$ en EyePACS (ciego a imágenes $45^\circ$).</li>
            <li><strong>Entrenamiento Bi-céntrico (ACRIMA + RIM-ONE):</strong> Recupera la varianza espectral ($0.7400$ en EyePACS).</li>
            <li><strong>Mega-Cohorte Federada + Malignant Scale:</strong>
              <br/><span class="badge-clara badge-green">ACRIMA: AUC 0.9950</span>
              <br/><span class="badge-clara badge-green">RIM-ONE: AUC 0.9710</span>
              <br/><span class="badge-clara badge-green">EyePACS: AUC 0.9400 (Especificidad: 100%)</span>
            </li>
            <li><strong>Lección de Ingeniería:</strong> El aumento estocástico cierra la brecha geométrica sin requerir costosas redes adversarias (GANs).</li>
          </ul>
        </div>
        <div class="slide-img-box-clara">
          <img src="@@IMG_GENERALIZATION@@" alt="Matriz Maestra de Generalización"/>
          <p style="font-size:0.78rem; color:var(--texto-tenue); margin-top:4px;">Figura: Matriz de transferencia cruzada y retención inter-dispositivo.</p>
        </div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Evaluación Inter-Hospitalaria</span>
      <span>Diapositiva 9 / 11</span>
    </div>
  </div>

  <!-- SLIDE 10: DEMOSTRACIÓN EN VIVO [FONDO VERDE MINAS OFICIAL] -->
  <div class="slide slide-minas-verde" id="slide-10">
    <div class="slide-header">
      <span class="slide-tag">9. Validación Experimental</span>
      <span class="slide-institution">Facultad de Minas — Universidad Nacional de Colombia</span>
    </div>
    <div class="slide-content" style="text-align: center;">
      <div style="max-width: 820px; margin: 0 auto;">
        <h2 class="slide-title-verde" style="font-size: 2.8rem; margin-bottom: 8px;">DEMOSTRACIÓN EN VIVO</h2>
        <p class="slide-subtitle-verde" style="font-size: 1.25rem;">Inferencia en Tiempo Real y Diagnóstico de Glaucoma</p>
        
        <div class="card-verde" style="text-align: left; padding: 26px;">
          <h3 style="font-size: 1.35rem; color: var(--dorado-unal); margin-bottom: 12px;">Protocolo de Prueba en Vivo ante el Jurado:</h3>
          <ul class="bullets-verde">
            <li><strong>Carga interactiva de retinografías:</strong> Prueba sobre imágenes de diferentes fuentes (hospitalarias $35^\circ$, cribado $45^\circ$ e imágenes heterogéneas).</li>
            <li><strong>Inferencia instantánea (< 15 ms en GPU / < 50 ms en CPU):</strong> Ejecución con los pesos congelados de <code>best_fine_tuned_convnext_tiny_glaucoma.pth</code>.</li>
            <li><strong>Resultados desplegados en pantalla:</strong>
              <div style="margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="badge-clara badge-green">Predicción Binaria (Normal vs. Glaucoma)</span>
                <span class="badge-clara badge-gold">Estadificación (Sano / Incipiente / Avanzado)</span>
                <span class="badge-clara badge-red">Probabilidad Softmax & Alerta de Calidad</span>
              </div>
            </li>
          </ul>
        </div>
        <p style="margin-top: 22px; font-style: italic; color: var(--dorado-suave); font-size: 1.05rem;">
          [ Transición para ejecutar la Live Demo interactiva ]
        </p>
      </div>
    </div>
    <div class="slide-footer">
      <span>Momento Cumbre de la Sustentación</span>
      <span>Diapositiva 10 / 11</span>
    </div>
  </div>

  <!-- SLIDE 11: CONCLUSIONES Y CIERRE [FONDO VERDE MINAS OFICIAL] -->
  <div class="slide slide-minas-verde" id="slide-11">
    <div class="slide-header">
      <span class="slide-tag">10. Conclusiones</span>
      <span class="slide-institution">Facultad de Minas — Universidad Nacional de Colombia</span>
    </div>
    <div class="slide-content">
      <h2 class="slide-title-verde" style="font-size: 2.2rem; margin-bottom: 6px;">Conclusiones y Recomendaciones de Ingeniería</h2>
      <p class="slide-subtitle-verde" style="font-size: 1.1rem; margin-bottom: 18px;">Lecciones metodológicas para futuros sistemas de cribado asistencial.</p>
      <div class="grid-2-eq">
        <div class="card-verde">
          <h3>Hallazgos Metodológicos Clave</h3>
          <ul class="bullets-verde">
            <li><strong>El espejismo mono-centro:</strong> Las métricas en un único hospital no son transferibles; la disparidad angular ($35^\circ$ vs $45^\circ$) es el mayor destructor de rendimiento.</li>
            <li><strong>CNNs superan a Transformers:</strong> En regímenes biomédicos moderados ($N \approx 2,000$), el sesgo inductivo convolucional supera a la auto-atención irrestricta.</li>
          </ul>
        </div>
        <div class="card-verde">
          <h3>Aportes en Ingeniería de Datos</h3>
          <ul class="bullets-verde">
            <li><strong>La estadificación de 3 clases es indispensable:</strong> Tratar el glaucoma como variable binaria oculta el daño inicial; el esquema trinario reduce a la mitad los fallos graves.</li>
            <li><strong>Data-Centric AI sobre complejidad de modelos:</strong> El aumento estocástico de escala (<em>Malignant Scale</em>) resolvió la brecha panorámica sin costosas arquitecturas adversarias.</li>
          </ul>
        </div>
      </div>
      <div style="text-align: center; margin-top: 22px;">
        <p style="font-size: 1.4rem; font-weight: 700; color: var(--dorado-unal);">¡Muchas gracias por su atención!</p>
        <p style="font-size: 0.95rem; color: #E2E8F0; margin-top: 4px;">Espacio abierto para preguntas y discusión académica.</p>
      </div>
    </div>
    <div class="slide-footer">
      <span>Cierre y Preguntas (Q&A)</span>
      <span>Diapositiva 11 / 11</span>
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

    rendered_html = template.replace("@@IMG_DOMAIN_SHIFT@@", img_domain_shift) \
                            .replace("@@IMG_BACKBONES@@", img_backbones) \
                            .replace("@@IMG_CONFUSION_3X3@@", img_confusion_3x3) \
                            .replace("@@IMG_DATA_AUG_STRIP@@", img_data_aug_strip) \
                            .replace("@@IMG_GENERALIZATION@@", img_generalization)

    print(f"[*] 2. Guardando presentación híbrida Minas en: {HTML_PATH.name}")
    HTML_PATH.write_text(rendered_html, encoding="utf-8")

    # Guión de Sustentación adaptado
    script_md = """# Guión Oficial de Exposición (11 Diapositivas / 12–14 Minutos)
## Formato Híbrido Institucional — Facultad de Minas (Universidad Nacional de Colombia)

**Materia:** Deep Learning (2026-2)  
**Tiempo estimado total:** 12 a 14 minutos + 3 a 5 minutos de preguntas.  

---

### Diapositiva 1: Portada y Presentación [FONDO VERDE MINAS] (1:00 min)
* "Buenas tardes profesor y compañeros. Hoy presentamos nuestra investigación titulada: *Evaluación de la generalización de clasificadores profundos de glaucoma ante heterogeneidad de adquisición de datos en retinografías*."
* "El propósito central de este trabajo fue superar el espejismo de los laboratorios cerrados: evaluar con rigor por qué los modelos que reportan 98% de AUC en un hospital colapsan al llevarse a la práctica clínica real ante cámaras y calibraciones distintas, y cómo resolverlo mediante ingeniería centrada en los datos."

---

### Diapositiva 2: La Problemática Clínica y el Espejismo Monocéntrico [FONDO CLARO] (1:15 min)
* "El glaucoma es una neuropatía silenciosa e irreversible: cuando el paciente nota pérdida visual, ya ha perdido más del 40% de sus fibras nerviosas."
* "En la literatura abundan modelos con ROC-AUC mayores a 0.98. Sin embargo, al evaluarlos en condiciones externas reales, descubrimos un desplome crítico: un modelo entrenado en ACRIMA (Canon 35°) se derrumba a 0.528 en EyePACS, lo cual equivale a clasificar al azar."
* "Las causas físicas son determinantes: disparidad de encuadre (35° macro frente a 45° panorámico) y variaciones de temperatura de color entre fabricantes de cámaras."

---

### Diapositiva 3: Delimitación Metodológica y Enfoque Sanitario (TFP vs. TFN) [FONDO CLARO] (1:15 min)
* "Delimitamos nuestro objeto material (clasificadores profundos CNNs y Transformers sobre retinografías) y objeto formal (la capacidad de generalización inter-dispositivo)."
* "Y adoptamos un mandato clínico deontológico: en medicina, el ROC-AUC abstracto es insuficiente. Lo que manda en salud pública es el balance entre Tasa de Falsos Negativos (TFN) y Tasa de Falsos Positivos (TFP)."
* "Minimizar TFN es éticamente imperativo para no enviar a casa a un paciente enfermo que terminará ciego. Minimizar TFP es indispensable para no colapsar las consultas de oftalmología con falsas alarmas."

---

### Diapositiva 4: Estado del Arte y Vacíos en la Literatura (PRISMA 2020) [FONDO CLARO] (1:15 min)
* "Ejecutamos una Revisión Sistemática de Literatura buscando en 4 APIs científicas (OpenAlex, Europe PMC, PubMed y arXiv), validando 190 artículos en Crossref y alcanzando un acuerdo sustancial de Kappa inter-evaluador de 0.648."
* "La literatura reveló dos vacíos mayúsculos: una pérdida promedio inter-dominio de casi 10 puntos de AUC en la literatura, y el hecho de que más del 90% de los estudios tratan el glaucoma como una variable binaria trivial, omitiendo por completo los estadíos tempranos."

---

### Diapositiva 5: Ecosistema de Datos y Heterogeneidad de Sensores [FONDO CLARO] (1:15 min)
* "Consolidamos cuatro cohortes internacionales con características ópticas y epidemiológicas marcadamente distintas:"
  * "ACRIMA (España): cámara Canon digital, recorte a 35°, con un 58.6% de casos avanzados."
  * "RIM-ONE DL (Canarias): cámara Kowa 3D estereoscópica, con un 56.8% de casos tempranos incipientes."
  * "EyePACS (EE.UU.): campo abierto a 45° donde el disco ocupa menos del 6% de la imagen."
  * "Moondream: cohorte con anotación ordinal de severidad en tres estadíos."
* "Evaluamos también la recolección web en Google Images con un filtro inteligente de calidad retiniana; sin embargo, confirmamos que el ruido y la falta de metadatos clínicos de internet hacen indispensable trabajar con cohortes clínicas auditadas."

---

### Diapositiva 6: Benchmark de Backbones y Tradeoff TFP vs. TFN [FONDO CLARO] (1:15 min)
* "Evaluamos diez arquitecturas base con Transfer Learning desde ImageNet."
* "El modelo ganador fue **ConvNeXt-Tiny**, logrando un AUC de 0.9686 y la menor tasa de omisión de enfermos del benchmark (TFN = 6.13%)."
* "**ResNet-18** demostró ser la red más rápida y eficiente (5.86 milisegundos de inferencia)."
* "Un hallazgo teórico fundamental: los Vision Transformers puros como ViT-B/16 sufrieron fuertemente (TFN = 17.18%), confirmando que el sesgo inductivo 2D de las convoluciones es superior cuando se cuenta con conjuntos biomédicos de tamaño moderado."

---

### Diapositiva 7: Paradigmas de Decisión: 2 vs. 3 Clases [FONDO CLARO] (1:15 min)
* "Demostramos que el paradigma binario ofrece un falso confort: la red aprende a buscar ojos terminales con excavaciones masivas y falla en la detección temprana."
* "Implementamos una estadificación en tres clases: Sano, Incipiente y Avanzado."
* "Esto nos permitió descomponer los fallos: los errores adyacentes (Sano confundido con Incipiente) reflejan la incertidumbre humana normal; pero los errores catastróficos (un paciente avanzado catalogado como sano) son inaceptables."

---

### Diapositiva 8: Data Augmentation Maligno e Invarianza de Escala [FONDO CLARO] (1:30 min)
* "Para cerrar la brecha geométrica entre cámaras de 35° y 45°, diseñamos *Malignant Scale Augmentation*."
* "En la tira de 4 paneles pueden ver la técnica: Zoom-In focal para forzar atención en la regla ISNT, Zoom-Out severo incrustando la papila en un lienzo sintético con máscara pupilar para simular campo abierto, y perturbaciones fotométricas para emular diferencias de sensor."
* "El resultado fue contundente: **redujimos los errores catastróficos a más de la mitad (del 20% al 9.76%)** y logramos un acuerdo cuadrático casi perfecto de QWK = 0.8163."

---

### Diapositiva 9: Matriz de Generalización Inter-Cámara [FONDO CLARO] (1:15 min)
* "Al integrar el aumento de escala y la mega-cohorte federada, superamos el colapso inter-hospitalario."
* "El rendimiento en EyePACS —que inicialmente era de 0.5283 (puro azar)— escaló hasta un AUC de **0.9400 con un 100% de especificidad**."
* "Esto demuestra que una estrategia centrada en los datos resuelve la brecha de dominio sin requerir costosas arquitecturas adversarias (GANs)."

---

### Diapositiva 10: DEMOSTRACIÓN EN VIVO (LIVE DEMO) [FONDO VERDE MINAS] (2:00 - 2:30 min)
* "Como la mejor prueba de robustez es la verificación en tiempo real, pasamos ahora a la demostración en vivo de nuestro clasificador."
* *(Transición para ejecutar la inferencia interactiva en vivo frente al jurado: cargar una imagen, mostrar la predicción binaria, la estadificación en 3 niveles y la distribución de confianza Softmax).*

---

### Diapositiva 11: Conclusiones y Cierre [FONDO VERDE MINAS] (1:00 min)
* "En conclusión: 1) Los modelos binarios tradicionales son frágiles en entornos reales; 2) Las CNNs modernas con sesgo inductivo superan a los Transformers en volúmenes clínicos moderados; 3) La estadificación de 3 clases es indispensable para no omitir glaucomas incipientes; y 4) Las intervenciones centradas en los datos son la clave para un despliegue asistencial fiable."
* "Muchas gracias. Quedamos atentos a sus preguntas y comentarios."
"""
    print(f"[*] 3. Guardando guión adaptado en: {SCRIPT_MD_PATH.name}")
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
        print(f"[*] 4. Compilando PDF institucional vía {Path(browser).name}...")
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
    generate_hybrid_slides()
