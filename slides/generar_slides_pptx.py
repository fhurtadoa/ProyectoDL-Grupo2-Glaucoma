#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de Presentación PPTX Oficial — Facultad de Minas (UNAL)
Diseño limpio, profesional y consistente en fondo institucional claro.
Encabezados calibrados con margen de seguridad debajo de la línea dorada superior.
Basado estrictamente en la plantilla: 20240716_Presentacion_Modelo_Verde_V2.pptx
"""

import sys, os, shutil
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml import parse_xml

BASE_DIR = Path(__file__).resolve().parent.parent
SLIDES_DIR = BASE_DIR / "slides"
FIGURES_DIR = BASE_DIR / "figures"
TEMPLATE_PATH = Path("/Users/fabian/Desktop/20240716_Presentacion_Modelo_Verde_V2.pptx")
OUTPUT_REPO_PATH = SLIDES_DIR / "PRESENTACION_GLAUCOMA_FACULTAD_DE_MINAS.pptx"
OUTPUT_DESKTOP_PATH = Path("/Users/fabian/Desktop/PRESENTACION_GLAUCOMA_FACULTAD_DE_MINAS.pptx")

# Colores Oficiales Institucionales
C_VERDE_MINAS = RGBColor(6, 120, 79)       # #06784F Verde Institucional
C_VERDE_CLARO = RGBColor(20, 196, 134)     # #14C486 Acento Esmeralda
C_DORADO = RGBColor(200, 157, 60)          # #C89D3C Dorado Institucional
C_TEXTO_DARK = RGBColor(30, 41, 59)        # #1E293B Carbón Slate
C_TEXTO_MUTED = RGBColor(100, 116, 139)    # #64748B Gris Secundario
C_BLANCO = RGBColor(255, 255, 255)         # Blanco
C_ROJO_ALERTA = RGBColor(225, 29, 72)      # #E11D48 Rojo Médico
C_AZUL_INFO = RGBColor(2, 132, 199)        # #0284C7 Azul Métrica

FONT_TITLE = "Georgia"
FONT_BODY = "Calibri"

def set_slide_background(slide, image_part):
    """Asigna la imagen de fondo de la plantilla."""
    rId = slide.part.relate_to(image_part, RT.IMAGE)
    bg_xml = f"""
    <p:bg xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
          xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
        <p:bgPr>
            <a:blipFill dpi="0" rotWithShape="1">
                <a:blip r:embed="{rId}"><a:lum/></a:blip>
                <a:srcRect/>
                <a:stretch><a:fillRect/></a:stretch>
            </a:blipFill>
            <a:effectLst/>
        </p:bgPr>
    </p:bg>
    """
    cSld = slide._element.xpath("./p:cSld")[0]
    existing_bg = cSld.xpath("./p:bg")
    new_bg = parse_xml(bg_xml)
    if existing_bg:
        cSld.replace(existing_bg[0], new_bg)
    else:
        cSld.insert(0, new_bg)

def clear_shapes(slide):
    """Elimina formas viejas para un lienzo completamente limpio."""
    for shape in list(slide.shapes):
        sp = shape._element
        sp.getparent().remove(sp)

def add_header(slide, tag_text, title_text):
    """
    Inserta el encabezado institucional limpio sobre fondo claro.
    top = 0.52 in garantiza que el texto comience limpiamente DEBAJO de la línea
    dorada pregrabada en image3.png (que termina a 0.41 in).
    """
    box = slide.shapes.add_textbox(Inches(0.6), Inches(0.52), Inches(8.8), Inches(0.85))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    # Subtítulo / Tag
    p_tag = tf.paragraphs[0]
    p_tag.text = tag_text.upper()
    p_tag.font.name = FONT_BODY
    p_tag.font.size = Pt(9.5)
    p_tag.font.bold = True
    p_tag.font.color.rgb = C_VERDE_MINAS
    p_tag.space_after = Pt(2)

    # Título Principal
    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.name = FONT_TITLE
    p_title.font.size = Pt(17.5)
    p_title.font.bold = True
    p_title.font.color.rgb = C_VERDE_MINAS

def add_bullet(tf, bold_label, normal_text, bullet=True, pt_size=9.8, text_color=C_TEXTO_DARK, space_after=5):
    """Inserta viñeta limpia con prefijo en negrita."""
    p = tf.add_paragraph() if len(tf.paragraphs[0].text) > 0 else tf.paragraphs[0]
    p.space_after = Pt(space_after)
    p.line_spacing = 1.15

    if bullet:
        r_b = p.add_run()
        r_b.text = "▪ "
        r_b.font.name = FONT_BODY
        r_b.font.size = Pt(pt_size)
        r_b.font.bold = True
        r_b.font.color.rgb = C_VERDE_MINAS

    if bold_label:
        r_lbl = p.add_run()
        r_lbl.text = bold_label + " "
        r_lbl.font.name = FONT_BODY
        r_lbl.font.size = Pt(pt_size)
        r_lbl.font.bold = True
        r_lbl.font.color.rgb = text_color

    r_txt = p.add_run()
    r_txt.text = normal_text
    r_txt.font.name = FONT_BODY
    r_txt.font.size = Pt(pt_size)
    r_txt.font.bold = False
    r_txt.font.color.rgb = text_color

def build_presentation():
    print("[*] Leyendo plantilla oficial Facultad de Minas...")
    prs = Presentation(str(TEMPLATE_PATH))
    slide_layout = prs.slide_layouts[1]

    # Extraer imágenes de fondo institucionales de la plantilla
    image3_part = prs.slides[2].part.rels["rId2"].target_part  # Fondo Claro Institucional
    image4_part = prs.slides[5].part.rels["rId2"].target_part  # Fondo Verde Cierre

    # Exactamente 12 diapositivas: 1 Portada, 2-9 Contenido, 10 Live Demo, 11 Conclusiones, 12 Gracias
    while len(prs.slides) < 12:
        prs.slides.add_slide(slide_layout)
    while len(prs.slides) > 12:
        rId = prs.slides._sldIdLst[-1].rId
        prs.part.drop_rel(rId)
        del prs.slides._sldIdLst[-1]

    slides = list(prs.slides)
    for s in slides:
        clear_shapes(s)

    # Slides 1 a 11 usan el fondo claro institucional image3_part
    for i in range(11):
        set_slide_background(slides[i], image3_part)
    # Slide 12 usa el cierre verde image4_part
    set_slide_background(slides[11], image4_part)

    # =========================================================================
    # SLIDE 1: PORTADA OFICIAL (Fondo Claro — Estilo Diapositiva 2 Plantilla)
    # =========================================================================
    print("[*] Slide 1: Portada Oficial en Fondo Claro...")
    s1 = slides[0]
    
    # Título Principal centrado (comienza en 1.35 in, bien lejos de la línea superior)
    box_t = s1.shapes.add_textbox(Inches(1.0), Inches(1.35), Inches(8.0), Inches(1.4))
    tf_t = box_t.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_right = tf_t.margin_top = tf_t.margin_bottom = 0
    p_t = tf_t.paragraphs[0]
    p_t.text = "Evaluación de la Generalización de Clasificadores Profundos de Glaucoma ante Heterogeneidad de Adquisición de Datos"
    p_t.font.name = FONT_TITLE
    p_t.font.size = Pt(22.5)
    p_t.font.bold = True
    p_t.font.color.rgb = C_VERDE_MINAS
    p_t.alignment = PP_ALIGN.CENTER

    # Línea decorativa dorada
    line_portada = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(2.5), Inches(2.85), Inches(5.0), Inches(0.03))
    line_portada.fill.solid()
    line_portada.fill.fore_color.rgb = C_DORADO
    line_portada.line.fill.background()

    # Subtítulo y contexto centrado
    box_sub = s1.shapes.add_textbox(Inches(1.0), Inches(3.0), Inches(8.0), Inches(1.8))
    tf_sub = box_sub.text_frame
    tf_sub.word_wrap = True
    tf_sub.margin_left = tf_sub.margin_right = tf_sub.margin_top = tf_sub.margin_bottom = 0
    
    p_sub = tf_sub.paragraphs[0]
    p_sub.text = "Estudio Multicéntrico, Estandarización de Encuadre e Impacto Clínico (TFP vs. TFN)"
    p_sub.font.name = FONT_BODY
    p_sub.font.size = Pt(13)
    p_sub.font.bold = True
    p_sub.font.color.rgb = C_TEXTO_DARK
    p_sub.alignment = PP_ALIGN.CENTER
    p_sub.space_after = Pt(16)

    p_ctx = tf_sub.add_paragraph()
    p_ctx.text = "Deep Learning (2026-2) • Universidad Nacional de Colombia — Sede Medellín\nFacultad de Minas • Departamento de Ciencias de la Computación y de la Decisión"
    p_ctx.font.name = FONT_BODY
    p_ctx.font.size = Pt(11)
    p_ctx.font.color.rgb = C_VERDE_MINAS
    p_ctx.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 2: PROBLEMÁTICA CLÍNICA Y ESPEJISMO MONOCÉNTRICO (Fondo Claro)
    # =========================================================================
    print("[*] Slide 2: Problemática Clínica...")
    s2 = slides[1]
    add_header(s2, "Problemática Clínica y Sesgo de Dominio", "El Espejismo Monocéntrico: Desplome de Generalización")

    box_s2 = s2.shapes.add_textbox(Inches(0.6), Inches(1.50), Inches(4.5), Inches(3.35))
    tf_s2 = box_s2.text_frame
    tf_s2.word_wrap = True
    tf_s2.margin_left = tf_s2.margin_right = tf_s2.margin_top = tf_s2.margin_bottom = 0

    add_bullet(tf_s2, "Neuropatía Silenciosa e Irreversible:", "El paciente pierde más del 40% de fibras axonales antes de notar síntomas. El diagnóstico tardío conduce a ceguera irreversible.")
    add_bullet(tf_s2, "El Espejismo del ROC-AUC > 0.98:", "Modelos de la literatura reportan desempeño casi perfecto en su hospital de origen, pero colapsan al evaluarse en condiciones externas.")
    add_bullet(tf_s2, "Derrumbe Externo a 0.528:", "Al probarlos en cámaras externas (EyePACS panorámico), clasifican prácticamente al azar.")
    add_bullet(tf_s2, "Causas Físicas del Domain Shift:", "Disparidad de encuadre (35° macro papilar vs. 45° macular panorámico) y variaciones térmicas del sensor CCD.")

    fig1_path = FIGURES_DIR / "fig1_domain_shift_comparison.png"
    if fig1_path.exists():
        s2.shapes.add_picture(str(fig1_path), Inches(5.3), Inches(1.60), width=Inches(4.2))
        lbl = s2.shapes.add_textbox(Inches(5.3), Inches(3.85), Inches(4.2), Inches(0.4))
        p = lbl.text_frame.paragraphs[0]
        p.text = "Figura 1: Domain shift óptico y disparidad de resolución inter-cámara."
        p.font.name = FONT_BODY
        p.font.size = Pt(8.5)
        p.font.italic = True
        p.font.color.rgb = C_TEXTO_MUTED
        p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 3: DELIMITACIÓN METODOLÓGICA Y MANDATO SANITARIO (Fondo Claro)
    # =========================================================================
    print("[*] Slide 3: Delimitación Metodológica y Tradeoff...")
    s3 = slides[2]
    add_header(s3, "Delimitación Metodológica y Enfoque Sanitario", "Objeto de Estudio y el Imperativo Clínico (TFP vs. TFN)")

    b_izq = s3.shapes.add_textbox(Inches(0.6), Inches(1.50), Inches(4.2), Inches(3.35))
    tf_izq = b_izq.text_frame
    tf_izq.word_wrap = True
    tf_izq.margin_left = tf_izq.margin_right = tf_izq.margin_top = tf_izq.margin_bottom = 0

    p_h1 = tf_izq.paragraphs[0]
    p_h1.text = "DELIMITACIÓN FORMAL DEL PROYECTO"
    p_h1.font.name = FONT_BODY
    p_h1.font.size = Pt(11.5)
    p_h1.font.bold = True
    p_h1.font.color.rgb = C_VERDE_MINAS
    p_h1.space_after = Pt(8)

    add_bullet(tf_izq, "Objeto Material:", "Clasificadores profundos CNNs (ConvNeXt, ResNet, DenseNet) contrastados frente a Vision Transformers (ViT, Swin) sobre retinografías.")
    add_bullet(tf_izq, "Objeto Formal:", "Capacidad de generalización ante variaciones ópticas de encuadre y heterogeneidad de sensores.")
    add_bullet(tf_izq, "Mitigación Centrada en Datos:", "Normalización cromática de Reinhard, aumento de datos adaptativo y entrenamiento multifuente coordinado.")

    b_der = s3.shapes.add_textbox(Inches(5.2), Inches(1.50), Inches(4.2), Inches(3.35))
    tf_der = b_der.text_frame
    tf_der.word_wrap = True
    tf_der.margin_left = tf_der.margin_right = tf_der.margin_top = tf_der.margin_bottom = 0

    p_h2 = tf_der.paragraphs[0]
    p_h2.text = "POR QUÉ EL ROC-AUC ES INSUFICIENTE"
    p_h2.font.name = FONT_BODY
    p_h2.font.size = Pt(11.5)
    p_h2.font.bold = True
    p_h2.font.color.rgb = C_ROJO_ALERTA
    p_h2.space_after = Pt(8)

    add_bullet(tf_der, "TFN (Falsos Negativos) — Riesgo Vital:", "Paciente enfermo enviado a casa sin tratamiento terminará ciego. Su minimización es ética y médicamente imperativa.")
    add_bullet(tf_der, "TFP (Falsos Positivos) — Sostenibilidad:", "Falsas alarmas que saturan las salas de oftalmología, generan angustia y causan angiografías innecesarias.")
    add_bullet(tf_der, "Umbral Operativo Adaptativo:", "Calibración dinámica del punto de corte para garantizar sensibilidad >95% en cribado masivo.")

    # =========================================================================
    # SLIDE 4: ESTADO DEL ARTE Y VACÍOS (PRISMA 2020)
    # =========================================================================
    print("[*] Slide 4: Estado del Arte...")
    s4 = slides[3]
    add_header(s4, "Estado del Arte — Directrices PRISMA 2020", "Revisión Sistemática de Literatura y Vacíos Metodológicos")

    b_c1 = s4.shapes.add_textbox(Inches(0.6), Inches(1.50), Inches(2.8), Inches(3.35))
    tf_c1 = b_c1.text_frame
    tf_c1.word_wrap = True
    tf_c1.margin_left = tf_c1.margin_right = tf_c1.margin_top = tf_c1.margin_bottom = 0
    p = tf_c1.paragraphs[0]
    p.text = "RIGOR PRISMA 2020"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_VERDE_MINAS
    p.space_after = Pt(8)
    add_bullet(tf_c1, "4 APIs Científicas:", "Búsqueda federada en OpenAlex, Europe PMC, PubMed y arXiv.")
    add_bullet(tf_c1, "190 Artículos Curados:", "Filtro automatizado con validación bibliométrica Crossref.")
    add_bullet(tf_c1, "Acuerdo Inter-Evaluador:", "Kappa de Cohen sustancial: κ = 0.648.")

    b_c2 = s4.shapes.add_textbox(Inches(3.6), Inches(1.50), Inches(2.8), Inches(3.35))
    tf_c2 = b_c2.text_frame
    tf_c2.word_wrap = True
    tf_c2.margin_left = tf_c2.margin_right = tf_c2.margin_top = tf_c2.margin_bottom = 0
    p = tf_c2.paragraphs[0]
    p.text = "VACÍO 1: CAÍDA INTER-DOMINIO"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_ROJO_ALERTA
    p.space_after = Pt(8)
    add_bullet(tf_c2, "Desplome Promedio:", "Pérdida de 9.8 puntos de ROC-AUC al evaluar modelos fuera del centro de entrenamiento.")
    add_bullet(tf_c2, "Sin Validación Externa:", "Menos del 12% de los artículos evalúan en múltiples marcas de cámaras.")
    add_bullet(tf_c2, "Aporte del Proyecto:", "Protocolo de robustez inter-dispositivo sistemático.")

    b_c3 = s4.shapes.add_textbox(Inches(6.6), Inches(1.50), Inches(2.8), Inches(3.35))
    tf_c3 = b_c3.text_frame
    tf_c3.word_wrap = True
    tf_c3.margin_left = tf_c3.margin_right = tf_c3.margin_top = tf_c3.margin_bottom = 0
    p = tf_c3.paragraphs[0]
    p.text = "VACÍO 2: ESPEJISMO BINARIO"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_AZUL_INFO
    p.space_after = Pt(8)
    add_bullet(tf_c3, "91.6% Simplista:", "La gran mayoría trata el glaucoma como variable binaria (Sano vs. Glaucoma).")
    add_bullet(tf_c3, "Oculta el Fallo Temprano:", "Ignoran que el estadio temprano tiene características sutiles y alta tasa de error.")
    add_bullet(tf_c3, "Nuestra Solución:", "Estadificación ordinal trinaria (Control, Temprano, Avanzado).")

    # =========================================================================
    # SLIDE 5: ECOSISTEMA DE DATOS Y HETEROGENEIDAD DE SENSORES
    # =========================================================================
    print("[*] Slide 5: Ecosistema de Datos...")
    s5 = slides[4]
    add_header(s5, "Corpus Experimental y Calidad de Datos", "Ecosistema de Cohortes Clínicas y Auditoría de Datos")

    w4 = Inches(2.1)
    datasets = [
        ("ACRIMA (España)", Inches(0.6), [("Muestra:", "705 imágenes."), ("Cámara:", "Canon CR-DGi."), ("Campo:", "35° macro papilar."), ("Severidad:", "58.6% casos avanzados.")]),
        ("RIM-ONE DL (Canarias)", Inches(2.8), [("Muestra:", "485 imágenes."), ("Cámara:", "Kowa 3D estereoscópica."), ("Campo:", "45° centrado en disco."), ("Severidad:", "56.8% incipientes.")]),
        ("EyePACS (EE.UU.)", Inches(5.0), [("Muestra:", "Telemedicina abierta."), ("Cámara:", "Centervue / Topcon."), ("Campo:", "45° macular abierto."), ("Detalle:", "Disco ocupa <6% de área.")]),
        ("Moondream (Ordinal)", Inches(7.2), [("Anotación:", "Validación 3 estadíos."), ("Clases:", "Control, Temprano, Avanzado."), ("Objetivo:", "Evaluar subdiagnóstico."), ("Utilidad:", "Pérdida ordinal sensible.")])
    ]

    for title_ds, left_ds, bullets in datasets:
        box_ds = s5.shapes.add_textbox(left_ds, Inches(1.50), w4, Inches(1.65))
        tf_ds = box_ds.text_frame
        tf_ds.word_wrap = True
        tf_ds.margin_left = tf_ds.margin_right = tf_ds.margin_top = tf_ds.margin_bottom = 0
        p = tf_ds.paragraphs[0]
        p.text = title_ds
        p.font.name = FONT_BODY
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = C_VERDE_MINAS
        p.space_after = Pt(4)
        for lbl_b, txt_b in bullets:
            add_bullet(tf_ds, lbl_b, txt_b, pt_size=9, space_after=2)

    box_deb = s5.shapes.add_textbox(Inches(0.6), Inches(3.35), Inches(8.8), Inches(1.50))
    tf_deb = box_deb.text_frame
    tf_deb.word_wrap = True
    tf_deb.margin_left = tf_deb.margin_right = tf_deb.margin_top = tf_deb.margin_bottom = 0
    p = tf_deb.paragraphs[0]
    p.text = "LECCIÓN APRENDIDA: EVALUACIÓN DE SCRAPING WEB (GOOGLE IMAGES) VS. COHORTES AUDITADAS"
    p.font.name = FONT_BODY
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = C_DORADO
    p.space_after = Pt(4)
    add_bullet(tf_deb, "Filtro Inteligente de Calidad:", "Se evaluó recolección masiva en internet filtrada con clasificador retiniano de calidad.")
    add_bullet(tf_deb, "Dispersión Óptica Ingobernable:", "La ausencia de metadatos clínicos auditados y la mezcla de artefactos web introducen ruido destructivo en el clasificador.")
    add_bullet(tf_deb, "Decisión de Ingeniería:", "Para garantizar rigor biomédico, se determinó mandatorio basar el estudio en cohortes hospitalarias curadas.")

    # =========================================================================
    # SLIDE 6: BENCHMARK DE 10 BACKBONES Y TRADEOFF TFP/TFN
    # =========================================================================
    print("[*] Slide 6: Benchmark de Backbones...")
    s6 = slides[5]
    add_header(s6, "Benchmark Arquitectural — Transfer Learning", "Benchmark de 10 Backbones y Minimización de TFN")

    box_s6 = s6.shapes.add_textbox(Inches(0.6), Inches(1.50), Inches(4.5), Inches(3.35))
    tf_s6 = box_s6.text_frame
    tf_s6.word_wrap = True
    tf_s6.margin_left = tf_s6.margin_right = tf_s6.margin_top = tf_s6.margin_bottom = 0

    add_bullet(tf_s6, "ConvNeXt-Tiny (Ganador Clínico):", "ROC-AUC: 0.9686 | F1-Score: 0.9312.")
    add_bullet(tf_s6, "Menor Omisión de Enfermos:", "TFN = 6.13% (la más baja del benchmark) y TFP = 7.42%.")
    add_bullet(tf_s6, "ResNet-18 (Eficiencia Extrema):", "Latencia ultrarrápida de sólo 5.86 ms/imagen (ROC-AUC = 0.9521). Ideal para edge computing y telemedicina.")
    add_bullet(tf_s6, "Vision Transformers (ViT-B/16):", "Colapso en sensibilidad clínica: TFN = 17.18%. Los parches de atención sin sesgo 2D convolucional fallan con datos médicos moderados.")

    fig30_path = FIGURES_DIR / "fig30_backbone_benchmark_multicriteria_tradeoff.png"
    if fig30_path.exists():
        s6.shapes.add_picture(str(fig30_path), Inches(5.3), Inches(1.50), width=Inches(4.0))
        lbl = s6.shapes.add_textbox(Inches(5.3), Inches(4.35), Inches(4.0), Inches(0.4))
        p = lbl.text_frame.paragraphs[0]
        p.text = "Figura 30: Frontera de Pareto: AUC vs. TFN y latencia de inferencia."
        p.font.name = FONT_BODY
        p.font.size = Pt(8.5)
        p.font.italic = True
        p.font.color.rgb = C_TEXTO_MUTED
        p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 7: PARADIGMAS 2 VS 3 CLASES Y DESGLOSE DE ERRORES
    # =========================================================================
    print("[*] Slide 7: Paradigmas 2 vs 3 Clases...")
    s7 = slides[6]
    add_header(s7, "Análisis de Estadificación y Errores Clínicos", "El Crítico Subdiagnóstico del Estadío Temprano")

    box_s7 = s7.shapes.add_textbox(Inches(0.6), Inches(1.50), Inches(4.5), Inches(3.35))
    tf_s7 = box_s7.text_frame
    tf_s7.word_wrap = True
    tf_s7.margin_left = tf_s7.margin_right = tf_s7.margin_top = tf_s7.margin_bottom = 0

    add_bullet(tf_s7, "El Espejismo de la Tarea Binaria:", "Al clasificar sólo Sano vs. Glaucoma, el modelo parece tener alta precisión. Mezclar tempranos con avanzados infla las métricas artificialmente.")
    add_bullet(tf_s7, "38.2% de Tempranos a la Calle:", "En el esquema de 3 clases, el 38.2% de los casos tempranos son clasificados como Sanos.")
    add_bullet(tf_s7, "Avanzados Detectados al 94.6%:", "La pérdida visual evidente es fácil de clasificar; el reto médico y la ventana de salvación visual radican exclusivamente en la detección temprana.")
    add_bullet(tf_s7, "Implicación en Ingeniería:", "Obligatoriedad de pérdidas ordinales sensibles a la distancia clínica y modelado de incertidumbre bayesiana.")

    fig33_path = FIGURES_DIR / "fig33_confusion_matrices_2_vs_3_classes.png"
    if fig33_path.exists():
        s7.shapes.add_picture(str(fig33_path), Inches(5.6), Inches(1.50), width=Inches(3.2))
        lbl = s7.shapes.add_textbox(Inches(5.2), Inches(4.55), Inches(4.0), Inches(0.4))
        p = lbl.text_frame.paragraphs[0]
        p.text = "Figura 33: Matrices de confusión normalizadas: Binario vs. Trinario ordinal."
        p.font.name = FONT_BODY
        p.font.size = Pt(8.5)
        p.font.italic = True
        p.font.color.rgb = C_TEXTO_MUTED
        p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 8: DATA AUGMENTATION MALIGNO E INVARIANZA DE ESCALA
    # =========================================================================
    print("[*] Slide 8: Data Augmentation...")
    s8 = slides[7]
    add_header(s8, "Ingeniería Centrada en los Datos", "Data Augmentation Maligno vs. Invarianza de Escala")

    strip_path = FIGURES_DIR / "fig_data_augmentation_4_panel_strip.png"
    if strip_path.exists():
        s8.shapes.add_picture(str(strip_path), Inches(1.4), Inches(1.50), width=Inches(7.2))
        lbl_aug = s8.shapes.add_textbox(Inches(1.4), Inches(3.65), Inches(7.2), Inches(0.25))
        p = lbl_aug.text_frame.paragraphs[0]
        p.text = "Panel cuádruple: Original • Zoom papilar • Zoom-out con lienzo sintético • Perturbación fotométrica."
        p.font.name = FONT_BODY
        p.font.size = Pt(8.5)
        p.font.italic = True
        p.font.color.rgb = C_TEXTO_MUTED
        p.alignment = PP_ALIGN.CENTER

    cols_aug = [
        ("ZOOM-IN EXTREMO (MALIGNO)", Inches(0.6), C_ROJO_ALERTA, "Recorta bordes papilares y destruye la relación copa/disco (CDR), induciendo falsos positivos masivos al mutilar el anillo neurorretiniano."),
        ("ZOOM-OUT SEVERO (MALIGNO)", Inches(3.6), C_DORADO, "Genera bordes negros artificiales y encoge la papila a <5% del encuadre; la red aprende atajos de frontera en vez de patología."),
        ("AUMENTO FOTOMÉTRICO (SEGURO)", Inches(6.6), C_VERDE_MINAS, "Brillo controlado (±15%) y filtros CLAHE simulan disparidad de fuentes de luz preservando intacta la morfología vascular.")
    ]

    for title_a, left_a, col_a, txt_a in cols_aug:
        b_a = s8.shapes.add_textbox(left_a, Inches(3.95), Inches(2.8), Inches(1.0))
        tf_a = b_a.text_frame
        tf_a.word_wrap = True
        tf_a.margin_left = tf_a.margin_right = tf_a.margin_top = tf_a.margin_bottom = 0
        p = tf_a.paragraphs[0]
        p.text = title_a
        p.font.name = FONT_BODY
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = col_a
        p.space_after = Pt(2)
        p_txt = tf_a.add_paragraph()
        p_txt.text = txt_a
        p_txt.font.name = FONT_BODY
        p_txt.font.size = Pt(8.5)
        p_txt.font.color.rgb = C_TEXTO_DARK

    # =========================================================================
    # SLIDE 9: MATRIZ MAESTRA DE GENERALIZACIÓN INTER-CÁMARA
    # =========================================================================
    print("[*] Slide 9: Matriz de Generalización...")
    s9 = slides[8]
    add_header(s9, "Validación Externa y Generalización", "Evaluación Cruzada Inter-Dispositivo y Mitigación")

    box_s9 = s9.shapes.add_textbox(Inches(0.6), Inches(1.50), Inches(4.5), Inches(3.35))
    tf_s9 = box_s9.text_frame
    tf_s9.word_wrap = True
    tf_s9.margin_left = tf_s9.margin_right = tf_s9.margin_top = tf_s9.margin_bottom = 0

    add_bullet(tf_s9, "Colapso Basal ACRIMA → EyePACS:", "Caída crítica de ΔAUC = -0.44 (desempeño de 0.528, equivalente al azar) por disparidad de encuadre (35° vs 45°).")
    add_bullet(tf_s9, "Mitigación Centrada en Datos:", "1. Normalización de color de Reinhard para alinear histogramas RGB.\n2. Recorte papilar adaptativo para estandarizar escala anatómica.")
    add_bullet(tf_s9, "Recuperación a 0.8912:", "El entrenamiento multientorno balanceado recupera casi por completo la generalización externa.")
    add_bullet(tf_s9, "Lección Fundamental:", "La ingeniería de datos produce un impacto clínico superior al ajuste fino de hiperparámetros.")

    fig21_path = FIGURES_DIR / "fig21_cross_device_generalization_master.png"
    if fig21_path.exists():
        s9.shapes.add_picture(str(fig21_path), Inches(5.3), Inches(1.60), width=Inches(4.2))
        lbl = s9.shapes.add_textbox(Inches(5.3), Inches(3.60), Inches(4.2), Inches(0.4))
        p = lbl.text_frame.paragraphs[0]
        p.text = "Figura 21: Matriz cruzada de evaluación inter-dominio antes y después de la mitigación."
        p.font.name = FONT_BODY
        p.font.size = Pt(8.5)
        p.font.italic = True
        p.font.color.rgb = C_TEXTO_MUTED
        p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 10: DEMOSTRACIÓN EN VIVO (Fondo Claro Institucional)
    # =========================================================================
    print("[*] Slide 10: Live Demo en Fondo Claro...")
    s10 = slides[9]
    add_header(s10, "Demostración Interactiva en Vivo", "Inferencia en Tiempo Real sobre Retinografías Clínicas")

    box_s10 = s10.shapes.add_textbox(Inches(0.6), Inches(1.50), Inches(8.8), Inches(3.35))
    tf_s10 = box_s10.text_frame
    tf_s10.word_wrap = True
    tf_s10.margin_left = tf_s10.margin_right = tf_s10.margin_top = tf_s10.margin_bottom = 0

    p_lh = tf_s10.paragraphs[0]
    p_lh.text = "PROTOCOLO DE LA DEMOSTRACIÓN EN TIEMPO REAL"
    p_lh.font.name = FONT_BODY
    p_lh.font.size = Pt(12)
    p_lh.font.bold = True
    p_lh.font.color.rgb = C_VERDE_MINAS
    p_lh.space_after = Pt(10)

    add_bullet(tf_s10, "1. Carga del Modelo Campeón:", "Inicialización de ConvNeXt-Tiny con pesos preentrenados y optimizados para minimización de TFN.", pt_size=10.5, space_after=8)
    add_bullet(tf_s10, "2. Selección de Retinografía en Vivo:", "Comparación de imagen macro (ACRIMA 35°) frente a panorámica macular (EyePACS 45°).", pt_size=10.5, space_after=8)
    add_bullet(tf_s10, "3. Normalización Fotométrica y Recorte Dinámico:", "Preprocesamiento adaptativo para garantizar invarianza de escala y color.", pt_size=10.5, space_after=8)
    add_bullet(tf_s10, "4. Veredicto Diagnóstico y Alerta Sanitaria:", "Estimación instantánea de probabilidad y clasificación bajo política de salud pública.", pt_size=10.5, space_after=14)

    p_trans = tf_s10.add_paragraph()
    p_trans.text = "▶ PASANDO A LA EJECUCIÓN EN EL ENTORNO INTERACTIVO..."
    p_trans.font.name = FONT_BODY
    p_trans.font.size = Pt(12)
    p_trans.font.bold = True
    p_trans.font.color.rgb = C_ROJO_ALERTA
    p_trans.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 11: CONCLUSIONES Y RECOMENDACIONES CLÍNICAS (Fondo Claro)
    # =========================================================================
    print("[*] Slide 11: Conclusiones en Fondo Claro...")
    s11 = slides[10]
    add_header(s11, "Síntesis y Pautas de Despliegue", "Conclusiones y Recomendaciones de Ingeniería Clínica")

    # Columna 1
    box_k1 = s11.shapes.add_textbox(Inches(0.6), Inches(1.50), Inches(4.2), Inches(3.35))
    tf_k1 = box_k1.text_frame
    tf_k1.word_wrap = True
    tf_k1.margin_left = tf_k1.margin_right = tf_k1.margin_top = tf_k1.margin_bottom = 0

    p_c1_t = tf_k1.paragraphs[0]
    p_c1_t.text = "1. Métrica Deontológica Sanitaria"
    p_c1_t.font.name = FONT_BODY
    p_c1_t.font.size = Pt(11.5)
    p_c1_t.font.bold = True
    p_c1_t.font.color.rgb = C_VERDE_MINAS
    p_c1_t.space_after = Pt(2)
    add_bullet(tf_k1, "Superar el AUC Abstracto:", "En cribado médico masivo, la prioridad imperativa es minimizar la TFN sin desbordar la TFP por sobrecostos.", bullet=False, pt_size=9.8, space_after=14)

    p_c2_t = tf_k1.add_paragraph()
    p_c2_t.text = "2. Superioridad Convolucional (ConvNeXt)"
    p_c2_t.font.name = FONT_BODY
    p_c2_t.font.size = Pt(11.5)
    p_c2_t.font.bold = True
    p_c2_t.font.color.rgb = C_VERDE_MINAS
    p_c2_t.space_after = Pt(2)
    add_bullet(tf_k1, "El Sesgo Inductivo Importa:", "ConvNeXt-Tiny superó a los Vision Transformers (ViT) al retener priors espaciales 2D en cohortes médicas moderadas.", bullet=False, pt_size=9.8)

    # Columna 2
    box_k2 = s11.shapes.add_textbox(Inches(5.2), Inches(1.50), Inches(4.2), Inches(3.35))
    tf_k2 = box_k2.text_frame
    tf_k2.word_wrap = True
    tf_k2.margin_left = tf_k2.margin_right = tf_k2.margin_top = tf_k2.margin_bottom = 0

    p_c3_t = tf_k2.paragraphs[0]
    p_c3_t.text = "3. El Reto del Estadío Temprano"
    p_c3_t.font.name = FONT_BODY
    p_c3_t.font.size = Pt(11.5)
    p_c3_t.font.bold = True
    p_c3_t.font.color.rgb = C_VERDE_MINAS
    p_c3_t.space_after = Pt(2)
    add_bullet(tf_k2, "Subdiagnóstico Oculto:", "La clasificación binaria oculta un 38.2% de falsos negativos iniciales. Es mandatorio adoptar modelado ordinal.", bullet=False, pt_size=9.8, space_after=14)

    p_c4_t = tf_k2.add_paragraph()
    p_c4_t.text = "4. Pauta de Despliegue en Hospitales"
    p_c4_t.font.name = FONT_BODY
    p_c4_t.font.size = Pt(11.5)
    p_c4_t.font.bold = True
    p_c4_t.font.color.rgb = C_VERDE_MINAS
    p_c4_t.space_after = Pt(2)
    add_bullet(tf_k2, "Estandarización Obligatoria:", "Ningún clasificador debe recibir imágenes panorámicas directas sin recorte previo y normalización cromática de Reinhard.", bullet=False, pt_size=9.8)

    # =========================================================================
    # SLIDE 12: CIERRE OFICIAL (Verde Minas — Estilo Diapositiva 6 Plantilla)
    # =========================================================================
    print("[*] Slide 12: Cierre Oficial en Verde...")
    s12 = slides[11]
    
    # Cuadro central exclusivo para "Gracias", ubicado libre de logos
    box_gracias = s12.shapes.add_textbox(Inches(1.5), Inches(1.9), Inches(7.0), Inches(1.5))
    tf_g = box_gracias.text_frame
    tf_g.word_wrap = True
    tf_g.margin_left = tf_g.margin_right = tf_g.margin_top = tf_g.margin_bottom = 0

    p_g = tf_g.paragraphs[0]
    p_g.text = "¡Muchas Gracias!"
    p_g.font.name = FONT_TITLE
    p_g.font.size = Pt(36)
    p_g.font.bold = True
    p_g.font.color.rgb = C_BLANCO
    p_g.alignment = PP_ALIGN.CENTER
    p_g.space_after = Pt(8)

    p_preg = tf_g.add_paragraph()
    p_preg.text = "¿Preguntas o Discusión Técnica?"
    p_preg.font.name = FONT_BODY
    p_preg.font.size = Pt(14)
    p_preg.font.color.rgb = C_DORADO
    p_preg.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # GUARDAR
    # =========================================================================
    print(f"[*] Guardando en repositorio: {OUTPUT_REPO_PATH}")
    prs.save(str(OUTPUT_REPO_PATH))

    print(f"[*] Guardando copia en Desktop: {OUTPUT_DESKTOP_PATH}")
    shutil.copy2(OUTPUT_REPO_PATH, OUTPUT_DESKTOP_PATH)

    print("[EXITO] Presentación PPTX limpia y consistente generada con éxito.")

if __name__ == "__main__":
    build_presentation()
