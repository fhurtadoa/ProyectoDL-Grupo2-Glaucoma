"""Medidas visuales deterministas de los archivos tal como se almacenan."""

import hashlib
import io

import cv2
import numpy as np
from PIL import Image


def thumbnail(image, size=512):
    height, width = image.shape[:2]
    scale = size / max(height, width)
    shape = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, shape, interpolation=interpolation)


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def padding_mask(rgb):
    """Busca relleno casi uniforme conectado a una esquina, también si es de color."""
    height, width = rgb.shape[:2]
    block = 8
    detected = np.zeros((height, width), dtype=bool)
    colors = rgb.astype('int16')
    for x, y in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        patch = colors[:block] if y == 0 else colors[-block:]
        patch = patch[:, :block] if x == 0 else patch[:, -block:]
        if np.ptp(patch, axis=(0, 1)).max() > 6:
            continue
        level = np.median(patch, axis=(0, 1)).astype('int16')
        if np.max(level) <= 24:
            continue  # ya cubierto por el umbral de fondo oscuro
        similar = np.all(np.abs(colors - level) <= 8, axis=2).astype('uint8')
        if not similar[y, x]:
            continue
        n, _, _, _ = cv2.floodFill(similar, None, (x, y), 2, flags=4)
        if 0.003 * similar.size <= n <= 0.45 * similar.size:
            component = (similar == 2).astype('uint8')
            outside = cv2.dilate(component, np.ones((5, 5), 'uint8')) > component
            jumps = np.max(np.abs(colors[outside] - level), axis=1) >= 16
            if jumps.size and jumps.mean() >= 0.30:
                detected |= component.astype(bool)
    return detected


def retinal_mask(rgb):
    """Devuelve una aproximación del área visible y el estado de la estimación."""
    height, width = rgb.shape[:2]
    if min(height, width) < 16:
        return None, 'miniatura_demasiado_estrecha'
    foreground = ((gray(rgb) > 12) | (rgb[:, :, 0] > 24)) & ~padding_mask(rgb)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(foreground.astype('uint8'), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 'sin_componente_visible'
    largest = max(contours, key=cv2.contourArea)
    mask = cv2.drawContours(np.zeros_like(closed), [largest], -1, 1, cv2.FILLED).astype(bool)
    if mask.mean() < 0.18:
        return mask, 'componente_pequeno'
    if np.count_nonzero(foreground & mask) < 0.86 * np.count_nonzero(foreground):
        return mask, 'componentes_dispersos'
    if np.count_nonzero(foreground & mask) < 0.80 * np.count_nonzero(mask):
        return mask, 'componente_con_grandes_huecos'

    k = max(1, round(min(height, width) * 0.08))
    corners = (mask[:k, :k], mask[:k, -k:], mask[-k:, :k], mask[-k:, -k:])
    dark_corners = sum(corner.mean() < 0.25 for corner in corners)
    touch = max(edge.mean() for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]))
    x, y, w, h = cv2.boundingRect(largest)
    border_separated = min(x, y, width - x - w, height - y - h) > 0
    perimeter = cv2.arcLength(largest, True)
    circularity = 4 * np.pi * cv2.contourArea(largest) / perimeter**2 if perimeter else 0
    if dark_corners == 4 and min(w, h) / max(w, h) >= 0.80 and circularity >= 0.76:
        if border_separated or touch <= 0.16:
            return mask, 'circular_visible'
    if mask.mean() >= 0.95 and dark_corners == 0:
        return mask, 'rectangular_sin_contorno'
    return mask, 'parcial_o_irregular'


def laplacian_variance(luminance, mask):
    inner = cv2.erode(mask.astype('uint8'), np.ones((3, 3), 'uint8'),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    if not inner.any():
        return None
    return float(cv2.Laplacian(luminance, cv2.CV_32F)[inner].var(dtype='float64'))


def outer_band(shape, fraction):
    height, width = shape
    band = np.ones(shape, dtype=bool)
    y, x = max(1, round(fraction * height)), max(1, round(fraction * width))
    band[y:-y, x:-x] = False
    return band


def candidate_text(rgb, luminance, mask):
    """Cuenta grupos de componentes brillantes alineados en la periferia."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    top_hat = cv2.morphologyEx(luminance, cv2.MORPH_TOPHAT, kernel)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    search = outer_band(luminance.shape, 0.18)
    if mask is not None:
        search |= ~mask
    pixels = search & (top_hat >= 25) & (luminance >= 150) & (chroma <= 60)
    _, _, boxes, _ = cv2.connectedComponentsWithStats(pixels.astype('uint8'))
    glyphs = [(x, y, w, h) for x, y, w, h, area in boxes[1:]
              if 2 <= w <= 26 and 4 <= h <= 28 and 4 <= area <= 200]
    lines = np.zeros(luminance.shape, dtype='uint8')
    for x, y, w, h in glyphs:
        lines[int(y + h / 2), int(x + w / 2)] = 1
    connected = cv2.dilate(lines, np.ones((7, 23), 'uint8'))
    count, labels = cv2.connectedComponents(connected)
    per_line = np.bincount(labels[lines > 0], minlength=count)
    return int(np.count_nonzero(per_line[1:] >= 3))


def measure(path, kind):
    data = path.read_bytes()
    result = {'sha256_archivo': hashlib.sha256(data).hexdigest()}
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        width, height = image.size
        mode = 'RGBA' if 'A' in image.getbands() or 'transparency' in image.info else 'RGB'
        pixels = np.asarray(image.convert(mode))
        digest = hashlib.sha256(f'{mode}:{width}:{height}:'.encode())
        digest.update(pixels.tobytes())
        exif = image.getexif()
        result.update(ancho_px=width, alto_px=height, aspecto=width / height,
                      formato=image.format, modo_original=image.mode,
                      sha256_pixeles=digest.hexdigest(), camara_exif=' '.join(
                          str(exif[tag]).strip() for tag in (271, 272) if exif.get(tag)))
    rgb = np.ascontiguousarray(pixels[:, :, :3])
    result['negro_pct'] = 100 * np.mean(rgb.max(axis=2) <= 10)
    result['rgb_identico_pct'] = 100 * np.mean((rgb[:, :, 0] == rgb[:, :, 1]) & (rgb[:, :, 1] == rgb[:, :, 2]))
    if mode == 'RGBA':
        result['alfa_no_cero_pct'] = 100 * np.mean(pixels[:, :, 3] > 0)
    if kind == 'distractor_web_sintetico':
        thumb = thumbnail(rgb)
        result['grupos_texto_candidato'] = candidate_text(thumb, gray(thumb), None)
    if kind not in {'retinografia', 'armonizada', 'emulacion_web'}:
        result.update(estado='solo_archivo', motivo='tipo_sin_metricas_retinales', region='no_aplica')
        return result

    thumb = thumbnail(rgb)
    luminance_thumb = gray(thumb)
    mask, framing = retinal_mask(thumb)
    thumb_black = thumb.max(axis=2) <= 10
    border = outer_band(luminance_thumb.shape, 0.05)
    result.update(region=framing, negro_borde_pct=100 * np.mean(thumb_black[border]),
                  bordes_periferia_pct=100 * np.mean(cv2.Canny(luminance_thumb, 50, 100)[border] > 0),
                  grupos_texto_candidato=candidate_text(thumb, luminance_thumb, mask))
    if framing not in {'circular_visible', 'rectangular_sin_contorno', 'parcial_o_irregular'}:
        result.update(estado='region_no_evaluable', motivo=framing)
        return result

    y, x = np.nonzero(mask)
    fraction = mask.mean()
    result.update(estado='completa', region_pct=100 * fraction,
                  centro_x=(x.mean() + 0.5) / mask.shape[1],
                  centro_y=(y.mean() + 0.5) / mask.shape[0],
                  diametro_visible_equiv_px=2 * np.sqrt(fraction * height * width / np.pi),
                  extension_x=(x.max() - x.min() + 1) / mask.shape[1],
                  extension_y=(y.max() - y.min() + 1) / mask.shape[0])
    if framing == 'circular_visible':
        yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
        radius = np.sqrt(((xx - x.mean()) / ((x.max() - x.min() + 1) / 2))**2
                         + ((yy - y.mean()) / ((y.max() - y.min() + 1) / 2))**2)
        center = mask & (radius <= 0.40)
        ring = mask & (radius >= 0.72) & (radius <= 0.90)
        if center.any() and ring.any():
            result['caida_radial_yn'] = float(np.median(luminance_thumb[center]) - np.median(luminance_thumb[ring]))

    native_mask = cv2.resize(mask.astype('uint8'), (width, height),
                             interpolation=cv2.INTER_NEAREST_EXACT).astype(bool)
    luminance = gray(rgb)
    roi = luminance[native_mask]
    rgb_roi = rgb[native_mask]
    result.update(brillo_medio_yn=roi.mean(), brillo_mediano_yn=np.median(roi),
                  contraste_de_yn=roi.std(),
                  saturacion_media_pct=100 * cv2.cvtColor(rgb_roi[:, None, :], cv2.COLOR_RGB2HSV)[:, 0, 1].mean() / 255,
                  laplaciano_nativo=laplacian_variance(luminance, native_mask),
                  laplaciano_lado512=laplacian_variance(luminance_thumb, mask))
    for channel, mean, std in zip('rgb', rgb_roi.mean(axis=0), rgb_roi.std(axis=0)):
        result[f'{channel}_media'] = mean
        result[f'{channel}_de'] = std
    return result


def disk_area(path, row):
    """Máscara binaria verificada de disco completo en coordenadas del archivo."""
    with Image.open(path) as image:
        mask = np.asarray(image.convert('L'))
    if mask.shape != (row['alto_px'], row['ancho_px']):
        raise ValueError('dimensiones_distintas')
    values = np.unique(mask)
    if len(values) != 2 or values[0] != 0:
        raise ValueError('mascara_no_binaria_o_sin_fondo')
    return 100 * np.count_nonzero(mask) / mask.size
