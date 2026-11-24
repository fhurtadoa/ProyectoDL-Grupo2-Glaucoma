"""Inventario y verificación de correspondencias con los archivos fuente locales."""

import csv
import hashlib
import io
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter


EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp', '.gif', '.jp2', '.heic', '.avif'}


def relative(path, root):
    return Path(os.path.relpath(path, root)).as_posix()


def image_paths(path):
    return sorted(file for file in path.rglob('*') if file.is_file() and file.suffix.lower() in EXTENSIONS)


def manifest_rows(root, settings):
    if not settings or not (root / settings['ruta']).is_file():
        return []
    with (root / settings['ruta']).open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file, delimiter=settings.get('separador', ',')))


def manifest_path(raw_path, source_path, root):
    path = raw_path.replace('\\', '/')
    source_prefix = relative(source_path, root) + '/'
    if '/' + source_prefix in path:
        return path[path.index('/' + source_prefix) + 1:]
    if path.startswith(source_prefix):
        return path
    return relative((source_path / path).resolve(), root)


def inventory(config, root):
    rows, manifest_audit = [], {}
    sources = config['fuentes']
    if len({source['id'] for source in sources}) != len(sources):
        raise ValueError('Cada fuente requiere un id distinto')
    scanned = set()
    for source in sources:
        folder = (root / source['ruta']).resolve()
        settings = source.get('manifiesto', {})
        index = manifest_rows(root, settings)
        by_path, by_id = {}, {}
        for entry in index:
            path = manifest_path(entry[settings['ruta_col']], folder, root)
            if path in by_path:
                raise ValueError(f'Ruta repetida en {settings["ruta"]}: {path}')
            by_path[path] = entry
            if 'id_col' in settings:
                identifier = entry[settings['id_col']]
                if identifier in by_id:
                    raise ValueError(f'ID repetido en {settings["ruta"]}: {identifier}')
                by_id[identifier] = entry

        paths = image_paths(folder)
        matched = set()
        for path in paths:
            if path in scanned:
                raise ValueError(f'Raíces de fuentes superpuestas: {path}')
            scanned.add(path)
            local_path = relative(path, root)
            parts = path.relative_to(folder).parts
            kind = source['tipo']
            label = ''
            if source.get('etiquetas') == 'nombre_acrima':
                match = re.fullmatch(r'Im\d+(_g)?_ACRIMA', path.stem, flags=re.IGNORECASE)
                if match:
                    label = 'glaucoma' if match[1] else 'normal'
            if source.get('etiquetas') == 'carpeta' and len(parts) > 1:
                label = parts[-2]
            partition = source.get('particion', '')
            if 'particion_nivel' in source and len(parts) > source['particion_nivel'] + 1:
                partition = parts[source['particion_nivel']]
            manifest_state = 'sin_manifiesto'
            if settings:
                manifest_state = 'archivo_sin_fila' if index else 'manifiesto_no_disponible_o_vacio'
            entry = by_path.get(local_path)
            if not entry and source.get('enlace'):
                entry = by_id.get(path.name)
            row = {'id_imagen': local_path, 'coleccion': source['id'], 'ruta_relativa': local_path,
                   'archivo': path.name, 'tipo': kind, 'particion': partition, 'etiqueta': label,
                   'esquema_etiqueta': source.get('esquema', ''), 'etiqueta_carpeta': label if source.get('etiquetas') == 'carpeta' else '',
                   'origen_etiqueta': source.get('etiquetas', '') if label else '', 'estado_manifiesto': manifest_state,
                   'camara_documentada': source.get('camara_documentada', ''), 'fuente_indice': '',
                   'comparar': source.get('comparar', False), 'path': path}
            if entry:
                indexed_label = entry.get(settings.get('etiqueta_col'), '')
                indexed_partition = entry.get(settings.get('particion_col'), '')
                values = settings.get('clases', {})
                mapped_label = values.get(indexed_label, indexed_label)
                if source.get('enlace'):
                    aligned = parts == (indexed_partition, mapped_label, path.name)
                    row['estado_manifiesto'] = 'indexado' if aligned else 'duplicado_id_fuera_ruta_indexada'
                else:
                    row['estado_manifiesto'] = 'indexado'
                if row['estado_manifiesto'] == 'indexado':
                    matched.add(manifest_path(entry[settings['ruta_col']], folder, root))
                row.update(etiqueta_indice=indexed_label, etiqueta=mapped_label or label,
                           origen_etiqueta=settings['ruta'] if indexed_label else row['origen_etiqueta'],
                           particion_indice=indexed_partition, particion=partition or indexed_partition,
                           fuente_indice=entry.get(settings.get('fuente_col'), ''))
                row['camara_documentada'] = entry.get(settings.get('camara_col'), '') or row['camara_documentada']
            if source.get('enlace_web') and not re.fullmatch(r'web_fundus_\d+\.jpg', path.name):
                row['tipo'] = 'distractor_web_sintetico'
            rows.append(row)
        manifest_audit[source['id']] = {
            'filas_manifiesto': len(index), 'filas_sin_archivo_en_ruta_esperada': len(by_path.keys() - matched),
            'rutas_manifiesto_faltantes': sorted(by_path.keys() - matched),
        }

    data = (root / config['raiz_datos']).resolve()
    for path in image_paths(data):
        if path not in scanned:
            rel = relative(path, root)
            rows.append({'id_imagen': rel, 'coleccion': 'Sin configurar', 'ruta_relativa': rel, 'archivo': path.name,
                         'tipo': 'sin_configurar', 'comparar': False, 'path': path})
    return rows, manifest_audit


def pair_candidates(rows, sources):
    """Recupera el candidato por la numeración de los scripts 03 y 07."""
    by_source = defaultdict(list)
    for row in rows:
        by_source[row['coleccion']].append(row)
    candidates = {}
    for source in sources:
        collection = by_source[source['id']]
        for original_name, prefix, transform in source.get('enlace', {}).values():
            original = sorted(by_source[original_name], key=lambda row: row['ruta_relativa'])
            suffix = '.png' if transform == 'recortar_color' or prefix == 'RIMONE_' else '.jpg'
            original = [row for row in original if row['path'].suffix.lower() == suffix]
            pattern = re.compile(re.escape(prefix) + r'(\d+)\.jpg', re.IGNORECASE)
            for row in collection:
                match = pattern.fullmatch(row['archivo'])
                if match and int(match[1]) < len(original):
                    candidates[row['id_imagen']] = (original[int(match[1])], transform, ())
        if 'enlace_web' not in source:
            continue
        originals = sorted(by_source[source['enlace_web']], key=lambda row: row['ruta_relativa'])[:40]
        rng = random.Random(42)
        params = [(rng.uniform(0.4, 0.9), index % 3 == 0, rng.randint(60, 85)) for index in range(len(originals))]
        for row in collection:
            match = re.fullmatch(r'web_fundus_(\d+)\.jpg', row['archivo'])
            if match and int(match[1]) < len(originals):
                number = int(match[1])
                candidates[row['id_imagen']] = (originals[number], 'emular_web',
                                                (*params[number], row.get('particion') == 'filtered'))
    return candidates


def reproduce_crop_and_color(image):
    """Replica el cálculo de 07 para verificar el parentesco, sin ejecutar sus escrituras."""
    width, height = image.size
    thumb = np.asarray(image.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
    y, x = np.ogrid[:128, :128]
    central = ((x - 64)**2 + (y - 64)**2) < 56**2
    response = gaussian_filter((thumb[:, :, 0] * 0.70 + thumb[:, :, 1] * 0.30) * central, sigma=4)
    cy, cx = np.unravel_index(np.argmax(response), response.shape)
    side = int(min(width, height) * 0.55)
    left = min(max(0, int(cx / 128 * width) - side // 2), width - side)
    top = min(max(0, int(cy / 128 * height) - side // 2), height - side)
    resized = image.crop((left, top, left + side, top + side)).resize((512, 512), Image.Resampling.BICUBIC)
    pixels = np.asarray(resized, dtype=np.float32)
    means, deviations = (165, 85, 32), (28, 26, 14)
    for channel in range(3):
        values = pixels[:, :, channel]
        pixels[:, :, channel] = (values - values.mean()) / (values.std() + 1e-5) * deviations[channel] + means[channel]
    return Image.fromarray(pixels.clip(0, 255).astype('uint8'))


def verify_pair(row, candidate):
    if not candidate:
        return {'estado_par': 'sin_candidato'}
    original, transform, params = candidate
    result = {'candidato_original': original['id_imagen'], 'transformacion_par': transform}
    if not row.get('sha256_pixeles') or not original.get('sha256_pixeles'):
        return result | {'estado_par': 'archivo_no_legible'}
    with Image.open(original['path']) as stored:
        image = stored.convert('RGB')
    quality = 95
    if transform == 'emular_web':
        factor, blur, quality, resave = params
        width, height = image.size
        image = image.resize((int(width * factor), int(height * factor)), Image.Resampling.BILINEAR)
        image = image.resize((224, 224), Image.Resampling.BICUBIC)
        if blur:
            image = image.filter(ImageFilter.GaussianBlur(radius=0.8))
    elif transform == 'recortar_color':
        image = reproduce_crop_and_color(image)
    else:
        image = image.resize((512, 512), Image.Resampling.BICUBIC)
    encoded = io.BytesIO()
    image.save(encoded, 'JPEG', quality=quality)
    if transform == 'emular_web' and resave:
        filtered = io.BytesIO()
        encoded.seek(0)
        with Image.open(encoded) as decoded:
            decoded.save(filtered, 'JPEG')
        encoded = filtered
    if hashlib.sha256(encoded.getbuffer()).hexdigest() == row['sha256_archivo']:
        return result | {'estado_par': 'verificado_bytes', 'id_original': original['id_imagen'], 'error_rgb_par': 0.0}
    encoded.seek(0)
    with Image.open(row['path']) as observed, Image.open(encoded) as predicted:
        if observed.size != predicted.size:
            return result | {'estado_par': 'dimensiones_no_coinciden'}
        residual = np.abs(np.asarray(observed.convert('RGB'), dtype=np.int16) - np.asarray(predicted, dtype=np.int16))
    error = float(residual.mean())
    result.update(error_rgb_par=error)
    if error <= 1.0 and np.quantile(residual, 0.95) <= 3:
        return result | {'estado_par': 'verificado_pixeles_aproximados', 'id_original': original['id_imagen']}
    return result | {'estado_par': 'prediccion_no_coincide'}
