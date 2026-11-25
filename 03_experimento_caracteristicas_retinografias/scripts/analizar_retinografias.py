"""Extrae y resume características de imágenes del repositorio, sin modificar las fuentes."""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import matplotlib
import numpy as np
import PIL
import scipy
from PIL import Image

from coincidencias_retinografias import geometric_matches
from graficas_retinografias import distributions, geometric_examples, paired_plot, visual_examples
from metricas_retinografias import disk_area, measure
from procedencia_retinografias import inventory, manifest_rows, pair_candidates, relative, verify_pair


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name('fuentes_retinografias.json')
DEFAULT_DESTINATION = ROOT / '03_experimento_caracteristicas_retinografias/datos'
METRICS = ('ancho_px', 'alto_px', 'aspecto', 'negro_pct', 'rgb_identico_pct', 'alfa_no_cero_pct',
           'brillo_medio_yn', 'brillo_mediano_yn', 'contraste_de_yn',
           'r_media', 'r_de', 'g_media', 'g_de', 'b_media', 'b_de', 'saturacion_media_pct',
           'laplaciano_nativo', 'laplaciano_lado512', 'region_pct', 'centro_x', 'centro_y',
           'diametro_visible_equiv_px', 'extension_x', 'extension_y', 'negro_borde_pct',
           'bordes_periferia_pct', 'caida_radial_yn', 'grupos_texto_candidato', 'disco_area_imagen_pct')
IMAGE_COLUMNS = ('id_imagen', 'coleccion', 'tipo', 'ruta_relativa', 'archivo', 'particion', 'etiqueta',
                 'esquema_etiqueta', 'origen_etiqueta', 'etiqueta_carpeta', 'etiqueta_indice', 'particion_indice',
                 'estado_manifiesto', 'fuente_indice', 'formato', 'modo_original', 'camara_documentada', 'camara_exif',
                 'estado', 'motivo', 'region', 'estado_disco', 'motivo_disco', 'ruta_mascara_disco',
                 'sha256_archivo', 'sha256_pixeles', 'n_mismo_pixel', 'id_canonico_pixeles', 'conflicto_etiqueta_duplicado',
                 'id_grupo_geometrico', 'n_grupo_geometrico', 'conflicto_etiqueta_grupo',
                 'en_resumen_grupo', 'en_comparacion_solo_exactos', 'en_comparacion_fuentes',
                 'candidato_original', 'id_original', 'transformacion_par',
                 'estado_par', 'error_rgb_par') + METRICS
STATS = ('metrica', 'n_total', 'n_validas', 'n_faltantes', 'p05', 'p25', 'mediana', 'p75', 'p95', 'ric')


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n', extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f'{value:.8g}' if isinstance(value, (float, np.floating)) else value
                             for key, value in row.items() if key in fields})


def select_pilot(rows, count, candidates):
    rng = random.Random(42)
    by_source = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group = (row['tipo'], row.get('particion', ''), row.get('etiqueta', ''), row.get('fuente_indice', ''))
        by_source[row['coleccion']][group].append(row['id_imagen'])
    selected = set()
    for strata in by_source.values():
        for values in strata.values():
            rng.shuffle(values)
        offset = 0
        while offset < count:
            available = [values.pop() for values in strata.values() if values]
            if not available:
                break
            selected.update(available[:count - offset])
            offset += len(available)
    selected.update(candidates[key][0]['id_imagen'] for key in selected.copy() if key in candidates)
    return [row for row in rows if row['id_imagen'] in selected]


def extract(row, annotations):
    try:
        row.update(measure(row['path'], row['tipo']))
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        row.update(estado='no_legible', motivo=f'{type(error).__name__}: {error}', region='no_evaluable')
    row['estado_disco'] = 'sin_anotacion_verificada'
    annotation = annotations.get(row['id_imagen'])
    if annotation and row.get('sha256_pixeles') and row['tipo'] in {'retinografia', 'armonizada', 'emulacion_web'}:
        row['ruta_mascara_disco'] = annotation
        try:
            row['disco_area_imagen_pct'] = disk_area(ROOT / annotation, row)
            row['estado_disco'] = 'mascara_verificada_en_manifiesto'
        except (OSError, ValueError) as error:
            row.update(estado_disco='mascara_no_evaluable', motivo_disco=str(error))
    return row


def grouped(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field, '') for field in fields)].append(row)
    return groups


def analytical_group(row):
    if row['tipo'] == 'armonizada':
        return row.get('fuente_indice') or 'sin identificar en el índice'
    if row['tipo'] in {'emulacion_web', 'distractor_web_sintetico'}:
        return row['particion']
    return ''


def duplicate_audit(rows):
    by_pixel = grouped((row for row in rows if row.get('sha256_pixeles')), ('sha256_pixeles',))
    order = {row['id_imagen']: idx for idx, row in enumerate(rows)}
    for row in rows:
        row['grupo_derivado'] = analytical_group(row)
        row['en_resumen_grupo'] = row.get('estado_manifiesto') != 'duplicado_id_fuera_ruta_indexada'
        row['en_comparacion_fuentes'] = row['comparar'] and row['en_resumen_grupo']
    duplicate_rows = []
    for members in by_pixel.values():
        canonical = min(members, key=lambda row: (not row['comparar'], not row['en_resumen_grupo'],
                                                  row.get('particion') != 'raw', order[row['id_imagen']]))
        seen_groups = set()
        for row in sorted(members, key=lambda item: (not item['en_resumen_grupo'], item.get('particion') != 'raw',
                                                     order[item['id_imagen']])):
            key = (row['coleccion'], row['tipo'], row['grupo_derivado'])
            if row['en_resumen_grupo']:
                row['en_resumen_grupo'] = key not in seen_groups
                seen_groups.add(key)
            local_labels = {item.get('etiqueta') for item in members if item['coleccion'] == row['coleccion'] and item.get('etiqueta')}
            row.update(n_mismo_pixel=len(members), id_canonico_pixeles=canonical['id_imagen'],
                       conflicto_etiqueta_duplicado=len(local_labels) > 1,
                       en_comparacion_fuentes=row['comparar'] and row is canonical)
            if row['comparar']:
                row['en_resumen_grupo'] = row['en_comparacion_fuentes']
            if len(members) > 1:
                duplicate_rows.append(row)
    return duplicate_rows


def relate(row, candidates):
    result = verify_pair(row, candidates.get(row['id_imagen']))
    if row['tipo'] in {'armonizada', 'emulacion_web'}:
        row.update(result)
    if row['tipo'] == 'emulacion_web' and row.get('id_original'):
        original = candidates[row['id_imagen']][0]
        row.update(etiqueta=original.get('etiqueta'), origen_etiqueta='heredada del par verificado de ACRIMA',
                   esquema_etiqueta=original['esquema_etiqueta'])
    return row


def consolidate_geometric_matches(rows, matches):
    originals = {row['id_imagen']: row for row in rows if row['comparar']}
    priority = {identifier: index for index, identifier in enumerate(originals)}
    parents = {identifier: identifier for identifier in originals}

    def representative(identifier):
        while parents[identifier] != identifier:
            parents[identifier] = parents[parents[identifier]]
            identifier = parents[identifier]
        return identifier

    def merge(first, second):
        chosen, other = sorted((representative(first), representative(second)), key=priority.get)
        parents[other] = chosen

    for row in originals.values():
        canonical = row.get('id_canonico_pixeles')
        if canonical in originals:
            merge(row['id_imagen'], canonical)
    for match in matches:
        merge(match['id_a'], match['id_b'])
    groups = defaultdict(list)
    for identifier, row in originals.items():
        groups[representative(identifier)].append(row)
    for canonical, members in groups.items():
        labels = defaultdict(set)
        for row in members:
            if row.get('etiqueta'):
                labels[row['esquema_etiqueta']].add(row['etiqueta'])
        conflict = any(len(values) > 1 for values in labels.values())
        for row in members:
            keep = row['id_imagen'] == canonical
            row.update(id_grupo_geometrico=canonical, n_grupo_geometrico=len(members), conflicto_etiqueta_grupo=conflict,
                       en_comparacion_solo_exactos=row['en_comparacion_fuentes'], en_resumen_grupo=keep,
                       en_comparacion_fuentes=keep)


def pairs(rows):
    originals = {row['id_imagen']: row for row in rows}
    records, seen = [], set()
    eligible = [row for row in rows if row['en_resumen_grupo'] and row.get('id_original')]
    eligible.sort(key=lambda row: (row.get('particion') == 'filtered',
                                  not originals[row['id_original']]['en_comparacion_fuentes'], row['id_imagen']))
    for row in eligible:
        original = originals[row['id_original']]
        group = original.get('id_grupo_geometrico') or original['sha256_pixeles']
        key = (row['coleccion'], row['transformacion_par'], group)
        if key in seen:
            continue
        seen.add(key)
        record = {'coleccion': row['coleccion'], 'fuente': original['coleccion'], 'id_original': original['id_imagen'],
                  'id_derivada': row['id_imagen'], 'transformacion': row['transformacion_par'],
                  'estado_par': row['estado_par'], 'error_rgb_par': row['error_rgb_par'],
                  'grupo_original': original.get('id_grupo_geometrico'),
                  'particion_original': original.get('particion'), 'etiqueta_original': original.get('etiqueta'),
                  'particion_derivada': row.get('particion'), 'etiqueta_derivada': row.get('etiqueta')}
        for metric in METRICS:
            before, after = original.get(metric), row.get(metric)
            if before is not None and after is not None:
                record[metric] = after - before
        records.append(record)
    return records


def statistics(groups, fields):
    for key, rows in groups.items():
        for metric in METRICS:
            values = [row[metric] for row in rows if row.get(metric) is not None and np.isfinite(row[metric])]
            result = dict(zip(fields, key)) | {'metrica': metric, 'n_total': len(rows),
                                              'n_validas': len(values), 'n_faltantes': len(rows) - len(values)}
            if values:
                p05, p25, median, p75, p95 = np.quantile(values, [.05, .25, .5, .75, .95])
                result.update(p05=p05, p25=p25, mediana=median, p75=p75, p95=p95, ric=p75 - p25)
            yield result


def counts(rows, field):
    counter = Counter(row.get(field) or 'sin dato' for row in rows)
    return json.dumps(dict(sorted(counter.items())), ensure_ascii=False)


def inventory_report(config, found, processed, audit):
    found_groups = grouped(found, ('coleccion', 'tipo'))
    processed_groups = grouped(processed, ('coleccion', 'tipo'))
    sources = {source['id']: source for source in config['fuentes']}
    expected = list(found_groups) + [(name, source['tipo']) for name, source in sources.items()
                                    if not any(key[0] == name for key in found_groups)]
    for name, kind in expected:
        source = sources.get(name, {})
        all_rows, examined = found_groups.get((name, kind), []), processed_groups.get((name, kind), [])
        readable = [row for row in examined if row.get('sha256_pixeles')]
        manifest = audit.get(name, {})
        yield {'coleccion': name, 'tipo': kind, 'ruta': source.get('ruta', config['raiz_datos']),
               'disponibilidad': 'local' if all_rows else 'ruta_sin_imagenes',
               'representacion': source.get('representacion'), 'procedencia': source.get('procedencia'),
               'manifiesto': source.get('manifiesto', {}).get('ruta'), 'camara_documentada': source.get('camara_documentada'),
               'n_encontradas': len(all_rows), 'n_inspeccionadas': len(examined), 'n_legibles': len(readable),
               'n_con_etiqueta': sum(bool(row.get('etiqueta')) for row in all_rows),
               'n_resumen_grupo': sum(row['en_resumen_grupo'] for row in examined),
               'n_comparacion_fuentes': sum(row['en_comparacion_fuentes'] for row in examined),
               'n_region_evaluable': sum(row.get('estado') == 'completa' for row in examined),
               'n_disco_evaluable': sum(row.get('disco_area_imagen_pct') is not None for row in examined),
               'n_pares_archivos_verificados': sum(bool(row.get('id_original')) for row in examined),
               'filas_manifiesto': manifest.get('filas_manifiesto'),
               'filas_manifiesto_sin_archivo': manifest.get('filas_sin_archivo_en_ruta_esperada'),
               'particiones': counts(all_rows, 'particion'), 'etiquetas': counts(all_rows, 'etiqueta'),
               'formatos': counts(readable, 'formato'), 'modos': counts(readable, 'modo_original'),
               'estados': counts(examined, 'estado'), 'regiones': counts(examined, 'region')}
    for source in config.get('ausentes', []):
        if source['id'] not in sources:
            yield {'coleccion': source['id'], 'procedencia': source['procedencia'], 'disponibilidad': 'mencionada_sin_archivos',
                   'n_encontradas': 0, 'n_inspeccionadas': 0, 'n_legibles': 0}


def publish_results(destination, config, found, rows, audit, duplicates, matches, paired, limit):
    comparisons = {key[0]: value for key, value in grouped(
        (row for row in rows if row['en_comparacion_fuentes']), ('coleccion',)).items()}
    group_fields = ('coleccion', 'tipo', 'grupo_derivado')
    groups = grouped((row for row in rows if row['en_resumen_grupo']), group_fields)
    exact_only = grouped((row for row in rows if row.get('en_comparacion_solo_exactos')), group_fields)
    stratify = {source['id'] for source in config['fuentes'] if source.get('estratificar')}
    class_fields = ('coleccion', 'esquema_etiqueta', 'etiqueta')
    classes = grouped((row for row in rows if row['en_comparacion_fuentes'] and row['coleccion'] in stratify
                       and row.get('etiqueta') and not row.get('conflicto_etiqueta_grupo')), class_fields)
    pair_fields = ('coleccion', 'fuente', 'transformacion')
    paired_groups = grouped(paired, pair_fields)
    inventory_rows = list(inventory_report(config, found, rows, audit))

    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination) as temporary:
        target = Path(temporary)
        write_csv(target / 'inventario.csv', inventory_rows, tuple(inventory_rows[0]))
        write_csv(target / 'por_imagen.csv', rows, IMAGE_COLUMNS)
        write_csv(target / 'resumen_por_dataset.csv', statistics(groups, group_fields), group_fields + STATS)
        write_csv(target / 'resumen_solo_duplicados_exactos.csv', statistics(exact_only, group_fields), group_fields + STATS)
        write_csv(target / 'resumen_por_clase.csv', statistics(classes, class_fields), class_fields + STATS)
        write_csv(target / 'duplicados.csv', duplicates, IMAGE_COLUMNS[:5] + (
            'particion', 'etiqueta', 'estado_manifiesto', 'sha256_archivo', 'sha256_pixeles', 'n_mismo_pixel',
            'id_canonico_pixeles', 'conflicto_etiqueta_duplicado', 'en_resumen_grupo', 'en_comparacion_fuentes'))
        match_fields = ('id_a', 'id_b', 'fuente_a', 'fuente_b', 'particion_a', 'particion_b', 'etiqueta_a', 'etiqueta_b',
                        'mismos_pixeles', 'n_inliers', 'n_matches', 'extension_minima_px',
                        'afin_00', 'afin_01', 'afin_02', 'afin_10', 'afin_11', 'afin_12')
        write_csv(target / 'coincidencias_geometricas.csv', matches, match_fields)
        pair_columns = pair_fields + ('id_original', 'id_derivada', 'grupo_original', 'estado_par', 'error_rgb_par', 'particion_original',
                                      'etiqueta_original', 'particion_derivada', 'etiqueta_derivada')
        write_csv(target / 'pares.csv', paired, pair_columns + METRICS)
        write_csv(target / 'resumen_pares.csv', statistics(paired_groups, pair_fields), pair_fields + STATS)
        distributions(comparisons, target)
        paired_plot({f'{source} → {collection}': values for (collection, source, _), values in paired_groups.items()}, target)
        mosaic_groups = {f'{name} · {derived}' if derived else name: values
                         for (name, kind, derived), values in groups.items()
                         if kind != 'distractor_web_sintetico' and (kind != 'emulacion_web' or derived == 'raw')}
        for (name, kind, derived), values in groups.items():
            if kind == 'distractor_web_sintetico' and derived == 'raw':
                mosaic_groups[f'{name} · distractores'] = values
        examples = visual_examples(mosaic_groups, target)
        write_csv(target / 'muestra_visual.csv', examples, ('fila_mosaico', 'criterio', 'fuente', 'id_imagen',
                                                            'region', 'grupos_texto_candidato'))
        geometric_examples(matches, rows, target)
        metadata = {'modo': 'piloto' if limit else 'completo', 'limite_piloto_por_coleccion': limit,
                    'n_archivos_encontrados': len(found), 'n_archivos_inspeccionados': len(rows),
                    'n_legibles': sum(bool(row.get('sha256_pixeles')) for row in rows),
                    'n_comparacion_fuentes': sum(len(group) for group in comparisons.values()),
                    'n_coincidencias_geometricas': len(matches), 'n_pares': len(paired), 'manifiestos': audit,
                    'versiones': {'python': sys.version.split()[0], 'numpy': np.__version__, 'opencv': cv2.__version__,
                                  'pillow': PIL.__version__, 'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__}}
        (target / 'ejecucion.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        for path in target.iterdir():
            os.replace(path, destination / path.name)
    print(json.dumps({key: metadata[key] for key in metadata if key not in {'manifiestos', 'versiones'}}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--salida', type=Path)
    parser.add_argument('--piloto', type=int, metavar='N', help='hasta N archivos por colección, más sus fuentes locales')
    parser.add_argument('--trabajadores', type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.trabajadores < 1 or (args.piloto is not None and args.piloto < 1):
        parser.error('--trabajadores y --piloto deben ser enteros positivos')
    destination = (args.salida or (DEFAULT_DESTINATION / 'piloto' if args.piloto else DEFAULT_DESTINATION)).resolve()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    inputs = [config['raiz_datos']] + [source['ruta'] for source in config['fuentes']]
    if any(destination.is_relative_to((ROOT / path).resolve()) for path in inputs):
        parser.error('--salida debe estar fuera de los directorios de imágenes')
    cv2.setNumThreads(1)
    found, audit = inventory(config, ROOT)
    if not found:
        parser.error('No se encontraron imágenes en las rutas configuradas')
    candidates = pair_candidates(found, config['fuentes'])
    rows = select_pilot(found, args.piloto, candidates) if args.piloto else found
    annotations = {relative((ROOT / entry['ruta_imagen']).resolve(), ROOT): entry['ruta_mascara']
                   for entry in manifest_rows(ROOT, config.get('mascaras_disco'))
                   if entry['verificada'].strip().lower() in {'1', 'true', 'sí', 'si'}}
    print(f'Extrayendo {len(rows)} de {len(found)} archivos.', flush=True)
    with ThreadPoolExecutor(max_workers=args.trabajadores) as workers:
        list(workers.map(partial(extract, annotations=annotations), rows))
        print(f'Verificando {sum(row["id_imagen"] in candidates for row in rows)} correspondencias candidatas.', flush=True)
        list(workers.map(partial(relate, candidates=candidates),
                         (row for row in rows if row['tipo'] in {'armonizada', 'emulacion_web'})))
    duplicates = duplicate_audit(rows)
    print('Buscando coincidencias geométricas en las colecciones de retinografías.', flush=True)
    matches = geometric_matches(rows, args.trabajadores)
    consolidate_geometric_matches(rows, matches)
    publish_results(destination, config, found, rows, audit, duplicates, matches, pairs(rows), args.piloto)


if __name__ == '__main__':
    main()
