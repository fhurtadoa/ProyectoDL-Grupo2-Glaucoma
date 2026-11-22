"""Detecta coincidencias visuales conservadoras con correspondencias geométricas."""

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image


def features(row):
    with Image.open(row['path']) as image:
        image = image.convert('RGB')
        image.thumbnail((320, 320), Image.Resampling.BICUBIC)
        luminance = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=150, contrastThreshold=.016).detectAndCompute(luminance, None)
    if descriptors is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32)
    positions = np.array([key.pt for key in keypoints], dtype=np.float32)
    return positions, np.sqrt(descriptors / np.maximum(1e-9, descriptors.sum(axis=1, keepdims=True)))


def geometric_support(first, second, matcher):
    points_a, desc_a = first
    points_b, desc_b = second
    matches = [pair[0] for pair in matcher.knnMatch(desc_a, desc_b, k=2) if len(pair) == 2
               and pair[0].distance < .76 * pair[1].distance and pair[0].distance < .44]
    distinct = {}
    for match in sorted(matches, key=lambda item: item.distance):
        distinct.setdefault(match.trainIdx, match)
    matches = list(distinct.values())
    if len(matches) < 12:
        return None
    a = np.array([points_a[match.queryIdx] for match in matches])
    b = np.array([points_b[match.trainIdx] for match in matches])
    matrix, support = cv2.estimateAffinePartial2D(a, b, ransacReprojThreshold=2.6, maxIters=1000)
    if support is None:
        return None
    inliers = support[:, 0] > 0
    n = int(inliers.sum())
    if n < 12 or n / len(matches) < .75:
        return None
    extent = min(np.ptp(a[inliers], axis=0).min(), np.ptp(b[inliers], axis=0).min())
    if extent < 45:
        return None
    return n, len(matches), float(extent), matrix


def geometric_matches(rows, workers):
    rows = sorted((row for row in rows if row['comparar'] and row.get('sha256_pixeles')),
                  key=lambda row: row['id_imagen'])
    if len(rows) < 2:
        return []
    cv2.setRNGSeed(42)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        models = list(pool.map(features, rows))
    descriptors = np.concatenate([model[1] for model in models])
    if len(descriptors) < 12:
        return []
    owners = np.repeat(np.arange(len(rows)), [len(model[1]) for model in models])
    index = cv2.flann_Index(descriptors, dict(algorithm=1, trees=4))
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    found = {}
    for i, model in enumerate(models):
        _, own_descriptors = model
        if len(own_descriptors) < 12:
            continue
        ids, distances = index.knnSearch(own_descriptors, 12, params=dict(checks=50))
        who = owners[ids]
        votes = np.where((who != i) & (distances < .20), 1 / (distances + .035), 0)
        scores = np.bincount(who.ravel(), weights=votes.ravel(), minlength=len(rows))
        for j in np.argsort(scores, kind='stable')[-8:][::-1]:
            if scores[j] < 13 or len(models[j][1]) < 12:
                continue
            support = geometric_support(model, models[j], matcher)
            if support is None:
                continue
            n, total, extent, matrix = support
            a, b = sorted((i, j))
            old = found.get((a, b))
            if old and (n, n / total) <= (old['n_inliers'], old['n_inliers'] / old['n_matches']):
                continue
            if i != a:
                matrix = cv2.invertAffineTransform(matrix)
            left, right = rows[a], rows[b]
            found[(a, b)] = {'id_a': left['id_imagen'], 'id_b': right['id_imagen'],
                             'fuente_a': left['coleccion'], 'fuente_b': right['coleccion'],
                             'particion_a': left.get('particion'), 'particion_b': right.get('particion'),
                             'etiqueta_a': left.get('etiqueta'), 'etiqueta_b': right.get('etiqueta'),
                             'mismos_pixeles': left['sha256_pixeles'] == right['sha256_pixeles'],
                             'n_inliers': n, 'n_matches': total, 'extension_minima_px': extent,
                             **{f'afin_{y}{x}': float(matrix[y, x]) for y in range(2) for x in range(3)}}
    return [found[key] for key in sorted(found)]
