"""Distribuciones empíricas y muestra visual con contorno estimado superpuesto."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from metricas_retinografias import retinal_mask, thumbnail


PANELS = {
    'fotometria': [('brillo_mediano_yn', 'Mediana del brillo [0–255]'), ('contraste_de_yn', 'Contraste (DE del brillo)'),
                   ('saturacion_media_pct', 'Saturación HSV [%]'), ('r_media', 'Media del rojo [0–255]'),
                   ('g_media', 'Media del verde [0–255]'), ('b_media', 'Media del azul [0–255]')],
    'resolucion_nitidez': [('ancho_px', 'Ancho almacenado [px]'), ('alto_px', 'Alto almacenado [px]'),
                          ('laplaciano_nativo', 'Varianza del Laplaciano · nativo'),
                          ('laplaciano_lado512', 'Varianza del Laplaciano · lado largo 512 px')],
    'encuadre': [('negro_pct', 'Píxeles casi negros [%]'), ('region_pct', 'Área visible [%]'),
                 ('centro_x', 'Centro X [0–1]'), ('centro_y', 'Centro Y [0–1]'),
                 ('bordes_periferia_pct', 'Bordes Canny en la periferia [%]'),
                 ('caida_radial_yn', 'Brillo interior − periférico · circular')],
}


def palette(groups):
    colors = plt.get_cmap('tab10').colors if len(groups) <= 10 else plt.get_cmap('tab20').resampled(len(groups)).colors
    return dict(zip(groups, colors))


def ecdf(axis, groups, field, colors):
    series = {label: np.sort([row[field] for row in rows if row.get(field) is not None])
              for label, rows in groups.items()}
    series = {label: values for label, values in series.items() if len(values)}
    axis.set_ylim(-0.025, 1.025)
    axis.grid(alpha=0.22)
    if not series:
        axis.text(.5, .5, 'Sin valores evaluables', ha='center', va='center', transform=axis.transAxes)
        return
    smallest = min(values[0] for values in series.values())
    largest = max(values[-1] for values in series.values())
    padding = max(.025 * (largest - smallest), .002 * max(abs(largest), 1))
    lower, upper = smallest - padding, largest + padding
    for label, values in series.items():
        axis.step(np.r_[lower, values, upper], np.r_[0, np.arange(1, len(values) + 1) / len(values), 1],
                  where='post', color=colors[label], linewidth=1.6, label=label)
    axis.set_xlim(lower, upper)


def distributions(groups, destination):
    colors = palette(groups)
    for name, panels in PANELS.items():
        fig, axes = plt.subplots(len(panels) // 2, 2, figsize=(11, 2.65 * len(panels) / 2 + 0.8), squeeze=False)
        for ax, (field, title) in zip(axes.flat, panels):
            ecdf(ax, groups, field, colors)
            ax.set_xlabel(title)
            if field.startswith('laplaciano'):
                ax.set_xscale('symlog', linthresh=1)
                ax.set_xlim(left=0)
            if field in {'centro_x', 'centro_y'}:
                ax.set_xlim(0, 1)
            if field in {'negro_pct', 'region_pct', 'saturacion_media_pct'}:
                ax.set_xlim(0, 100)
        for ax in axes[:, 0]:
            ax.set_ylabel('Fracción acumulada')
        handles = [plt.Line2D([], [], color=colors[label], linewidth=2, label=f'{label} (n={len(rows)})')
                   for label, rows in groups.items()]
        fig.legend(handles=handles, loc='lower center', ncols=min(3, max(1, len(handles))), frameon=False)
        fig.tight_layout(rect=(0, min(0.24, 0.04 + 0.025 * len(handles)), 1, 1))
        fig.savefig(destination / f'{name}.png', dpi=155)
        plt.close(fig)


def paired_plot(groups, destination):
    if not groups:
        return
    fields = [('brillo_mediano_yn', 'Brillo mediano'), ('saturacion_media_pct', 'Saturación [pp]'),
              ('region_pct', 'Área visible [pp]'), ('laplaciano_lado512', 'Varianza del Laplaciano · lado 512')]
    colors = palette(groups)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2))
    for ax, (field, title) in zip(axes.flat, fields):
        ecdf(ax, groups, field, colors)
        ax.axvline(0, color='0.4', linestyle=':', linewidth=1)
        lower, upper = ax.get_xlim()
        ax.set_xlim(min(lower, 0), max(upper, 0))
        if field.startswith('laplaciano'):
            ax.set_xscale('symlog', linthresh=1)
        ax.set_xlabel(title)
    for ax in axes[:, 0]:
        ax.set_ylabel('Fracción acumulada')
    fig.suptitle('Diferencia por par · derivada menos imagen fuente local')
    handles = [plt.Line2D([], [], color=colors[label], linewidth=2, label=f'{label} (n={len(rows)})')
               for label, rows in groups.items()]
    fig.legend(handles=handles, loc='lower center', ncols=min(3, len(handles)), frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, .95))
    fig.savefig(destination / 'diferencias_pareadas.png', dpi=155)
    plt.close(fig)


def visual_examples(groups, destination):
    criteria = ['Mediana de brillo', 'Brillo bajo (p01)', 'Brillo alto (p99)', 'Nitidez baja',
                'Caso difícil de región', 'Texto candidato']
    selected = []
    if not groups:
        return selected
    fig, axes = plt.subplots(len(groups), len(criteria), figsize=(13, len(groups) * 1.98 + .7), squeeze=False)
    for (label, rows), axis_row in zip(groups.items(), axes):
        used = set()
        for col, (criterion, ax) in enumerate(zip(criteria, axis_row)):
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if label == next(iter(groups)):
                ax.set_title(criterion, fontsize=8)
            row = pick_example(rows, criterion, used)
            if row is None:
                ax.text(.5, .45, 'Sin caso', va='center', ha='center', fontsize=8, color='0.43', transform=ax.transAxes)
                continue
            used.add(row['id_imagen'])
            with Image.open(row['path']) as original:
                rgb = thumbnail(np.asarray(original.convert('RGB')))
            ax.imshow(rgb, interpolation='nearest')
            if row['tipo'] in {'retinografia', 'armonizada', 'emulacion_web'}:
                mask, _ = retinal_mask(rgb)
                if mask is not None and mask.any() and not mask.all():
                    ax.contour(mask, levels=[0.5], colors=['#53f2db'], linewidths=0.55)
            source = row.get('fuente_indice') or row['coleccion']
            ax.set_xlabel(f"{row['archivo']}\n{row['region']}", fontsize=6, labelpad=2)
            selected.append({'fila_mosaico': label, 'criterio': criterion, 'fuente': source,
                             'id_imagen': row['id_imagen'], 'region': row['region'],
                             'grupos_texto_candidato': row.get('grupos_texto_candidato')})
    fig.suptitle('Muestra visual · el contorno turquesa indica la región estimada', fontsize=13)
    fig.tight_layout(rect=(0.012, 0, 1, .98))
    fig.savefig(destination / 'mosaico_revision.png', dpi=175)
    plt.close(fig)
    return selected


def pick_example(rows, criterion, used):
    candidates = [row for row in rows if row.get('sha256_pixeles') and row['id_imagen'] not in used]
    if not candidates:
        return None
    if criterion == 'Caso difícil de región':
        return min((row for row in candidates if row.get('estado') in {'region_no_evaluable', 'solo_archivo'}),
                   key=lambda row: row['id_imagen'], default=None)
    if criterion == 'Texto candidato':
        return max((row for row in candidates if row.get('grupos_texto_candidato', 0) > 0),
                   key=lambda row: row['grupos_texto_candidato'], default=None)
    field, quantile = {'Mediana de brillo': ('brillo_mediano_yn', .5), 'Brillo bajo (p01)': ('brillo_mediano_yn', .01),
                       'Brillo alto (p99)': ('brillo_mediano_yn', .99), 'Nitidez baja': ('laplaciano_lado512', .01)}[criterion]
    valid = [row for row in candidates if row.get(field) is not None]
    if valid:
        target = np.quantile([row[field] for row in valid], quantile)
        return min(valid, key=lambda row: (abs(row[field] - target), row['id_imagen']))
    return None


def geometric_examples(matches, rows, destination):
    by_source = {}
    for match in matches:
        if match['mismos_pixeles']:
            continue
        key = tuple(dict.fromkeys((match['fuente_a'], match['fuente_b'])))
        by_source.setdefault(key, []).append(match)
    if not by_source:
        return
    examples = []
    for sources, pairs in by_source.items():
        pairs.sort(key=lambda pair: (pair['n_inliers'], pair['extension_minima_px'], pair['id_a']))
        for pair in (pairs if len(pairs) <= 2 else [pairs[0], pairs[-1]]):
            examples.append((' / '.join(sources), pair))
    images = {row['id_imagen']: row for row in rows}
    fig, axes = plt.subplots(len(examples), 2, figsize=(6.5, 2.45 * len(examples) + .5), squeeze=False)
    for (source, pair), axis_row in zip(examples, axes):
        for side, ax in zip(('a', 'b'), axis_row):
            row = images[pair[f'id_{side}']]
            with Image.open(row['path']) as image:
                ax.imshow(thumbnail(np.asarray(image.convert('RGB')), size=300))
            ax.set_title(f"{row['archivo']} · {row.get('particion') or 'sin partición'}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        axis_row[0].set_ylabel(f"{source}\n{pair['n_inliers']} puntos", fontsize=8)
    fig.suptitle('Coincidencias geométricas entre archivos de píxeles distintos')
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(destination / 'mosaico_coincidencias.png', dpi=145)
    plt.close(fig)
