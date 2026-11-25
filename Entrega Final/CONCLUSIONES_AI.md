# Conclusiones generales del estudio experimental (IA · síntesis verificada)

Síntesis consolidada a partir de los resultados ejecutados y auditados de los experimentos [03](../03_experimento_caracteristicas_retinografias/03_caracteristicas_de_retinografias.ipynb), [04](../04_experimento_trinario_vs_binario/01_convnext_trinario_vs_binario.ipynb) y [05](../05_experimento_data_augmentation/02_convnext_data_augmentation.ipynb) sobre el inventario limpio sin fuga de datos (`utilidades/dataset_auditado.csv`, `2.056` retinografías).

---

## Tesis central

> **La generalización limitada de `ConvNeXt-Tiny` en la detección automática de glaucoma no responde a falta de capacidad discriminativa de la arquitectura, sino al acoplamiento entre tres factores: la heterogeneidad geométrica y fotométrica de adquisición (`03`), el esquema de supervisión clínica (`04`) y el traslado rígido del umbral de decisión $\tau$ (`04`–`05`).**

---

## 1. Heterogeneidad física entre cohortes (`Exp. 03`)

- **Encuadre incompatible entre fuentes:** `ACRIMA` y `RIM-ONE DL` consisten en recortes centrados alrededor del disco óptico (`100 %` de área retiniana visible mediana y `0 %` de fondo negro), mientras que `EyePACS REAL` (`78,18 %` área visible, `21,76 %` fondo casi negro) y `EyePACS TRAIN` (`72,83 %`, `27,12 %` fondo negro) conservan el campo amplio completo con aro circular oscuro.
- **Dispersión fotométrica y nitidez:** el brillo mediano (luma BT.601) varía desde `77` en `EyePACS REAL` hasta `121 [65–171]` en `Moondream Staged`; la saturación media oscila entre `61,57 %` (`EyePACS REAL`) y `87,82 %` (`ACRIMA`); y el enfoque (Laplaciano a lado 512 px) cambia de `59,97` en `EyePACS REAL` a `5,11` en `ACRIMA`.
- **Efecto medido del preprocesamiento:** en `199` pares verificados de `EyePACS REAL`, el recorte/ajuste eleva la mediana de brillo (`+25` niveles), la saturación (`+19,23 pp`) y el área visible (`+21,05 pp`).
- **Implicación:** ninguna evaluación fuera de dominio debe interpretarse sin considerar la diferencia geométrica entre recortes papilares y retinografías de campo amplio.

---

## 2. Supervisión trinaria frente a binaria desde un origen estadificado (`Exp. 04`)

- **Delimitación metodológica:** `Moondream Staged` es la única cohorte local con los tres estadios (`Sano`, `Etapa inicial`, `Avanzado`; `264 / 115 / 125` en `train`). Por ello se fija como origen común para comparar `3→2` vs. `2→2` en igualdad de condiciones, reportando brevemente la estadificación directa (`3→3`) en `Moondream test` (`n = 95`, `QWK = 0,801`, `recall inicial = 0,476`, `recall avanzado = 0,840`).
- **Ventaja consistente de entrenar por estadio (`3→2`):**
  - `Moondream · test` ($n=95$): `F1 = 0,916` (`AUROC = 0,980`) vs. binario `0,852` (`AUROC = 0,954`).
  - `ACRIMA` ($n=703$): `F1 = 0,671` (`AUROC = 0,740`) vs. `0,598` (`AUROC = 0,626`, IC 95 % $\Delta\text{AUROC}$ `[+9,0, +14,1 pp]`).
  - `RIM-ONE DL` ($n=447$): `F1 = 0,735` (`AUROC = 0,815`) vs. `0,682` (`AUROC = 0,780`, IC 95 % $\Delta\text{AUROC}$ `[+0,7, +6,4 pp]`).
  - La ventaja de `3→2` se sostiene en las tres reglas de umbral evaluadas (`Espec. val ≥ 0,95`, `Youden` y $\tau = 0{,}50$).
- **Límite de dominio único:** ninguno de los dos modelos entrenados solo en `Moondream` generaliza por sí solo al campo amplio de `EyePACS REAL` (`Sensibilidad = 0,0 %–2,0 %` con $\tau$ calibrado; `9,0 %–15,0 %` con $\tau = 0{,}50$).

---

## 3. Efecto del aumento de escala y color en *Leave-One-Dataset-Out* binario multi-semilla (`Exp. 05`)

Evaluado con **4 *folds* LODO × 2 brazos (`Base` vs. `Con aumento`) × 3 semillas (`42`, `123`, `777`) = 24 modelos**, calibrando el corte principal $\tau \in [0{,}20, 0{,}99]$ solo en validación (`Espec. val ≥ 0,95`):

| Cohorte ciega LODO | $n$ | `Base` F1 / AUROC | `Con aumento` F1 / AUROC | $\Delta\text{F1}$ / $\Delta\text{AUROC}$ (`3 semillas`) | IC 95 % ensamble `AUROC` |
| --- | ---: | --- | --- | --- | --- |
| `EyePACS REAL` | 200 | `0,343±0,001` / `0,655±0,019` | **`0,618±0,039` / `0,809±0,014`** | **`+27,5 pp` / `+15,4 pp`** | `[+10,9, +20,3 pp]` |
| `Moondream Staged` | 706 | `0,519±0,031` / `0,556±0,054` | **`0,580±0,011` / `0,605±0,016`** | **`+6,1 pp` / `+4,8 pp`** | `[+3,5, +7,1 pp]` |
| `ACRIMA` | 703 | `0,625±0,024` / `0,686±0,026` | `0,628±0,084` / **`0,730±0,055`** | `+0,3 pp` / **`+4,4 pp`** | `[+3,0, +6,6 pp]` |
| `RIM-ONE DL` | 447 | `0,484±0,054` / `0,671±0,039` | `0,485±0,044` / `0,677±0,043` | `+0,1 pp` / `+0,6 pp` | `[-1,5, +5,4 pp]` |

- **El aumento de escala rescata los saltos geométricos y lumínicos sin dañar los recortes cerrados:** en `EyePACS REAL` (completamente ciego tras entrenar solo en cohortes recortadas/mixtas), `AumentoEscalaFotometrico` eleva el `AUROC` en **`+15,4 pp`** y el `F1` en **`+27,5 pp`**; también mejora `Moondream` (`+6,1 pp` F1) y el ordenamiento en `ACRIMA` (`+4,7 pp` `AUROC` en ensamble), mientras que sobre `RIM-ONE DL` es neutro.
- **Valor de usar 3 semillas:** promediar `42`, `123` y `777` demostró que aparentes caídas puntuales en una sola semilla correspondían a variabilidad estocástica de época temprana y no a un deterioro sistemático causado por el aumento.

---

## 4. Separar discriminación (`AUROC`) de punto operativo ($\tau$)

- Exigir `especificidad ≥ 0,95` en validación empuja el umbral congelado hacia valores muy conservadores (`0,93–0,99` en varios *folds* LODO).
- El análisis de sensibilidad a $\tau$ (`01_sensibilidad_umbral.csv` y `02_sensibilidad_umbral_lodo.csv`) demuestra que gran parte de los falsos negativos en dominios nuevos no provienen de incapacidad para separar clases, sino del corte estricto:
  - En `EyePACS REAL` con aumento (`AUROC = 0,809`), la `TFN` pasa de **`64,7 %`** bajo el corte conservador ($\tau \approx 0{,}937$) a **`36,0 %`** con `Youden (val)` ($\tau = 0{,}623$, `F1 = 0,715`, `Espec = 79,7 %`) y a **`29,3 %`** con $\tau = 0{,}50$ (`F1 = 0,721`, `Espec = 73,7 %`).

---

## 5. Implicación práctica para despliegue y trabajo futuro

1. **Un `AUROC` cercano a `0,98` en test interno no garantiza utilidad clínica externa:** es indispensable reportar evaluación multicéntrica ciega (LODO o cohortes externas auditadas sin cruces).
2. **Estrategia mínima antes de desplegar en otra cámara o centro:** combinar **invariancia de escala/encuadre en entrenamiento (o armonización de ROI papilar)** con **recalibración local del umbral $\tau$** según la prevalencia y tolerancia a falsos positivos/negativos del sitio destino.
3. **Limitaciones abiertas del trabajo:** ausencia de etiquetas reales de severidad en fuentes externas (`ACRIMA`, `RIM-ONE DL`, `EyePACS REAL`), falta de máscaras verificadas de disco/copa y ausencia de mapas de atribución espacial (`Grad-CAM`) para auditar qué regiones anatómicas usa la red en cada cohorte.
