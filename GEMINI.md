# Memoria persistente del proyecto (ProyectoDL-Grupo2-Glaucoma)

Contexto y conclusiones verificadas de los experimentos `03`, `04` y `05`. Para el detalle completo con tablas, consultar [Entrega Final/CONCLUSIONES_AI.md](Entrega%20Final/CONCLUSIONES_AI.md).

## Tesis y conclusiones principales
- **Tesis central:** La generalización limitada de `ConvNeXt-Tiny` en glaucoma no se debe a falta de capacidad de la arquitectura, sino al acoplamiento entre: (1) heterogeneidad geométrica/fotométrica de adquisición (`03`), (2) esquema de supervisión clínica (`04`) y (3) traslado rígido del umbral operativo $\tau$ (`04`–`05`).
- **Exp. 03 (Características):** `ACRIMA` y `RIM-ONE DL` son recortes papilares (`100 %` área visible, `0 %` borde negro); `EyePACS REAL` es campo amplio (`21,76 %` fondo casi negro, luma mediana `77` vs. `121` en `Moondream`).
- **Exp. 04 (Trinario vs. Binario, origen único `Moondream`):** `Moondream Staged` es el único dataset con estadios (`264 / 115 / 125` en `train`). La proyección `3→2` supera consistentemente a `2→2` en `Moondream` (`AUROC 0,980` vs. `0,954`), `ACRIMA` (`0,740` vs. `0,626`, IC 95 % `[+9,0, +14,1 pp]`) y `RIM-ONE DL` (`0,815` vs. `0,780`, IC `[+0,7, +6,4 pp]`) bajo las tres reglas de umbral (`Espec. val ≥ 0,95`, `Youden` y $\tau = 0{,}50$), pero ningún modelo entrenado solo en `Moondream` generaliza a `EyePACS REAL` (`Sens ≤ 15 %`).
- **Exp. 05 (LODO binario multi-semilla: 4 folds × 2 brazos × 3 semillas `42, 123, 777` = 24 modelos):**
  - `EyePACS REAL` ($n=200$, ciego): `Con aumento` eleva `AUROC` de `0,655±0,019` a `0,809±0,014` (`+15,4 pp`, ensamble IC 95 % `[+10,9, +20,3 pp]`) y `F1` de `0,343` a `0,618` (`+27,5 pp`).
  - `Moondream Staged` ($n=706$, ciego): `F1` sube de `0,519` a `0,580` (`+6,1 pp`) y `AUROC` de `0,556` a `0,605` (`+4,8 pp`).
  - `ACRIMA` ($n=703$, ciego): empate en `F1` (`0,625` vs. `0,628`), pero mejora en ordenamiento (`AUROC 0,686 → 0,730`, IC `[+3,0, +6,6 pp]`).
  - `RIM-ONE DL` ($n=447$, ciego): efecto neutro (`F1 0,484` vs. `0,485`; `AUROC 0,671` vs. `0,677`).
- **Regla de umbral ($\tau$) y sensibilidad:** el corte principal $\tau \in [0{,}20, 0{,}99]$ se calibra **solo en validación** maximizando $0{,}5\cdot\text{F1 macro} + 0{,}5\cdot\text{Recall}$ sujeto a `especificidad ≥ 0,95` (empujando $\tau$ a `0,93–0,99` en varios folds LODO). En `EyePACS REAL + aumento`, `Youden (val)` ($\tau = 0{,}623$) o $\tau = 0{,}50$ reducen `TFN` de `64,7 %` a `36,0 %–29,3 %` (`F1 = 0,715–0,721`, `Espec = 79,7 %–73,7 %`).

## Reglas del repositorio
- Dataset limpio sin fuga: `utilidades/dataset_auditado.csv` (`2.056` imágenes elegibles).
- No modificar `02_experimento_seleccion_datasets_y_modelos/Comprobacion_mejor_modelo.ipynb` ni guías `.docx` de compañeros sin avisar.
- No usar `"maligno"` / `"Malignant"` en código o texto vigente; usar `AumentoEscalaFotometrico` / `Con aumento`.
