# Flujo escalable para Google Colab Pro (A100 / G4) y ejecución local

Esta carpeta (`colab/`) encapsula todo el entrenamiento escalado sin modificar el código base del repositorio (`utilidades/`, `02_...`, `04_...`, `05_...`, `demo_glaucoma/`).

## Características principales
- **Particiones idénticas a Exp. `04` y `05`:** consume `utilidades/dataset_auditado.csv` (`2.056` imágenes sin fuga) y replica exactamente los splits y folds LODO (`SEED = 42`).
- **Modelos `timm` + fallback `torchvision`:** soporta cualquier backbone CNN (ej. `convnext_tiny.fb_in22k_ft_in1k`, `convnext_small.fb_in22k_ft_in1k_384`, `convnext_base.fb_in22k_ft_in1k_384`, `resnet50d`, `tf_efficientnetv2_m.in21k_ft_in1k`) y resolución configurable (`--img_size 224` o `384`).
- **Estrategias de ajuste fino (`--ft_strategy`):**
  - `lp`: Linear Probing (solo cabeza clasificadora).
  - `lp_ft`: Linear Probing inicial (`--lp_epochs`) seguido de Fine-Tuning diferencial (`--lr * --backbone_lr_mult` en backbone, `--lr` en cabeza).
  - `full_ft`: warmup de cabeza (`--lp_epochs`, por defecto `1`) + Fine-Tuning completo con `CosineAnnealingLR`.
- **Optimizado para A100/G4 y tolerante a desconexiones:**
  - `torch.amp.autocast('cuda', dtype=torch.bfloat16)` por defecto cuando la GPU soporta `bf16` (sin `GradScaler`).
  - Guardado incremental por época (`flush()` + `os.fsync()`) en `metrics_epoch.csv` y checkpoints atómicos `checkpoint_last.pt` / `checkpoint_best.pt`.
  - Si una corrida ya tiene `run_done.json` en `--output_dir`, se omite automáticamente; si se cortó a mitad de épocas, `--resume` (activo por defecto) continúa desde la última época guardada.

## Ejecución rápida por CLI

```bash
# 1. Prueba rápida local (smoke test)
python -m colab.run_suite --config colab/configs/smoke_test.yaml --data_dir . --output_dir ./colab_outputs

# 2. Experimento 05 LODO con un modelo específico
python -m colab.run --task exp05_lodo --model_name convnext_small.fb_in22k_ft_in1k_384 --img_size 384 --ft_strategy lp_ft --lp_epochs 5 --epochs 30 --data_dir /content/data_local --output_dir /content/drive/MyDrive/Glaucoma_Colab_Results

# 3. Suite completa YAML en Colab A100/G4
python -m colab.run_suite --config colab/configs/cnn_lodo_a100.yaml --data_dir /content/data_local --output_dir /content/drive/MyDrive/Glaucoma_Colab_Results

# 4. Horizonte extendido (150 épocas) y escalado de N (300, 500, 800, full)
python -m colab.run_suite --config colab/configs/cnn_horizon_double_descent.yaml --data_dir /content/data_local --output_dir /content/drive/MyDrive/Glaucoma_Colab_Results
```
