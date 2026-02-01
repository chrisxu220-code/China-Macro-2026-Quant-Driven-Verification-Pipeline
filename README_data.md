# Data Registry & Ingestion (Task B)

This repo uses a **data registry** to make all raw inputs reproducible and auditable.

## What you edit
- `config.yaml` → `inputs.data_excel_path` (your local path, Downloads ok)
- `data/registry.yaml` → dataset metadata + parsing strategy

## What the pipeline produces
After running:
```bash
python -m src.run --config config.yaml
