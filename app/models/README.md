# app/models/

This directory is where the real ONNX prompt-injection classifier lives
once you've exported one. It ships empty — the proxy runs perfectly well
without it, using the dependency-free heuristic classifier as a fallback
(see `app/layers/layer2_classifier.py`).

To populate it:

```bash
pip install -r requirements.txt -r requirements-ml.txt
python scripts/export_onnx_model.py
```

Expected layout after running the script:

```
app/models/
├── prompt_guard_quant.onnx   # quantized INT8 ONNX model
└── tokenizer/                # matching HuggingFace tokenizer files
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ...
```

Both paths are configurable via `ONNX_MODEL_PATH` / `ONNX_TOKENIZER_PATH`
in `.env`. This directory's contents are `.gitignore`d — don't commit
multi-hundred-MB model binaries to the repo; distribute them via your own
artifact storage (S3, a model registry, Docker image layer, etc.) instead.
