#!/usr/bin/env python3
"""
Export & INT8-quantize a HuggingFace prompt-injection classifier to ONNX.

Requires `requirements-ml.txt` and network access to HuggingFace Hub. Not
run automatically anywhere in this repo — the shield works out of the box
on the heuristic fallback classifier (see app/layers/layer2_classifier.py);
run this script when you're ready to swap in a real trained model.

Usage:
    pip install -r requirements.txt -r requirements-ml.txt
    python scripts/export_onnx_model.py
    python scripts/export_onnx_model.py --model meta-llama/Prompt-Guard-2-86M --hf-token hf_...
    python scripts/export_onnx_model.py --cpu-arch avx2   # if your CPU/host doesn't support AVX512-VNNI

Default model: protectai/deberta-v3-base-prompt-injection-v2 — openly
licensed and does not require a gated-access HuggingFace token. Meta's
Prompt-Guard-2 is gated and needs `--hf-token` (or `huggingface-cli login`)
plus accepting the license on the model page first.

Output layout (matches app/config.py defaults):
    app/models/prompt_guard_quant.onnx
    app/models/tokenizer/
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

DEFAULT_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HuggingFace model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--hf-token", default=None, help="HuggingFace access token, for gated models")
    parser.add_argument(
        "--cpu-arch",
        default="avx512_vnni",
        choices=["avx2", "avx512", "avx512_vnni", "arm64"],
        help="Target CPU instruction set for INT8 quantization (default: avx512_vnni). "
             "Use avx2 for older/cloud CPUs without VNNI support, arm64 for Apple Silicon / Graviton.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "app" / "models"),
        help="Where to write prompt_guard_quant.onnx and tokenizer/ (default: app/models/)",
    )
    args = parser.parse_args()

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing ML dependencies. Install them first:\n"
            "  pip install -r requirements.txt -r requirements-ml.txt"
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = output_dir / "tokenizer"

    hub_kwargs = {"token": args.hf_token} if args.hf_token else {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        export_dir = tmp_path / "onnx"
        quant_dir = tmp_path / "quantized"

        print(f"[1/4] Downloading & exporting '{args.model}' to ONNX ...")
        model = ORTModelForSequenceClassification.from_pretrained(args.model, export=True, **hub_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(args.model, **hub_kwargs)
        model.save_pretrained(export_dir)
        tokenizer.save_pretrained(export_dir)

        print(f"[2/4] Quantizing to INT8 (target arch: {args.cpu_arch}) ...")
        quantizer = ORTQuantizer.from_pretrained(export_dir)
        qconfig_factory = getattr(AutoQuantizationConfig, args.cpu_arch)
        qconfig = qconfig_factory(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=quant_dir, quantization_config=qconfig)

        print("[3/4] Installing quantized artifact + tokenizer into app/models/ ...")
        quantized_onnx_files = list(quant_dir.glob("*quantized.onnx")) or list(quant_dir.glob("*.onnx"))
        if not quantized_onnx_files:
            raise SystemExit(f"No quantized .onnx file found in {quant_dir} — quantization may have failed.")
        shutil.copyfile(quantized_onnx_files[0], output_dir / "prompt_guard_quant.onnx")

        if tokenizer_dir.exists():
            shutil.rmtree(tokenizer_dir)
        tokenizer.save_pretrained(tokenizer_dir)

    print("[4/4] Done.")
    print(f"  Model:     {output_dir / 'prompt_guard_quant.onnx'}")
    print(f"  Tokenizer: {tokenizer_dir}")
    print("\nSet CLASSIFIER_BACKEND=onnx (or leave as 'auto') in your .env and restart the proxy.")


if __name__ == "__main__":
    main()
