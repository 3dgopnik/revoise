# VibeVoice TTS Guide

## Installation
1. Download the `vibe-voice` binary from the official releases and place it in your `PATH`.
2. Fetch model weights:
   ```bash
   uv run python tools/fetch_tts_models.py vibevoice --model 1.5b
   ```
3. Verify the setup:
   ```bash
   vibe-voice --help
   ```

## Model variants
- **300M** – light variant for CPU or low VRAM GPUs.
- **1.5B** – full quality model (~9.5 GB VRAM in fp16).
- **1.5B Q8** – int8 quantized weights (~5 GB VRAM).

## Attention backends
- `sdpa` – default PyTorch scaled dot-product attention.
- `flash` – FlashAttention 2; fastest on NVIDIA Ampere+ GPUs.
- `torch` – fall-back implementation when others are unavailable.

## VRAM usage
| Model | Backend | Precision | Approx VRAM |
|-------|---------|-----------|-------------|
|300M   | sdpa    | fp16      | ~2 GB|
|1.5B   | sdpa    | fp16      | ~9.5 GB|
|1.5B   | flash   | int8      | ~5 GB|

VRAM varies with batch size and sequence length. Use the smallest model and quantization that meet your quality needs.

## Quantization
- `fp16`/`bf16` – highest quality; requires most VRAM.
- `int8` – balances quality and size; good default for GPUs.
- `int4` – experimental; largest speed/VRAM savings with quality trade-offs.

## Long-form tips
- Chunk long scripts into 30–60 s segments to limit memory spikes.
- Enable `tts.autosave_minutes` to checkpoint progress for lengthy jobs.
- Set `tts.force_offload=true` to release GPU memory between segments.
- Prefer quantized weights when generating hours of audio.
- Log peak VRAM usage to tune model/precision for your hardware.
