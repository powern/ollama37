# K80 forced-MMQ experiment (sm_37)

## Purpose

Test whether the existing pre-Volta MMQ kernels are faster than the current
`dequantize -> FP32 cuBLAS` fallback during prompt processing on Tesla K80.

This experiment does **not** change production defaults. The source patch is
applied only while building `Dockerfile.mmq-experiment`, and MMQ is enabled
only through `GGML_CUDA_FORCE_MMQ=ON`.

## Build

```bash
cd /path/to/ollama37

docker build \
  --build-arg GIT_COMMIT="$(git rev-parse HEAD)" \
  --build-arg OLLAMA_VERSION="0.0.0-mmq-sm37" \
  -f docker/runtime/Dockerfile.mmq-experiment \
  -t ollama37:mmq-sm37 .
```

The build must show the patched `GGML_CUDA_FORCE_MMQ` block before the DP4A
hardware gate.

## Baseline and experiment

Use the same model volume and run only one Ollama container at a time.

Recommended first models:

- `gemma3:4b` or another Q4_K model;
- `gpt-oss:20b` for MXFP4.

Recommended `num_batch` grid:

```text
16 32 64 128 256 512
```

For every cell record:

- `prompt_eval_count / prompt_eval_duration`;
- `eval_count / eval_duration`;
- GPU offload percentage;
- VRAM used on every K80 die;
- output correctness;
- CUDA errors or crashes.

Example deterministic request:

```bash
curl -s http://127.0.0.1:11434/api/generate -d '{
  "model": "gemma3:4b",
  "prompt": "Explain how a computer works to a curious 10-year-old. Be fun and use analogies.",
  "stream": false,
  "options": {
    "temperature": 0,
    "seed": 42,
    "num_predict": 128,
    "num_ctx": 2048,
    "num_batch": 128
  }
}' | jq '{
  prompt_eval_count,
  prompt_eval_duration,
  eval_count,
  eval_duration,
  done_reason,
  response
}'
```

## Success criteria

Keep the experiment only when all conditions hold:

1. prompt-evaluation throughput is reproducibly higher;
2. generation throughput does not regress materially;
3. output remains correct and deterministic;
4. GPU offload and active-die count do not worsen;
5. VRAM usage remains acceptable;
6. there are no CUDA errors, hangs, or host instability.

A single run is not enough. Use at least three measured runs after one warmup
for each selected configuration.

## Rollback

Stop the experimental container and start the normal compose service again:

```bash
docker rm -f ollama37-mmq-sm37 2>/dev/null || true
cd docker
docker compose up -d --force-recreate
```
