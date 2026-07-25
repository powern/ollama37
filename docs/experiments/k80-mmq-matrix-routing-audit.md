# K80 MMQ matrix-routing audit: Q4_K and MXFP4

## Scope

This audit answers two questions for the Tesla K80 (`sm_37`) experiment:

1. Which Q4_K/MXFP4 matrix multiplications use the FP32 cuBLAS fallback during prefill?
2. Are those operations structurally supported by the existing pre-Volta MMQ implementation?

The conclusions below are from static source tracing. Hardware correctness and performance still require the A/B run on a real K80.

## CUDA dispatch rules

`nn.Linear.Forward` emits `Weight.Mulmat`, while `nn.LinearBatch.Forward` emits `Weight.MulmatID`.

For ordinary `GGML_OP_MUL_MAT`, the CUDA dispatcher uses this order:

1. MMVQ for quantized weights when the activation matrix has at most `MMVQ_MAX_BATCH_SIZE` columns;
2. MMQ when `ggml_cuda_should_use_mmq(...)` permits it;
3. FP32 cuBLAS fallback otherwise.

On `sm_37`, `ggml_cuda_should_use_mmq(...)` currently returns false before the explicit `GGML_CUDA_FORCE_MMQ` override because K80 is below the hardware-DP4A gate (`sm_61`). Therefore a quantized ordinary linear with more than the MMVQ limit reaches `ggml_cuda_op_mul_mat_cublas`, which allocates an FP32 weight buffer, dequantizes the selected weight rows into it, and calls `cublasSgemm`.

For `GGML_OP_MUL_MAT_ID` (MoE):

1. single-token decode (`ne2 == 1`) uses MMVQ directly for quantized weights;
2. multi-token execution tries MMQ;
3. when MMQ is rejected, the fallback copies expert IDs to the CPU, synchronizes, groups tokens per expert on the host, and recursively calls ordinary `ggml_cuda_mul_mat` for each expert slice.

Consequently, an MXFP4 expert slice with at most the MMVQ limit of routed tokens can still use MMVQ. A slice with more routed tokens reaches FP32 cuBLAS on K80.

## Gemma 3 Q4_K_M routing

The current `gemma3:4b` registry model is `Q4_K_M`. Its text blocks contain the following relevant weight classes:

| Matrix class | Typical type in the registry model | Operation | K80 prefill path when columns > MMVQ limit |
|---|---|---|---|
| attention query | Q4_K | `MUL_MAT` | FP32 dequantize + cuBLAS |
| attention key | Q4_K | `MUL_MAT` | FP32 dequantize + cuBLAS |
| attention value | Q6_K | `MUL_MAT` | FP32 dequantize + cuBLAS |
| attention output | Q4_K | `MUL_MAT` | FP32 dequantize + cuBLAS |
| FFN gate | Q4_K | `MUL_MAT` | FP32 dequantize + cuBLAS |
| FFN up | Q4_K | `MUL_MAT` | FP32 dequantize + cuBLAS |
| FFN down | mixed Q4_K/Q6_K by tensor importance | `MUL_MAT` | FP32 dequantize + cuBLAS |

`token_embd.weight` is Q6_K but input token lookup is `GET_ROWS`, not matrix multiplication. Norm tensors are F32 and are not candidates for MMQ.

Important: the current broad `GGML_CUDA_FORCE_MMQ` experiment affects every supported quantized type, so a Gemma Q4_K_M run tests Q4_K and Q6_K together, not Q4_K in isolation.

## GPT-OSS MXFP4 routing

The current `gpt-oss:20b` registry model stores attention projections in BF16 and the expert FFN projections in MXFP4:

- `ffn_gate_exps.weight` — MXFP4;
- `ffn_up_exps.weight` — MXFP4;
- `ffn_down_exps.weight` — MXFP4;
- an alternate combined `ffn_gate_up_exps` layout is also supported by the model code;
- `ffn_gate_inp.weight` (router) is F32, not MXFP4.

The MXFP4 weights are executed through `nn.LinearBatch`, therefore through `MUL_MAT_ID`.

For gpt-oss:20b there are 32 experts and 4 selected experts per token. The average number of routed token-expert assignments per expert is approximately:

`prefill_tokens * 4 / 32 = prefill_tokens / 8`.

This means:

| Prefill tokens in a micro-batch | Average assignments per expert | Expected fallback behaviour without MMQ |
|---:|---:|---|
| 16 | 2 | mostly per-expert MMVQ |
| 32 | 4 | mostly per-expert MMVQ |
| 64 | 8 | boundary; routing imbalance matters |
| 128 | 16 | many expert slices exceed MMVQ and use FP32 cuBLAS |
| 256 | 32 | predominantly FP32 cuBLAS |
| 512 | 64 | predominantly FP32 cuBLAS |

The distribution is not uniform, so the table is a planning estimate rather than a guarantee.

Forced MMQ changes more than the arithmetic kernel for GPT-OSS. It lets the top-level `MUL_MAT_ID` call enter `ggml_cuda_mul_mat_q(...)`, whose CUDA-side `mmq_ids_helper` performs expert grouping on the GPU. This can avoid the existing device-to-host ID copy, host synchronization, CPU grouping loop, per-expert recursion, and repeated FP32 weight dequantization.

This makes GPT-OSS the higher-leverage experiment even if software-emulated DP4A itself is not faster than SGEMM.

## Static MMQ compatibility

Both Q4_K and MXFP4 are explicitly present in:

- the `ggml_cuda_should_use_mmq` supported-type switch;
- the MMQ type-dispatch switch;
- MMQ shared-memory layout selection;
- DP4A tile-size selection;
- quantized weight tile loaders and vector-dot code.

The MMQ implementation has explicit pre-Volta tile dimensions (`mmq_x <= 64`, `mmq_y = 64`) and non-MMA branches. K80 lacks hardware DP4A, so `ggml_cuda_dp4a` expands each byte dot product into scalar `int8` multiplies and additions. MXFP4 loading uses the existing 16-entry lookup path and E8M0 scale conversion; no FP4 hardware is required.

Therefore the static verdict is:

- **Q4_K: structurally compatible with the pre-Volta MMQ path**;
- **MXFP4: structurally compatible with the pre-Volta MMQ path, including CUDA-side expert grouping**;
- **performance is unknown** because software DP4A can be slower than dequantize + SGEMM;
- **correctness is not yet proven on sm_37** and must be checked against the baseline output.

## Risks to test explicitly

1. `mmq_ids_helper` uses warp shuffle sync intrinsics on the K80 build.
2. Software DP4A may make dense Q4_K prefill slower.
3. Register pressure and occupancy may differ sharply between Q4_K, Q6_K, and MXFP4.
4. A global force flag may improve MXFP4 MoE while regressing Q4_K/Q6_K dense layers.
5. The MoE helper allocates dynamic shared memory proportional to token count; the planned 16–512 grid is comfortably below K80 limits, but larger batches need a separate check.

## Revised test priority

1. `gpt-oss:20b`, `num_batch=64,128,256,512` — highest expected gain because MMQ can eliminate host expert sorting and FP32 expert fallback.
2. `gemma3:4b`, `num_batch=16,32,64,128,256,512` — determines whether software-DP4A MMQ beats SGEMM for dense Q4_K/Q6_K projections.
3. `num_batch=1,8` controls — verify that decode/MMVQ behaviour is unchanged.
4. At least one warmup plus three measured runs per cell, with output correctness, prompt tok/s, generation tok/s, VRAM per die, active dies, and CUDA errors recorded.

## Recommended follow-up after the broad experiment

If GPT-OSS improves but Gemma regresses, replace the global force experiment with a scoped selector that permits pre-DP4A MMQ only for `GGML_TYPE_MXFP4` on `sm_37`. If Gemma improves as well, test Q4_K and Q6_K separately before proposing a production default.