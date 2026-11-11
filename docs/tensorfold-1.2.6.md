# TensorFold 1.2.6 on one GB10 Spark

This branch moves the recipe from TensorFold **1.1.4** to **1.2.6** and carries only
three patches: the FP4 vision tower (`0001`), the 16,384-token image budget
(`0002`) and the fp8 KV cache (`0003`). Everything else the 1.1.4 recipe carried is
either upstream now or was measured and left out (below).

## Measured, same engine settings, same box

Both runs use fp8 KV, four streams, 262,144 tokens, vision on, thinking on demand.
`tools/ab.py` drives them: distinct prompts per size, greedy decoding, a hash over
every reply.

| metric | 1.1.4 + 17 patches | 1.2.6 + 3 patches | |
|---|---|---|---|
| reply hash (whole suite) | `ac7aedb8…` | `ac7aedb8…` | identical |
| 2K cold prefill | 1,904 t/s | 1,907 t/s | level |
| 8K cold prefill | 2,320 t/s | 2,237 t/s | −4% |
| 105K cold prefill | 44.1 s | 48.4 s | −9% |
| 105K follow-up turn | 0.45 s | 0.45 s | level |
| decode | 99.6 t/s | 102.4 t/s | +3% |
| 16 concurrent follow-ups, median | 2.18 s | 1.65 s | **−24%** |
| engine load | 176 s | 57 s | **−68%** |
| prompt cache at 4 × 262K | 12.0 GiB | 12.0 GiB | level |

The reply hash matching across the whole suite matters more than any timing: the
engine changed, the model's answers to the same prompts did not.

The prefill gap is not a missing patch. 1.2.6 already contains upstream's own
MXFP8 prompt tile and its own 256-column hyper-connection kernels, so the 1.1.4
patches that added those are superseded; the one piece of that patch upstream does
not have, a skipped host drain between prompt chunks, measured as a no-op here
(48.7 s vs 48.8 s at 105K) and was left out.

## The bug this port had to find

The fp8 patch's kernels are template instantiations. Upstream's template is
`<NB, ONE, FUSE, PAGED>`; the merged one adds `KV8` as a fifth parameter. The fp8
entry points still passed the 1.1.4-era four arguments, so
`fn_attn_chunks_kv8` instantiated `PAGED=true, KV8=false` — the paged bf16 path
with no page table — and every prompt wider than one 512-key chunk read from a
wild address.

`compute-sanitizer` with `-lineinfo` named it directly: a 16-byte read in
`cp_async.cuh` from the `PAGED=false/KV8=false` instance, 9.5e18 bytes from any
allocation. One missing template argument, both entry points.

## Settings that differ from the 1.1.4 recipe

- `TF_FLASHNEXT_CHUNK=4096` — 1,536-row passes are the 1.2.6 default and cost
  12–14% of prefill; 4,096 recovers it while keeping ~12 GiB of prompt cache.
  8,192 is another 2% of prefill but loses 3 GiB of cache.
- `--vision-image-tokens 16384` still needs patch `0002`: the stock check refuses
  anything above 4,096.

## What is not here

`patches/hold/` keeps the fourteen remaining 1.1.4 patches for reference. Five of
them (sampled heads, keep-late, shared side, GDN lanes, draft confidence) now
exist upstream as the same environment variables. The rest are prompt/MoE kernel
work that either duplicates upstream's newer tiles or needs a numerics review
before it can be trusted on 1.2.6.

## Reproducing

```bash
tools/ab.py --url http://127.0.0.1:8003/v1 --label tf-1.1.4 --out base.json
tools/ab.py --url http://127.0.0.1:8004/v1 --label tf-1.2.6 --baseline base.json
```

Capture a baseline on a freshly started engine: a warm lane answers cold prompts
from its cache and the gate will say so and skip them.
