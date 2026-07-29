# CUDA graph capture for the GR00T N1.7 model forward

`Gr00tPolicy(..., use_cuda_graph=True)` takes the model forward from **90.5 ms to
32.3 ms (2.80x)** — 11.1 Hz to 31.0 Hz — on an RTX 5090 with the LIBERO
checkpoint at batch 1.

The graph replay itself is **bit-identical**: every action key matches eager
execution to `0.000e+00`. The measured numbers, the reasoning behind them, and
the one caveat are below.

---

## Why there was anything to win

The model is not compute-bound at batch 1. It is *dispatch*-bound: the GPU
spends most of the forward pass idle, waiting to be told what to do next.

Profiling the two stages separately (RTX 5090, batch 1, threads pinned):

| stage | wall time | actual CUDA kernel time | kernel launches | GPU idle |
|---|---|---|---|---|
| backbone (ViT + LLM) | 30.2 ms | 10.40 ms | 2 324 | 66% |
| action head (DiT ×4 steps) | 61.9 ms | 15.18 ms | 3 164 | 75% |
| **total** | **92.1 ms** | **25.58 ms** | **5 488** | **72%** |

Roughly **72% of the model forward is dead time**, about 12 µs of launch overhead
per kernel across 5 488 launches. That is the entire opportunity, and CUDA graph
replay addresses it directly: record the launch sequence once, then issue it with
a single host call.

### This is specifically a robotics problem

Batched serving hides this cost. Sweeping batch size on the action head:

| batch | total wall | per-sample | launches |
|---|---|---|---|
| 1 | 55.3 ms | 55.3 ms | 6 268 |
| 4 | 62.5 ms | 15.6 ms | 6 128 |
| 16 | **62.7 ms** | 3.92 ms | 6 124 |
| 32 | 103.2 ms | 3.22 ms | 6 188 |

Sixteen robots cost the same wall-clock as one, and the launch count never
changes. A server amortises dispatch overhead across a batch; **a robot's control
loop is irreducibly batch 1** and has nothing to amortise against. The mainstream
inference stack (vLLM, TensorRT-LLM, continuous batching) optimises a regime
robotics is structurally not in.

The effect should be *larger* on Jetson Orin and Thor, where dispatch is paid by
a much weaker ARM host.

### What is *not* the problem

Two plausible-sounding explanations were tested and rejected, which is why this
change is graph capture and not a kernel rewrite:

- **"cuBLAS is bad at skinny batch-1 GEMMs."** No. Auditing all 11 unique
  `nn.Linear` shapes in the action head, the large projections reach **88–93% of
  achievable HBM bandwidth**. Total GEMM headroom is only 1.6x (9.55 ms measured
  vs 6.04 ms at the bandwidth ceiling).
- **"Fuse the elementwise ops."** No. GPU time splits matmul 74.8%, elementwise
  10.4%, attention 7.8%, normalisation 4.2%, memory movement 2.9%. Making every
  non-GEMM kernel free would buy about 20%.

> **Benchmarking note.** Timing `F.linear` in a Python loop measures *host
> dispatch*, not the kernel — it bottoms out around 9 µs per call regardless of
> shape. Capture N calls into a CUDA graph and time the replay instead. The first
> version of the GEMM audit was meaningless for exactly this reason.

---

## Why capture did not simply work

A CUDA graph is recorded by capturing a stream. If anything in the captured
region synchronises with the host, it tries to read a result that has not been
computed yet, and the capture is invalidated with
`cudaErrorStreamCaptureInvalidated`.

HuggingFace's Qwen3-VL forward contains **seven** such synchronisations. None of
them perform modelling work — they size buffers, validate shapes, and answer
questions about masks. Each was located with
`torch.cuda.set_sync_debug_mode("error")`, which raises at the exact offending
call, and each fix was verified to leave the backbone features bit-identical.

| # | location | what it does | fix |
|---|---|---|---|
| 1 | `fast_pos_embed_interpolate` | `torch.linspace(0, n-1, h)` with `h` from GPU-resident `grid_thw`; also round-trips indices through Python lists via `.tolist()` every forward | memoise |
| 2 | vision forward | `repeat_interleave` with tensor `repeats` to build `cu_seqlens` | host-side `grid_thw` |
| 3 | `rot_pos_emb` | `int(grid_thw[:, 1:].max().item())` | memoise |
| 4 | `get_placeholder_mask` | `inputs_embeds[special_image_mask].numel() != ...` — a **pure validation assert** whose result feeds nothing | drop the assert |
| 5 | `get_rope_index` | `input_ids[attention_mask[i] == 1]` | memoise |
| 6 | `_ignore_causal_mask_sdpa` | `padding_mask.all()` — "is the mask all ones?" | memoise |
| 7 | `_deepstack_process` | `hidden_states[visual_pos_masks, :]`, a data-dependent boolean gather | cached indices + `index_select`/`index_copy_` |

### The common cause

Five of the seven exist to read **`image_grid_thw`** — a `(num_images, 3)` int64
tensor. For a two-camera setup that is **six integers** describing the patch
grid. Keeping them on the GPU means the pipeline stalls every time some helper
needs them as Python ints.

They are moved to the host instead (`host_side_geometry`). The index tensors
built from them are still allocated on the GPU, so the numerics are untouched.

### The assumption, and its guard

Memoising these is valid because they depend only on **image geometry and prompt
shape**, which are fixed for a deployed robot — the cameras do not change
resolution mid-episode. That is a domain assumption a general-purpose compiler is
not permitted to make, and it is the reason this win is available here.

It is enforced, not assumed. `ModelGraphRunner` fingerprints every input —
shapes, dtypes, devices, and the *values* of the small host-side geometry tensors
— and falls back to eager execution if anything changes. That is a correctness
guard: replaying a graph against mismatched inputs would silently produce wrong
actions.

Capture is also best-effort. If it fails for any reason, the runner logs a
warning, restores the original implementations, and runs eagerly.

### What "static" actually means here

Only the *geometry* is frozen. The pixels and robot state change every control
step — that is the entire point. What must stay constant is:

- image resolution and patch grid,
- number of cameras,
- prompt token length.

A fixed camera rig satisfies the first two trivially. **Prompt length is the real
constraint**, not the camera: a different language instruction with a different
token count changes the input signature.

Two dependencies are stacked, and they are worth separating:

1. CUDA graphs *inherently* require static shapes and addresses. True of any
   model, unrelated to the patches above.
2. The memoisation additionally caches *values* — interpolated position
   embeddings, rope indices, the mask decision — which are constant only under
   the geometry assumption.

**Known limitation.** The runner currently holds a single graph, so a changed
signature falls back to eager permanently for those inputs — correct, but with no
speedup. A small cache of graphs keyed by signature would cover multi-task
setups with varying instruction lengths, at the cost of memory per graph. That is
a natural follow-up.

---

## Results

RTX 5090, LIBERO `libero_10`, batch 1, `torch.set_num_threads(8)`, measured
through the public `policy.get_action(obs)` API.

| configuration | latency | rate | vs default |
|---|---|---|---|
| eager, flash-attention-2 (shipped default) | 90.46 ms | 11.1 Hz | — |
| eager, sdpa | 83.49 ms | 12.0 Hz | 1.08x |
| **`use_cuda_graph=True`** | **32.29 ms** | **31.0 Hz** | **2.80x** |

Isolating the whole model forward (excluding data preprocessing) gives
79.8 ms → 26.2 ms, a **3.04x**, against 25.58 ms of measured kernel time — i.e.
the forward now runs essentially at its kernel-time floor.

### Correctness

| compared against | max abs difference |
|---|---|
| eager with sdpa | **0.000e+00 on every action key** |
| eager with flash-attention-2 | up to 9.5e-03 (`y`, range 0.70) |

**The graph replay is exact.** Replay executes identical kernels in identical
order on identical buffers, and that is what the first row measures.

The second row is *not* caused by capture. Enabling the flag also switches the
backbone from flash-attention-2 to sdpa, which is required — flash-attention's
varlen kernel takes `cu_seqlens` as a CUDA tensor, which is incompatible with
host-side geometry. sdpa also measured faster here (backbone 30.6 ms → 24.5 ms),
since these sequences are short enough that flash-attention's advantage does not
materialise while its setup cost does.

Both are *exact* attention implementations, so the difference is floating-point
accumulation order, not a modelling change — the same class of numerical
difference as a TensorRT engine.

> A drift figure alone does not establish that behaviour is unchanged. Relative
> error is misleading when a key's range is small. The appropriate standard is
> closed-loop task success, as used to validate the TensorRT path.

### Verifying it yourself

The action head samples from noise (flow matching), so two calls legitimately
differ. To compare outputs, pin the noise:

```python
# torch.randn must be patched AND the result cloned -- the denoising loop writes
# into the returned tensor, so handing back a cached object lets call N corrupt
# call N+1, which looks exactly like nondeterminism.
```

With the noise pinned, eager-vs-eager is `0.000e+00`, so any nonzero delta is
real signal.

---

## Usage

```python
from gr00t.policy.gr00t_policy import Gr00tPolicy

policy = Gr00tPolicy(
    model_path="...",
    embodiment_tag="LIBERO_PANDA",
    device="cuda",
    use_cuda_graph=True,   # default False
)

action = policy.get_action(observation)   # first call captures, then replays
```

Requirements and behaviour:

- CUDA only; ignored with a warning on CPU builds.
- The first `get_action` warms up and records the graph, so it is slower.
- Camera configuration and prompt length must stay fixed. If they change,
  execution falls back to eager with a warning.
- The backbone is switched to sdpa attention while capture is active, and
  restored if capture fails.

---

## Files

- `gr00t/model/graph_capture.py` — `ModelGraphRunner`, `CaptureCompatPatcher`,
  `host_side_geometry`
- `gr00t/policy/gr00t_policy.py` — the `use_cuda_graph` flag
- `tests/test_graph_capture.py` — signature guarding and eager-fallback tests
