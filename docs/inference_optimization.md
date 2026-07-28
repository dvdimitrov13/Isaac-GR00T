# GR00T N1.7 Inference Optimization

Findings and verified results from profiling the N1.7 inference path on an
RTX 5090. Everything here was measured through the real
`Gr00tPolicy.get_action()` loop, not composed from per-stage arithmetic.

**Setup:** RTX 5090 (32 GB), base `GR00T-N1.7-3B`, DROID embodiment
(`OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`, 2 cameras, 2 frames), 4 denoising
steps, camera frame held for 3 consecutive control steps (10 Hz camera driving
a 30 Hz control loop).

---

## Results

Mean over all steps. `p50` is reported separately because with caching enabled
two of every three steps are cache hits, so the median lands on the cheap path
and overstates real throughput.

| Configuration | mean | p50 | p90 | Hz | Speedup |
|---|---|---|---|---|---|
| Eager baseline | 198.2 ms | 199.1 | 206.9 | 5.05 | 1.00x |
| PyTorch: caches + CUDA graphs | 66.2 ms | 37.6 | 125.0 | 15.11 | 3.00x |
| TensorRT alone | 81.8 ms | 81.6 | 86.7 | 12.22 | 2.42x |
| **TensorRT + caches** | **49.1 ms** | 30.0 | 88.7 | **20.37** | **4.04x** |

Two results worth noting:

- **TensorRT alone (2.42x) is slower than caching alone (3.00x).** They are
  complementary — TRT makes work cheaper, caching removes work entirely. Going
  straight to TRT and skipping the caching work is a net loss.
- **TensorRT has a much tighter tail** (p90 88.7 vs 125.0 ms). A fixed-rate
  control loop budgets for worst case, so this may matter more than the mean.

---

## The forward pass is dispatch-bound, not compute-bound

The single most important finding: **reducing image resolution does not reduce
end-to-end latency**, because you are not waiting on GPU compute.

Sweeping the VLM token budget (`image_processor.size["longest_edge"]`):

| ceiling | seq len | patches | backbone GPU | backbone wall | E2E total |
|---|---|---|---|---|---|
| 65536 (256²) | 461 | 1792 | 43.1 ms | 51.5 ms | 181.8 ms |
| 16384 (128²) | 73 | 240 | 14.3 ms | 59.6 ms | 186.7 ms |
| 4096 (64²) | 21 | 32 | 11.2 ms | 59.1 ms | 184.4 ms |

56x fewer patches and 3.9x less backbone GPU work produced **no change in
end-to-end latency**. Backbone GPU utilisation collapses from 79% to 19% while
wall time stays pinned.

The backbone issues ~4,860 kernel launches per frame. That per-layer dispatch
cost is fixed and independent of token count, so cutting tokens removes GPU work
you were never waiting on. The action head is worse: 104.9 ms wall against
37.6 ms of GPU time — **36% utilisation, ~68 ms of pure dispatch**.

Roughly 63% of the eager baseline is spent launching kernels, not computing.

### Two traps that made this confusing

1. **Upstream resizing is silently undone.** `Qwen2VLImageProcessorFast` is
   configured with `size = {"shortest_edge": 65536, "longest_edge": 16777216}`.
   `shortest_edge` is a *pixel-count floor*: resize images to 112x112 upstream
   and they are scaled straight back up to 256x256 before the VLM sees them.
   Measured token counts were identical (461 tokens / 1792 patches) for every
   upstream size from 256px down to 112px. The binding knob is `longest_edge`,
   the ceiling — lowering `shortest_edge` alone does nothing, since lowering a
   floor never forces a downscale.

2. **`gc.collect()` appearing to "fix" latency is a measurement artifact.** It
   does not warm CPU caches; a ~300 ms pause simply lets the async CUDA queue
   drain, moving where wall-clock time lands. This is why it "worked" while
   freeing zero objects, and why `gc.disable()` did not reproduce it.

---

## What works

### 1. Pin the thread count (one line, free)

`os.cpu_count()` reports the *host's* CPU topology inside a container. On a
24-vCPU cloud instance it returned 192, so torch sized its pool at 96 threads
for 24 real cores.

| torch threads | data prep p50 | data prep p90 |
|---|---|---|
| 96 (default) | 57.1 ms | 82.7 ms |
| 8 | 29.9 ms | 33.7 ms |

The cost is contention, not thread creation (pools are persistent). Torch
parallel regions end at a barrier, so each one waits for the slowest thread;
with 4x oversubscription most threads are descheduled at any moment, so you pay
the tail every time. The p90 improves more than the median, which is the
signature of contention rather than slow code.

It also makes profiling lie: the same ~90 ms of work migrates between whichever
stages you instrument, so independent per-stage medians sum to far more than the
measured end-to-end. **Always reconcile stage sums against E2E.**

> `scripts/deployment/benchmark_inference.py` does not pin threads. On such a
> machine it reports data processing at ~107 ms instead of ~25 ms, making
> preprocessing look like the dominant pipeline stage when it is ~13%.

### 2. CUDA graphs on the action head

```python
policy.model.action_head.model.forward = torch.compile(
    policy.model.action_head.model.forward, mode="reduce-overhead"
)
```

104.9 ms -> 22.5 ms. This removes dispatch overhead *and* fuses kernels; the
result is below the original GPU time, so both mechanisms contribute.

### 3. Backbone + preprocessing caching

See `Gr00tPolicy(cache_backbone=True, cache_preprocessing=True)`. On a repeated
camera frame both the vision features and the entire VLM preprocessing branch
are unchanged, so both are reused; only `state` is recomputed (~0.5 ms), since
it changes every control step.

Preprocessing on a cache hit: 24.7 ms -> 0.5 ms (46x), bit-identical output.

### 4. TensorRT for the backbone

```bash
python scripts/deployment/build_trt_pipeline.py \
  --model-path <checkpoint> \
  --dataset-path demo_data/droid_sample \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --export-mode full_pipeline \
  --output-dir ./trt_deploy
```

~10 minutes total (7 ONNX files, 7 engines, verification, benchmark). Load with
`setup_tensorrt_engines(policy, engines_dir, mode=InferenceMode.n17_full_pipeline)`.

Backbone 54.1 -> 28.5 ms; action head -> 17.7 ms.

---

## What does not work

### torch.compile on the VLM backbone

| variant | wall time |
|---|---|
| eager + flash-attn | **54.1 ms** |
| eager + sdpa | 61.0 ms |
| flash-attn + compile | fails to trace |
| sdpa + compile (default) | 99.1 ms, **drift 1.87e+01** |
| sdpa + reduce-overhead | CUDA capture error |

Every compiled variant is slower than eager, and the one that runs produces
large numerical drift. The blockers are data-dependent control flow, not just
the untraceable flash-attn kernel:

- `rot_pos_emb`: `int(grid_thw[:, 1:].max().item())`
- `get_image_features`: `(image_grid_thw.prod(-1) // merge**2).tolist()`

Each forces a graph break, leaving many small compiled fragments whose runtime
overhead exceeds plain eager dispatch. Disabling the KV cache
(`use_cache=False`) does **not** help — it is slower (64.6 ms) — so the
`recompile_limit` warnings are a symptom, not the cause.

This matches NVIDIA's published figures, where `torch.compile` moves the
backbone only 31.3 -> 30.4 ms. TensorRT is the supported path for the backbone.

---

## Cross-check against NVIDIA's published numbers

H100, GR00T-N1.7-LIBERO, 1 camera, 4 denoising steps:

| | data proc | backbone | action head | E2E |
|---|---|---|---|---|
| PyTorch eager | 6.2 | 31.3 | 48.2 | 85.8 ms |
| torch.compile | 6.2 | 30.4 | 12.0 | 48.6 ms |
| TensorRT | 6.2 | 8.8 | 12.3 | 27.9 ms |

Consistent with our measurements: preprocessing is a small share (7% theirs,
13.5% ours — we run 2 cameras on a slower CPU), the action head dominates eager
(56% theirs, 57% ours), `torch.compile` fixes the action head but not the
backbone, and TensorRT is what moves the backbone.

---

## Benchmarking notes

Mistakes that produced wrong conclusions during this work, all worth avoiding:

1. **Report the mean, not p50,** when caching makes some steps cheap. p50 lands
   on cache hits and overstated one result as 5.31x versus a true 2.95x.
2. **Run an A/A control** (same configuration twice) to establish the noise
   floor before treating a difference as real.
3. **Latency benchmarks cannot detect correctness regressions.** A cache bug
   here produced ~1.5 (physical units) action error while looking like a clean
   1.4x speedup. Always diff actions against an unoptimised reference.
4. **Reconcile per-stage timings against end-to-end.** If they do not add up,
   the breakdown is wrong.
5. **Composed arithmetic overestimates.** Summing separately-measured stages
   projected 19.8 Hz where the measured pipeline gave 15.6 Hz.

---

## Validating accuracy without a robot

Two levels, neither requiring hardware:

- **Open-loop** (`gr00t/eval/open_loop_eval.py`): MSE/MAE of predicted actions
  against ground-truth trajectories. Cheap, but understates risk — control is
  closed-loop, so small per-step errors compound.
- **Closed-loop simulation** (`gr00t/eval/rollout_policy.py` with LIBERO,
  SimplerEnv, or Robocasa): task success rate, which is the metric that matters.
  The repo documents PyTorch 100% (20/20) vs TRT 95% (19/20) on a LIBERO task
  over 20 episodes — within simulation noise.

### How much numerical drift is acceptable?

TensorRT is not bitwise-equal to PyTorch by design, and NVIDIA publishes no
universal tolerance. A raw max-absolute difference is meaningless without the
action scale, so compare per key, relative to each signal's own range:

| action key | range | PyTorch CUDA graphs | TensorRT |
|---|---|---|---|
| `eef_9d` | 2.00 | 0.42% of range, cos 0.999996 | 4.38%, cos 0.999768 |
| `joint_position` | 4.64 | 0.49% of range, cos 0.999993 | 1.71%, cos 0.999949 |
| `gripper_position` | 0.0215 | 18.18%, cos 0.894751 | 18.18%, cos 0.940046 |

Two traps here:

- **The alarming gripper figure is not a TensorRT problem.** Max absolute error
  is identical (0.0039) on both paths, and PyTorch's cosine is the *worse* of
  the two. `gripper_position` is a near-binary signal with range 0.0215, so any
  small perturbation looks huge relative to it. This is an artifact of
  normalising by a tiny range.
- **Do not aggregate across keys.** A single max-abs over all actions is
  dominated by whichever key has the largest units and hides the per-signal
  picture entirely.

The most informative check is error versus the action's own step-to-step
change: PyTorch 0.01-0.26x, TensorRT 0.05-0.19x. In both cases the numerical
perturbation is well below the motion the task itself commands each step.

Per-tensor cosine similarity is a weak gate: this TRT build verifies at
cosine 0.999961 overall while `gripper_position` sits at 0.94. Judge with
closed-loop success rate, not cosine.
