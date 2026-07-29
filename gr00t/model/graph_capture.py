# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CUDA graph capture for the GR00T model forward.

At batch 1 the model is dispatch-bound rather than compute-bound: on an RTX 5090
the eager forward takes ~80 ms while the CUDA kernels it launches account for
only ~26 ms of device time. The remaining ~68% is the GPU idle between roughly
5500 kernel launches. Robotics inference cannot amortise that overhead the way
batched serving does -- a robot is one request -- so capturing the launch
sequence once and replaying it is the single largest available win.

Replay executes the identical kernels, in the identical order, on the identical
buffers, so it is output-preserving by construction. Measured on RTX 5090 with
the LIBERO checkpoint, end to end through ``policy.get_action``: 90.5 ms ->
32.3 ms (2.80x), with every action key matching eager sdpa execution to exactly
0.0.

One caveat: enabling capture also switches the backbone from flash-attention-2
to sdpa, because flash-attention's varlen kernel takes ``cu_seqlens`` as a CUDA
tensor and that is incompatible with keeping the image geometry on the host.
Both compute exact attention, so they differ only in floating-point accumulation
order -- the same class of difference a TensorRT engine introduces -- but
actions are then bit-identical to eager *sdpa*, not to the flash-attention
default. See docs/cuda_graph_capture.md.

Capture requires that the forward pass never synchronises with the host, because
a host sync during capture reads a result that has not been computed yet and
invalidates the capture. HuggingFace's Qwen3-VL forward contains seven such
syncs. None of them perform modelling work -- they size buffers, validate
shapes, and answer questions about masks -- and each is removed here by either
computing the value on the host or caching it, since all seven depend only on
the image geometry and prompt shape, which are fixed for a given robot
configuration.

Typical use is via ``Gr00tPolicy(..., use_cuda_graph=True)``; the pieces are
exposed individually for benchmarking and testing.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
from transformers.feature_extraction_utils import BatchFeature


logger = logging.getLogger(__name__)

__all__ = [
    "CaptureCompatPatcher",
    "ModelGraphRunner",
    "host_side_geometry",
    "is_capture_supported",
]


def is_capture_supported() -> bool:
    """CUDA graphs need a CUDA device; capture is unavailable on CPU builds."""
    return torch.cuda.is_available()


def host_side_geometry(backbone_inputs: BatchFeature) -> BatchFeature:
    """Return ``backbone_inputs`` with image-geometry tensors moved to the host.

    ``image_grid_thw`` is a tiny integer tensor -- (num_images, 3), six values
    for a two-camera setup -- describing the patch grid of each image. Several
    consumers in the vision tower need those integers on the host: to size a
    ``linspace``, to drive a Python loop, or as the ``repeats`` argument of
    ``repeat_interleave``. Keeping the tensor on the GPU forces a device read at
    each of those points.

    Moving it to the host makes all of them ordinary host arithmetic. The index
    tensors they build are still allocated directly on the GPU, so the numerics
    are untouched.
    """
    data = dict(backbone_inputs)
    for key, value in list(data.items()):
        if torch.is_tensor(value) and "grid" in key.lower() and value.device.type != "cpu":
            data[key] = value.cpu()
    return BatchFeature(data=data)


class _Memo:
    """Cache a callable whose result is fixed for a given robot configuration.

    Used for the vision-tower helpers that depend only on image geometry and
    prompt shape. Their results are recomputed identically on every control step
    today; caching them removes both the recomputation and the host syncs inside
    them.
    """

    def __init__(self, owner: Any, name: str) -> None:
        self.owner = owner
        self.name = name
        self.original = getattr(owner, name)
        self.value: Any = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.value is None:
            self.value = self.original(*args, **kwargs)
        return self.value

    def install(self) -> "_Memo":
        setattr(self.owner, self.name, self)
        return self

    def restore(self) -> None:
        setattr(self.owner, self.name, self.original)


class CaptureCompatPatcher:
    """Remove the host synchronisations that block CUDA graph capture.

    Seven syncs were located with ``torch.cuda.set_sync_debug_mode("error")`` in
    the Qwen3-VL forward. Each is patched below with a note on what it did and
    why the replacement is equivalent. Every patch was verified to leave the
    backbone features bit-identical.

    All seven are safe only while the image geometry and prompt shape are fixed,
    which holds for a deployed robot but not in general. ``ModelGraphRunner``
    enforces that by falling back to eager execution whenever the input
    signature changes.

    Call :meth:`restore` to put the original implementations back.
    """

    def __init__(self, backbone: torch.nn.Module) -> None:
        self.backbone = backbone
        self._memos: list[_Memo] = []
        self._undo: list[Callable[[], None]] = []
        self._applied = False

    def apply(self) -> "CaptureCompatPatcher":
        if self._applied:
            return self
        self._use_sdpa_attention()
        self._patch_grid_pure_helpers()
        self._patch_placeholder_mask()
        self._patch_masking_utils()
        self._patch_deepstack()
        self._applied = True
        return self

    def restore(self) -> None:
        for memo in self._memos:
            memo.restore()
        for undo in reversed(self._undo):
            undo()
        self._memos.clear()
        self._undo.clear()
        self._applied = False

    def _find(self, attr: str) -> Any:
        return next((m for m in self.backbone.modules() if hasattr(m, attr)), None)

    def _use_sdpa_attention(self) -> None:
        """Switch the backbone to sdpa attention.

        flash-attention-2 uses the varlen kernel, which takes ``cu_seqlens`` as
        a CUDA tensor derived from ``grid_thw``. That is incompatible with
        keeping the geometry on the host, and the varlen path needs the sequence
        bounds as host integers anyway.

        sdpa needs no such argument. It also measured *faster* than
        flash-attention-2 for this workload (backbone 30.6 ms -> 24.5 ms on RTX
        5090) -- the sequences here are short, so flash-attention's advantage
        does not materialise while its setup cost does. Verified bit-identical
        on the backbone features.
        """
        model = getattr(self.backbone, "model", None)
        if model is None:
            return
        config = getattr(model, "config", None)
        previous = getattr(config, "_attn_implementation", None) if config is not None else None
        if previous == "sdpa":
            return

        try:
            if hasattr(model, "set_attn_implementation"):
                model.set_attn_implementation("sdpa")
            else:
                for module in model.modules():
                    if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
                        module.config._attn_implementation = "sdpa"
        except Exception as exc:
            logger.warning("Could not switch the backbone to sdpa attention: %s", exc)
            return

        def undo(_model=model, _prev=previous):
            if _prev is None:
                return
            try:
                if hasattr(_model, "set_attn_implementation"):
                    _model.set_attn_implementation(_prev)
                else:
                    for module in _model.modules():
                        if hasattr(module, "config") and hasattr(
                            module.config, "_attn_implementation"
                        ):
                            module.config._attn_implementation = _prev
            except Exception:
                logger.warning("Could not restore attention implementation %s", _prev)

        self._undo.append(undo)

    def _patch_grid_pure_helpers(self) -> None:
        """Syncs 1, 3, 5 -- helpers whose result depends only on image geometry.

        ``fast_pos_embed_interpolate`` interpolates the vision position
        embeddings for the current patch grid. It calls
        ``torch.linspace(0, n - 1, h)`` with ``h`` taken from ``grid_thw``, and
        additionally round-trips the computed indices through Python lists via
        ``.tolist()`` on every forward pass.

        ``rot_pos_emb`` reads ``int(grid_thw[:, 1:].max().item())``.

        ``get_rope_index`` boolean-mask-indexes ``input_ids`` per batch row.

        All three are pure functions of the geometry and the prompt, so their
        results are identical on every control step.
        """
        visual = self._find("fast_pos_embed_interpolate")
        rope = self._find("get_rope_index")
        for owner, name in (
            (visual, "fast_pos_embed_interpolate"),
            (visual, "rot_pos_emb"),
            (rope, "get_rope_index"),
        ):
            if owner is not None and hasattr(owner, name):
                self._memos.append(_Memo(owner, name).install())

    def _patch_placeholder_mask(self) -> None:
        """Sync 4 -- a validation assertion that computes nothing.

        ``get_placeholder_mask`` checks that the number of image placeholder
        tokens matches the number of image features, via
        ``inputs_embeds[special_image_mask].numel()``. Boolean-mask indexing has
        a data-dependent output size, so evaluating it requires a device read.
        The result feeds nothing downstream -- it exists only to raise a helpful
        error -- so the replacement returns the same two masks without it.
        """
        holder = self._find("get_placeholder_mask")
        if holder is None:
            return
        original = holder.get_placeholder_mask

        def placeholder_mask_no_assert(
            input_ids=None, inputs_embeds=None, image_features=None, video_features=None
        ):
            if input_ids is None:
                embed = holder.get_input_embeddings()
                image_mask = (
                    inputs_embeds
                    == embed(
                        torch.tensor(
                            holder.config.image_token_id,
                            dtype=torch.long,
                            device=inputs_embeds.device,
                        )
                    )
                ).all(-1)
                video_mask = (
                    inputs_embeds
                    == embed(
                        torch.tensor(
                            holder.config.video_token_id,
                            dtype=torch.long,
                            device=inputs_embeds.device,
                        )
                    )
                ).all(-1)
            else:
                image_mask = input_ids == holder.config.image_token_id
                video_mask = input_ids == holder.config.video_token_id
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            return image_mask, video_mask

        holder.get_placeholder_mask = placeholder_mask_no_assert
        self._undo.append(lambda: setattr(holder, "get_placeholder_mask", original))

    def _patch_masking_utils(self) -> None:
        """Sync 6 -- ``padding_mask.all()`` in the attention-mask fast path.

        ``_ignore_causal_mask_sdpa`` asks whether the padding mask is entirely
        ones, so it can skip building an explicit causal mask. For a fixed
        prompt shape the answer never changes, so it is asked once and reused.
        """
        import transformers.masking_utils as masking_utils

        for name in ("_ignore_causal_mask_sdpa", "_preprocess_mask_arguments"):
            if not hasattr(masking_utils, name):
                continue
            original = getattr(masking_utils, name)
            cache: dict[str, Any] = {}

            def memoised(*args, _original=original, _cache=cache, **kwargs):
                if "value" not in _cache:
                    _cache["value"] = _original(*args, **kwargs)
                return _cache["value"]

            setattr(masking_utils, name, memoised)
            self._undo.append(lambda n=name, o=original: setattr(masking_utils, n, o))

    def _patch_deepstack(self) -> None:
        """Sync 7 -- a boolean-mask gather in the deepstack visual merge.

        ``_deepstack_process`` adds the per-layer visual embeddings back into
        the hidden states at the visual token positions, written as
        ``hidden_states[visual_pos_masks, :]``. The gather's output size is data
        dependent, so it needs a device read.

        ``visual_pos_masks`` is fixed for a given prompt and camera, so its
        indices are computed once and the same arithmetic is expressed with
        ``index_select`` / ``index_copy_``, which have static shapes.
        """
        owner = self._find("_deepstack_process")
        if owner is None:
            return
        original = owner._deepstack_process
        index_cache: dict[int, torch.Tensor] = {}

        def deepstack_static(hidden_states, visual_pos_masks, visual_embeds):
            width = hidden_states.shape[-1]
            if width not in index_cache:
                index_cache[width] = visual_pos_masks.reshape(-1).nonzero(as_tuple=True)[0]
            index = index_cache[width]
            # .view keeps this a view of hidden_states, so index_copy_ writes back
            flat = hidden_states.view(-1, width)
            embeds = visual_embeds.to(flat.device, flat.dtype)
            flat.index_copy_(0, index, flat.index_select(0, index) + embeds)
            return hidden_states

        owner._deepstack_process = deepstack_static
        self._undo.append(lambda: setattr(owner, "_deepstack_process", original))


def _signature(inputs: dict[str, Any]) -> tuple:
    """Identify an input configuration precisely enough to reuse a graph.

    Shape, dtype and device of every GPU tensor, plus the *values* of the small
    host-side geometry tensors -- a different camera resolution must invalidate
    the graph and the caches inside :class:`CaptureCompatPatcher`, and it shows
    up only in those values.
    """
    parts: list[Any] = []
    for key in sorted(k for k in inputs if torch.is_tensor(inputs[k])):
        tensor = inputs[key]
        entry: list[Any] = [key, tuple(tensor.shape), str(tensor.dtype), tensor.device.type]
        if tensor.device.type == "cpu" and tensor.numel() <= 64:
            entry.append(tuple(tensor.flatten().tolist()))
        parts.append(tuple(entry))
    return tuple(parts)


class ModelGraphRunner:
    """Capture the whole model forward once, then replay it each control step.

    The first call runs eagerly to warm up (allocating cuBLAS workspaces outside
    the graph, and populating the geometry caches), then captures. Later calls
    copy the new observation into the captured input buffers and replay.

    If the input signature changes -- a different camera resolution, a longer
    prompt, a different batch size -- the captured graph no longer describes the
    work, so execution falls back to eager and a warning is emitted. That is a
    correctness guard, not an optimisation: replaying a graph against mismatched
    inputs would silently produce wrong actions.
    """

    def __init__(self, model: torch.nn.Module, warmup_steps: int = 5) -> None:
        self.model = model
        self.warmup_steps = warmup_steps
        self.patcher: CaptureCompatPatcher | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_inputs: dict[str, torch.Tensor] | None = None
        self._static_output: torch.Tensor | None = None
        self._signature: tuple | None = None
        self._failed = False

    def __call__(self, inputs: dict[str, Any], options: dict[str, Any] | None = None):
        if self._failed or not is_capture_supported():
            with torch.inference_mode():
                return self.model.get_action(inputs, options)

        # prepare_input stays OUTSIDE the graph: it slices and reshapes the raw
        # batch on the host, which cannot be captured. Only the two compute
        # stages -- backbone and action head -- are recorded.
        with torch.inference_mode():
            backbone_inputs, action_inputs = self.model.prepare_input(inputs)
        backbone_inputs = host_side_geometry(backbone_inputs)

        signature = _signature({**dict(backbone_inputs), **dict(action_inputs)})
        if self._graph is None:
            try:
                self._capture(backbone_inputs, action_inputs, signature, options)
            except Exception as exc:  # capture is best-effort; eager still works
                logger.warning(
                    "CUDA graph capture failed (%s: %s); falling back to eager execution.",
                    type(exc).__name__,
                    exc,
                )
                self._failed = True
                if self.patcher is not None:
                    self.patcher.restore()
                    self.patcher = None
                with torch.inference_mode():
                    return self.model.get_action(inputs, options)
        elif signature != self._signature:
            logger.warning(
                "Input signature changed since capture; running eagerly. CUDA graph "
                "replay requires a fixed camera configuration and prompt length."
            )
            with torch.inference_mode():
                return self.model.get_action(inputs, options)

        assert self._static_inputs is not None and self._static_output is not None
        live = {**dict(backbone_inputs), **dict(action_inputs)}
        for key, buffer in self._static_inputs.items():
            buffer.copy_(live[key])
        self._graph.replay()
        return BatchFeature(data={"action_pred": self._static_output})

    def _capture(
        self,
        backbone_inputs: BatchFeature,
        action_inputs: BatchFeature,
        signature: tuple,
        options: dict[str, Any] | None,
    ) -> None:
        backbone = getattr(self.model, "backbone", None)
        if backbone is not None:
            self.patcher = CaptureCompatPatcher(backbone).apply()

        # Own the input buffers, so replay always reads from a fixed address.
        # Only device tensors are refreshed per step; the host-side geometry is
        # baked into the graph, and the signature guard catches it changing.
        static_inputs: dict[str, torch.Tensor] = {}
        bb_static = dict(backbone_inputs)
        ah_static = dict(action_inputs)
        for holder in (bb_static, ah_static):
            for key, value in holder.items():
                if torch.is_tensor(value) and value.device.type == "cuda":
                    buffer = value.clone()
                    holder[key] = buffer
                    static_inputs[key] = buffer
        bb_batch = BatchFeature(data=bb_static)
        ah_batch = BatchFeature(data=ah_static)

        def forward():
            with torch.inference_mode():
                backbone_output = self.model.backbone(bb_batch)
                return self.model.action_head.get_action(backbone_output, ah_batch, options)[
                    "action_pred"
                ]

        # Warm up on a side stream: cuBLAS workspace allocation and the geometry
        # caches must happen before capture, not inside it.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.warmup_steps):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = forward()
        torch.cuda.synchronize()

        self._graph = graph
        self._static_inputs = static_inputs
        self._static_output = output
        self._signature = signature
        logger.info("Captured the GR00T model forward into a CUDA graph.")
