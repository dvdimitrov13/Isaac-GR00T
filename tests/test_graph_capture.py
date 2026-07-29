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

"""Tests for CUDA graph capture of the model forward."""

from gr00t.model.graph_capture import (
    ModelGraphRunner,
    _signature,
    host_side_geometry,
    is_capture_supported,
)
import torch
from transformers.feature_extraction_utils import BatchFeature


def test_host_side_geometry_moves_only_grid_tensors():
    inputs = BatchFeature(
        data={
            "image_grid_thw": torch.tensor([[1, 16, 16], [1, 16, 16]]),
            "input_ids": torch.zeros(1, 8, dtype=torch.long),
        }
    )
    out = host_side_geometry(inputs)
    assert out["image_grid_thw"].device.type == "cpu"
    # values must survive the move untouched
    assert torch.equal(out["image_grid_thw"], inputs["image_grid_thw"].cpu())
    assert out["input_ids"].device == inputs["input_ids"].device


def test_signature_distinguishes_shape_and_dtype():
    base = {"a": torch.zeros(1, 4), "b": torch.zeros(2, 2)}
    assert _signature(base) == _signature({"a": torch.zeros(1, 4), "b": torch.zeros(2, 2)})
    assert _signature(base) != _signature({"a": torch.zeros(1, 5), "b": torch.zeros(2, 2)})
    assert _signature(base) != _signature(
        {"a": torch.zeros(1, 4, dtype=torch.float64), "b": torch.zeros(2, 2)}
    )


def test_signature_tracks_host_geometry_values():
    """A camera resolution change shows up only in the values of the small
    host-side geometry tensor, and must invalidate a captured graph."""
    a = {"image_grid_thw": torch.tensor([[1, 16, 16]])}
    b = {"image_grid_thw": torch.tensor([[1, 32, 32]])}
    assert _signature(a) != _signature(b)


def test_runner_falls_back_to_eager_without_cuda(monkeypatch):
    """On a CPU-only build the runner must still produce actions."""
    calls = []

    class FakeModel:
        def get_action(self, inputs, options=None):
            calls.append(inputs)
            return BatchFeature(data={"action_pred": torch.zeros(1, 4)})

    monkeypatch.setattr("gr00t.model.graph_capture.is_capture_supported", lambda: False)
    runner = ModelGraphRunner(FakeModel())
    out = runner({"x": torch.zeros(1, 2)})
    assert out["action_pred"].shape == (1, 4)
    assert len(calls) == 1


def test_runner_falls_back_when_capture_raises(monkeypatch):
    """Capture is best-effort: a failure must degrade to eager, not crash."""
    if not is_capture_supported():
        return

    class ExplodingModel(torch.nn.Module):
        def prepare_input(self, inputs):
            return BatchFeature(data=dict(inputs)), BatchFeature(data={})

        def get_action(self, inputs, options=None):
            return BatchFeature(data={"action_pred": torch.zeros(1, 4)})

    runner = ModelGraphRunner(ExplodingModel())
    monkeypatch.setattr(
        runner, "_capture", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    out = runner({"x": torch.zeros(1, 2, device="cuda")})
    assert out["action_pred"].shape == (1, 4)
    assert runner._failed
