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

"""The ``process_observation`` state-dimension guard checks the state axis.

``process_observation`` takes *batched* observations, so the concatenated state
tensor is ``(B, T, D)``: ``T`` is the state history length and ``D`` is the state
dimension. The padding immediately below the guard is computed from ``shape[-1]``
(``D``), so the guard has to inspect the same axis.

It previously inspected ``shape[1]`` (``T``). With the default
``state_history_length`` of 1 that is always 1, so the guard was vacuous, and an
embodiment whose state dimension exceeded ``max_state_dim`` fell through to
``torch.zeros`` with a negative width -- surfacing as
``RuntimeError: zeros: Dimension size must be non-negative`` instead of the
message written for exactly that case.

CPU-only: uses the checked-in processor fixture with a mocked VLM processor, so
no checkpoint download or GPU is required.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from gr00t.data.embodiment_tags import EmbodimentTag
import numpy as np
import pytest


FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "processor_config"
EMBODIMENT = "libero_sim"


@pytest.fixture
def processor():
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    mock_vlm = MagicMock()
    mock_vlm.apply_chat_template.return_value = "mock text"
    mock_vlm.tokenizer.padding_side = "left"

    with patch(
        "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor",
        return_value=mock_vlm,
    ):
        proc = Gr00tN1d7Processor.from_pretrained(FIXTURE_DIR)
    proc.eval()
    return proc


@pytest.fixture
def proc_config():
    with open(FIXTURE_DIR / "processor_config.json") as f:
        return json.load(f)["processor_kwargs"]


def _make_batched_observation(proc_config, batch_size: int = 2, history: int = 1):
    """Batched observation dict for ``process_observation``: video (B,T,H,W,C), state (B,T,D)."""
    mc = proc_config["modality_configs"][EMBODIMENT]
    with open(FIXTURE_DIR / "statistics.json") as f:
        statistics = json.load(f)

    observation = {}
    for view in mc["video"]["modality_keys"]:
        observation[f"video.{view}"] = np.random.randint(
            0, 255, (batch_size, history, 256, 256, 3), dtype=np.uint8
        )
    for key in mc["state"]["modality_keys"]:
        dim = len(statistics[EMBODIMENT]["state"][key]["min"])
        observation[f"state.{key}"] = np.random.randn(batch_size, history, dim).astype(np.float32)
    language_key = mc["language"]["modality_keys"][0]
    observation[language_key] = ["pick up the apple"] * batch_size

    state_dim = sum(
        len(statistics[EMBODIMENT]["state"][key]["min"]) for key in mc["state"]["modality_keys"]
    )
    return observation, state_dim


def test_state_is_padded_to_max_state_dim(processor, proc_config):
    """Baseline: a within-limits state is padded on the last axis, batch/history preserved."""
    observation, _ = _make_batched_observation(proc_config, batch_size=2, history=1)

    out = processor.process_observation(observation, EmbodimentTag(EMBODIMENT))

    assert out["state"].shape == (2, 1, proc_config["max_state_dim"])


def test_oversized_state_dim_raises_assertion_not_negative_zeros(processor, proc_config):
    """An oversized state dim must hit the guard, not a negative-width torch.zeros.

    Regression test: the guard used to read the history axis, so this raised
    ``RuntimeError: zeros: Dimension size must be non-negative``.
    """
    observation, state_dim = _make_batched_observation(proc_config, batch_size=2, history=1)
    processor.max_state_dim = state_dim - 1

    with pytest.raises(AssertionError, match=r"State dimension \d+ exceeds max_state_dim"):
        processor.process_observation(observation, EmbodimentTag(EMBODIMENT))


def test_guard_reports_the_state_dim_not_the_history_length(processor, proc_config):
    """The message must name the offending state dim, so it points at the real problem."""
    observation, state_dim = _make_batched_observation(proc_config, batch_size=2, history=1)
    processor.max_state_dim = state_dim - 1

    with pytest.raises(AssertionError) as excinfo:
        processor.process_observation(observation, EmbodimentTag(EMBODIMENT))

    assert f"State dimension {state_dim} exceeds max_state_dim {state_dim - 1}" in str(
        excinfo.value
    )


def test_guard_is_insensitive_to_state_history_length(processor, proc_config):
    """A history longer than max_state_dim is legal and must not trip the guard.

    This isolates the axis: the history length (16) exceeds ``max_state_dim``
    while the state dim (``state_dim``) does not. Reading the wrong axis makes the
    guard fire spuriously here and reject a perfectly valid observation.
    """
    history = 16
    observation, state_dim = _make_batched_observation(proc_config, batch_size=1, history=history)
    processor.max_state_dim = state_dim
    assert history > processor.max_state_dim, "test must put the two axes on opposite sides"

    out = processor.process_observation(observation, EmbodimentTag(EMBODIMENT))

    assert out["state"].shape == (1, history, state_dim)
