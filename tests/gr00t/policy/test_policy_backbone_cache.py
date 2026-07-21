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

"""Tests for Gr00tPolicy backbone caching.

Uses mocked model and processor to avoid downloading checkpoints.
Verifies cache hit/miss behavior, correctness, and invalidation.
"""

from unittest.mock import MagicMock, patch

from gr00t.data.types import ModalityConfig
import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature


EMBODIMENT = "libero_sim"
VIDEO_KEYS = ["observation.images.rgb.head_256_256", "observation.images.rgb.left_wrist_256_256"]
STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
LANGUAGE_KEY = "annotation.human.action.task_description"


def _build_modality_configs():
    return {
        EMBODIMENT: {
            "video": ModalityConfig(delta_indices=[0], modality_keys=VIDEO_KEYS),
            "state": ModalityConfig(delta_indices=[0], modality_keys=STATE_KEYS),
            "action": ModalityConfig(delta_indices=list(range(16)), modality_keys=ACTION_KEYS),
            "language": ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE_KEY]),
        }
    }


def _make_observation(batch_size=1, seed=None):
    rng = np.random.RandomState(seed)
    return {
        "video": {
            k: rng.randint(0, 255, (batch_size, 1, 256, 256, 3), dtype=np.uint8) for k in VIDEO_KEYS
        },
        "state": {k: rng.randn(batch_size, 1, 1).astype(np.float32) for k in STATE_KEYS[:-1]}
        | {"gripper": rng.randn(batch_size, 1, 2).astype(np.float32)},
        "language": {
            LANGUAGE_KEY: [["pick up the apple"]] * batch_size,
        },
    }


def _make_policy(cache_backbone: bool):
    """Build a Gr00tPolicy with mocked model/processor."""
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.device = torch.device("cpu")
    mock_model.dtype = torch.bfloat16

    action_pred = torch.randn(1, 16, 7)
    backbone_outputs = BatchFeature(data={"backbone_features": torch.randn(1, 64, 2048)})

    mock_model.get_action = MagicMock(
        return_value=BatchFeature(data={"action_pred": action_pred.clone()})
    )
    mock_model.get_action_cached = MagicMock(
        return_value=(
            BatchFeature(data={"action_pred": action_pred.clone()}),
            backbone_outputs,
        )
    )

    mock_processor = MagicMock()
    mock_processor.modality_configs = _build_modality_configs()
    mock_processor.get_modality_configs.return_value = _build_modality_configs()
    mock_processor.state_action_processor = MagicMock()
    mock_processor.action_dim = {EMBODIMENT: 7}
    mock_processor.max_action_dim = 128
    mock_processor.max_action_horizon = 50
    mock_processor.eval = MagicMock()
    mock_processor.training = False
    mock_processor.collator = MagicMock()

    def fake_decode_action(action, embodiment_tag, state=None):
        return {k: np.zeros((1, 16, 1), dtype=np.float32) for k in ACTION_KEYS}

    mock_processor.decode_action = MagicMock(side_effect=fake_decode_action)

    with (
        patch("gr00t.policy.gr00t_policy.AutoModel") as MockAutoModel,
        patch("gr00t.policy.gr00t_policy.AutoProcessor") as MockAutoProcessor,
        patch("pathlib.Path.is_dir", return_value=False),
        patch("pathlib.Path.exists", return_value=True),
    ):
        MockAutoModel.from_pretrained.return_value = mock_model
        MockAutoProcessor.from_pretrained.return_value = mock_processor

        from gr00t.policy.gr00t_policy import Gr00tPolicy

        p = Gr00tPolicy(
            embodiment_tag=EMBODIMENT,
            model_path="/fake/path",
            device="cpu",
            cache_backbone=cache_backbone,
        )
    return p


class TestBackboneCacheDisabled:
    def test_default_cache_off(self):
        policy = _make_policy(cache_backbone=False)
        assert policy.cache_backbone is False

    def test_uses_get_action_not_cached(self):
        policy = _make_policy(cache_backbone=False)
        obs = _make_observation()
        policy.get_action(obs)
        policy.model.get_action.assert_called_once()
        policy.model.get_action_cached.assert_not_called()

    def test_info_has_no_cache_hit_key(self):
        policy = _make_policy(cache_backbone=False)
        obs = _make_observation()
        _, info = policy.get_action(obs)
        assert "backbone_cache_hit" not in info


class TestBackboneCacheEnabled:
    def test_first_call_is_cache_miss(self):
        policy = _make_policy(cache_backbone=True)
        obs = _make_observation(seed=42)
        _, info = policy.get_action(obs)
        assert info["backbone_cache_hit"] is False
        call_kwargs = policy.model.get_action_cached.call_args
        assert call_kwargs.kwargs["cached_backbone_outputs"] is None

    def test_same_observation_is_cache_hit(self):
        policy = _make_policy(cache_backbone=True)
        obs = _make_observation(seed=42)
        policy.get_action(obs)
        _, info = policy.get_action(obs)
        assert info["backbone_cache_hit"] is True
        call_kwargs = policy.model.get_action_cached.call_args
        assert call_kwargs.kwargs["cached_backbone_outputs"] is not None

    def test_different_video_is_cache_miss(self):
        policy = _make_policy(cache_backbone=True)
        obs1 = _make_observation(seed=42)
        obs2 = _make_observation(seed=99)
        policy.get_action(obs1)
        _, info = policy.get_action(obs2)
        assert info["backbone_cache_hit"] is False
        call_kwargs = policy.model.get_action_cached.call_args
        assert call_kwargs.kwargs["cached_backbone_outputs"] is None

    def test_same_video_different_state_is_cache_hit(self):
        """Same image but different state should reuse cached backbone features."""
        policy = _make_policy(cache_backbone=True)
        obs = _make_observation(seed=42)
        policy.get_action(obs)
        # Change state but keep video identical
        for key in obs["state"]:
            obs["state"][key] = obs["state"][key] + 1.0
        _, info = policy.get_action(obs)
        assert info["backbone_cache_hit"] is True

    def test_reset_clears_cache(self):
        policy = _make_policy(cache_backbone=True)
        obs = _make_observation(seed=42)
        policy.get_action(obs)
        assert policy._cached_backbone_outputs is not None
        policy.reset()
        assert policy._cached_backbone_outputs is None
        assert policy._cached_video_fingerprint is None
        _, info = policy.get_action(obs)
        assert info["backbone_cache_hit"] is False

    def test_new_array_same_content_is_cache_hit(self):
        """Even if the caller creates a new ndarray with identical content, cache should hit."""
        policy = _make_policy(cache_backbone=True)
        obs1 = _make_observation(seed=42)
        policy.get_action(obs1)
        obs2 = _make_observation(seed=42)
        _, info = policy.get_action(obs2)
        assert info["backbone_cache_hit"] is True

    def test_get_action_cached_called_with_correct_args(self):
        policy = _make_policy(cache_backbone=True)
        obs = _make_observation(seed=42)
        policy.get_action(obs)
        assert policy.model.get_action_cached.call_count == 1
        policy.get_action(obs)
        assert policy.model.get_action_cached.call_count == 2
