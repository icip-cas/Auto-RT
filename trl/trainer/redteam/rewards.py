# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
import torch
from ..redteam_ppo_config import RedTeamPPOConfig


def compute_rewards(
    diversity_mask: torch.Tensor,
    consistency_scores: List[int],
    downgrade_safety: List[int],
    target_safety: List[int],
    config: RedTeamPPOConfig,
) -> torch.Tensor:
    batch_size = len(diversity_mask)
    rewards = torch.zeros(batch_size, dtype=torch.float32)

    for i in range(batch_size):
        reward = 0.0

        if not diversity_mask[i]:
            reward -= config.diversity_penalty

        if consistency_scores[i] != 1:
            reward -= config.consistency_penalty

        if downgrade_safety[i] == 0:
            pass
        elif downgrade_safety[i] == 1:
            reward += config.downgrade_safe_reward

            if target_safety[i] == 1:
                reward += config.target_safe_reward

        rewards[i] = reward

    return rewards
