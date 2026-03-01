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

from dataclasses import dataclass
from typing import Optional

from .ppo_config import PPOConfig


@dataclass
class RedTeamPPOConfig(PPOConfig):

    # Model paths
    rewrite_model_path: Optional[str] = None
    target_model_path: Optional[str] = None
    downgrade_model_path: Optional[str] = None

    # Diversity settings
    diversity_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    diversity_threshold: float = 0.85
    diversity_history_size: int = 1000

    # Judge API settings
    sglang_endpoint: str = "http://localhost:30000"
    consistency_judge_model: str = "default"
    safety_judge_model: str = "default"
    max_judge_requests: int = 100000
    judge_timeout: int = 30

    # Inference batch sizes
    rewrite_batch_size: int = 8
    target_batch_size: int = 8

    # Reward weights
    diversity_penalty: float = 1.0
    consistency_penalty: float = 0.5
    downgrade_safe_reward: float = 1.0
    target_safe_reward: float = 1.0
