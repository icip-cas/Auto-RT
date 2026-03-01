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

from collections import deque
from typing import List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class DiversityChecker:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        history_size: int = 1000,
        similarity_threshold: float = 0.85,
        device: str = "cuda",
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.history = deque(maxlen=history_size)
        self.threshold = similarity_threshold
        self.device = device

    def check_batch(self, strategies: List[str]) -> Tuple[torch.Tensor, List[float]]:
        new_embeddings = self.model.encode(
            strategies, convert_to_tensor=True, show_progress_bar=False, device=self.device
        )

        if len(self.history) == 0:
            for emb in new_embeddings.cpu().numpy():
                self.history.append(emb)
            return torch.ones(len(strategies), dtype=torch.bool), [0.0] * len(strategies)

        history_embeddings = torch.tensor(np.array(list(self.history)), device=new_embeddings.device)

        similarities = torch.nn.functional.cosine_similarity(
            new_embeddings.unsqueeze(1), history_embeddings.unsqueeze(0), dim=2
        )

        max_sims, _ = similarities.max(dim=1)
        diversity_mask = max_sims < self.threshold

        for i, is_diverse in enumerate(diversity_mask):
            if is_diverse:
                self.history.append(new_embeddings[i].cpu().numpy())

        return diversity_mask, max_sims.cpu().tolist()

    def reset_history(self):
        self.history.clear()

    def get_history_size(self) -> int:
        return len(self.history)
