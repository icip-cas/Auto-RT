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


class TargetModelWrapper:
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        max_length: int = 512,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        # Ensure model is in eval mode and frozen
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def generate_responses(self, queries: List[str], batch_size: int = 8) -> List[str]:
        responses = []

        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            ).to(self.device)

            # Generate
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # Decode only the generated part (exclude input)
            input_lengths = encoded["input_ids"].shape[1]
            generated_tokens = outputs[:, input_lengths:]
            decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            responses.extend(decoded)

        return responses

    def __call__(self, queries: List[str], batch_size: int = 8) -> List[str]:
        return self.generate_responses(queries, batch_size)
