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


class RewriteModelWrapper:
    TEMPLATE = "Strategy: {strategy}\nOriginal Prompt: {prompt}\nRewritten Prompt:"

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        max_length: int = 512,
        max_new_tokens: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def rewrite_batch(self, prompts: List[str], strategies: List[str], batch_size: int = 8) -> List[str]:
        if len(prompts) != len(strategies):
            raise ValueError("prompts and strategies must have the same length")

        inputs_text = [self.TEMPLATE.format(strategy=strat, prompt=prompt) for prompt, strat in zip(prompts, strategies)]

        rewritten = []
        for i in range(0, len(inputs_text), batch_size):
            batch = inputs_text[i : i + batch_size]

            encoded = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            ).to(self.device)

            outputs = self.model.generate(
                **encoded, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.pad_token_id
            )

            input_lengths = encoded["input_ids"].shape[1]
            generated_tokens = outputs[:, input_lengths:]
            decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            rewritten.extend(decoded)

        return rewritten

    def __call__(self, prompts: List[str], strategies: List[str], batch_size: int = 8) -> List[str]:
        """Alias for rewrite_batch."""
        return self.rewrite_batch(prompts, strategies, batch_size)
