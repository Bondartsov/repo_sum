from types import SimpleNamespace
from typing import Optional
import torch
import torch.nn as nn


class MockSparseModel(nn.Module):
    """
    Фиктивная sparse-модель для offline-режима SparseEncoder.
    Наследуется от torch.nn.Module и возвращает объект с .logits.
    """

    def __init__(self, vocab_size: int = 32768, boost: float = 3.0) -> None:
        super().__init__()
        self.device: str = "cpu"
        self.vocab_size: int = vocab_size
        self.boost: float = boost

    def to(self, device: Optional[str] = None) -> "MockSparseModel":
        if device is not None:
            self.device = device
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(
            (batch_size, seq_len, self.vocab_size), device=self.device
        )

        for b in range(batch_size):
            for t in range(seq_len):
                if attention_mask[b, t] == 1:
                    token_id = int(input_ids[b, t].item())
                    if 0 <= token_id < self.vocab_size:
                        logits[b, t, token_id] = self.boost

        return SimpleNamespace(logits=logits)