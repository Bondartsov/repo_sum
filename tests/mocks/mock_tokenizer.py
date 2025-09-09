import hashlib
import torch
from typing import List, Union, Dict, Any


class _BatchEncoding(dict):
    def __init__(self, input_ids: List[List[int]], attention_mask: List[List[int]]) -> None:
        max_len: int = max(len(seq) for seq in input_ids) if input_ids else 1
        padded_ids: List[List[int]] = [seq + [0] * (max_len - len(seq)) for seq in input_ids]
        padded_mask: List[List[int]] = [mask + [0] * (max_len - len(mask)) for mask in attention_mask]
        self.input_ids: torch.Tensor = torch.tensor(padded_ids, dtype=torch.long)
        self.attention_mask: torch.Tensor = torch.tensor(padded_mask, dtype=torch.long)
        super().__init__({"input_ids": self.input_ids, "attention_mask": self.attention_mask})

    def to(self, device: Union[str, torch.device]) -> "_BatchEncoding":
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self["input_ids"] = self.input_ids
        self["attention_mask"] = self.attention_mask
        return self


class MockTokenizer:
    pad_token_id: int = 0
    unk_token_id: int = 1

    def __init__(self, vocab_size: int = 30522, max_length: int = 32) -> None:
        self.vocab_size: int = vocab_size
        self.max_length: int = max_length

    def _token_to_id(self, token: str) -> int:
        if not token:
            return self.unk_token_id
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        return 2 + (int(h, 16) % (self.vocab_size - 2))

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens: List[str] = text.lower().split()
        if not tokens:
            return [self.unk_token_id]
        ids: List[int] = [self._token_to_id(tok) for tok in tokens]
        return ids[: self.max_length]

    def __call__(
        self,
        texts: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
        max_length: Union[int, None] = None,
    ) -> _BatchEncoding:
        if isinstance(texts, str):
            texts = [texts]

        max_len: int = max_length or self.max_length
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []

        for text in texts:
            ids: List[int] = self.encode(text)[:max_len]
            mask: List[int] = [1] * len(ids)
            input_ids.append(ids)
            attention_mask.append(mask)

        return _BatchEncoding(input_ids, attention_mask)