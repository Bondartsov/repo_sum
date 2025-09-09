import torch


class MockBatchEncoding(dict):
    def __init__(self, input_ids, attention_mask):
        # паддинг до максимальной длины
        max_len = max(len(seq) for seq in input_ids) if input_ids else 1
        padded_ids = [seq + [0] * (max_len - len(seq)) for seq in input_ids]
        padded_mask = [mask + [0] * (max_len - len(mask)) for mask in attention_mask]
        self.input_ids = torch.tensor(padded_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(padded_mask, dtype=torch.long)
        super().__init__({"input_ids": self.input_ids, "attention_mask": self.attention_mask})

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self["input_ids"] = self.input_ids
        self["attention_mask"] = self.attention_mask
        return self


class MockTokenizer:
    """
    Простейший mock-токенизатор для offline-режима.
    Совместим с интерфейсом HuggingFace AutoTokenizer.
    """

    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]

        input_ids = []
        attention_mask = []

        for text in texts:
            tokens = text.split()
            ids = list(range(1, len(tokens) + 1))  # простая схема id
            mask = [1] * len(tokens)

            input_ids.append(ids)
            attention_mask.append(mask)

        return MockBatchEncoding(input_ids, attention_mask)

    def encode(self, text, add_special_tokens=False):
        # Возвращаем фиктивные id (например, индексы слов)
        return [i + 1 for i, _ in enumerate(text.split())]