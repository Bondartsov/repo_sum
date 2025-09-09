import torch


class MockBatchEncoding:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
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
        # Возвращаем фиктивные id (например, индексы символов)
        return [i + 1 for i, _ in enumerate(text)]