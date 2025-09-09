class MockTokenizer:
    """
    Простейший mock-токенизатор для offline-режима.
    Разбивает текст по пробелам и назначает id = индекс слова.
    """

    def __call__(self, texts, padding=True, truncation=True, return_tensors=None):
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

        # эмуляция поведения HuggingFace: возвращаем dict
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }