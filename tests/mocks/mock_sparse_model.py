import torch

class MockOutput:
    def __init__(self, batch_size, seq_len, hidden_dim, device):
        self.last_hidden_state = torch.ones((batch_size, seq_len, hidden_dim), device=device)
        self.logits = torch.ones((batch_size, seq_len, hidden_dim), device=device)

class MockSparseModel:
    """
    Фиктивная модель для offline-режима SparseEncoder.
    Имеет метод .to(device), чтобы имитировать поведение torch.nn.Module.
    """
    def __init__(self, hidden_dim: int = 16):
        self.device = "cpu"
        self.hidden_dim = hidden_dim

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        batch_size, seq_len = input_ids.shape
        return MockOutput(batch_size, seq_len, self.hidden_dim, self.device)