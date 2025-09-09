class MockSparseModel:
    """
    Фиктивная модель для offline-режима SparseEncoder.
    Имеет метод .to(device), чтобы имитировать поведение torch.nn.Module.
    """

    def __init__(self):
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self