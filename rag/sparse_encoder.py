from typing import List, Dict, Any, Optional
import torch
import logging
import os
logging.getLogger("transformers").setLevel(logging.ERROR)
from transformers import AutoTokenizer, AutoModelForMaskedLM

try:
    from tests.mocks.mock_tokenizer import MockTokenizer
    from tests.mocks.mock_sparse_model import MockSparseModel
    from tests.mocks import is_socket_disabled
except ImportError:
    MockTokenizer = None
    MockSparseModel = None
    is_socket_disabled = lambda: False


class SparseEncoder:
    """
    Базовый sparse encoder для RAG.
    Использует трансформер-модель (например, SPLADE или BERT) для генерации разреженных векторов.
    """

    def __init__(self, model_name: str = "bert-base-uncased", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if ((os.environ.get("MOCK_MODE") == "1") or is_socket_disabled()) and MockTokenizer is not None:
            logging.info("SparseEncoder: offline/mock режим активен, используется MockTokenizer (model=None)")
            self.tokenizer = MockTokenizer()
            self.model = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=True).to(self.device)
            self.model.eval()

    def encode(self, texts: List[str]) -> List[Dict[int, float]]:
        """
        Кодирует список текстов в sparse-вектора.
        Возвращает список словарей {token_id: weight}.
        """
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits  # [batch, seq_len, vocab_size]

        # Простейший способ: берём max по seq_len → получаем важность токена
        scores, _ = torch.max(logits, dim=1)  # [batch, vocab_size]
        scores = torch.log1p(torch.relu(scores))  # сглаживание

        sparse_vectors: List[Dict[int, float]] = []
        for vec in scores:
            nonzero = torch.nonzero(vec > 0, as_tuple=True)[0]
            sparse_dict = {int(idx): float(vec[idx].cpu().item()) for idx in nonzero}
            sparse_vectors.append(sparse_dict)

        return sparse_vectors

    def save(self, path: str) -> None:
        """Сохраняет модель и токенизатор."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "SparseEncoder":
        """Загружает модель и токенизатор."""
        instance = cls.__new__(cls)
        instance.model_name = path
        instance.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForMaskedLM.from_pretrained(path).to(instance.device)
        instance.model.eval()
        return instance