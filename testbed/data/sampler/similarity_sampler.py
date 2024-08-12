from typing import Iterator, Sized
import torch
from torch.utils.data.sampler import Sampler
import clip
import faiss
from pathlib import Path


class SimilaritySampler(Sampler):
    SUPPORTED_METHODS = [
        "similar-question",
        "similar-answer",
        "similar-question-answer",
    ]

    MAX_CACHE_TOP_K = 100

    def __init__(
        self, data_source: Sized, num_shots, method, cache_dir=".cache", force_reload=True, device=None
    ) -> None:
        super().__init__(data_source)

        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method, it should be one of {self.SUPPORTED_METHODS}, got {method}")

        model, preprocess = clip.load("ViT-B/32", device=device)
        self.data_source = data_source

        if not Path(cache_dir).exists() or force_reload:
            pass

    def __iter__(self) -> Iterator:
        pass

    def __len__(self) -> int:
        return len(self.data_source)
