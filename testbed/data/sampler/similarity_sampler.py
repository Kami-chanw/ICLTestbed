from functools import cached_property
import hashlib
import os
from typing import Any, Callable, Iterator, List, Optional, Sized, Union
import weakref
import io
import re

from numpy import extract
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data.sampler import Sampler, BatchSampler

import faiss
import transformers
import filelock
import dill
import xxhash
from tqdm import tqdm

from testbed.utils.fingerprint import Fingerprint, Hasher

class SimilaritySampler(Sampler[List[int]]):

    MAX_CACHE_TOP_K = 100
    _file_locks = {}
    _hash_to_cache = {}

    _instance_count = {}
    _force_reload = {}

    def __init__(
        self,
        query_set: Sized,
        support_set: Sized,
        num_shots: int,
        id_field: str,
        feat_field: str,
        extract_feat_fn: Callable[[Any], torch.Tensor],
        hash_method: str = "xxh3_64",
        cache_dir: Optional[str] = None,
        force_reload: bool = False,
        drop_last: bool = False,
    ) -> None:

        if num_shots < 1:
            raise ValueError("num_shots should be a positive integer.")

        self.hash_method = hash_method
        self.query_set = query_set
        self.support_set = support_set
        self.id_field = id_field
        self.feat_field = feat_field
        self.batch_size = num_shots + 1
        self.drop_last = drop_last
        self.extract_feat_fn = extract_feat_fn
        self.cache_dir = (
            cache_dir
            if not cache_dir is None
            else transformers.utils.default_cache_path
        )
        self._fingerprint = Fingerprint(hash_method)

        # instance-level temporary features which is equivalent to _hash_to_features[cache_filename].
        # if force_reload is True, only this will be written to cache file.
        self._features = None

        # handle class variables
        filename = self.cache_filename
        if filename not in SimilaritySampler._file_locks:  # is the first instance
            SimilaritySampler._file_locks[filename] = filelock.FileLock(
                os.path.join(self.cache_dir, filename)
            )
            SimilaritySampler._instance_count[filename] = 0
            SimilaritySampler._force_reload[self.cache_filename] |= force_reload

        SimilaritySampler._force_reload[self.cache_filename] |= force_reload
        SimilaritySampler._instance_count[self.cache_filename] += 1

        # handle cache loading and saving
        cache_file_path = os.path.join(self.cache_dir, self.cache_filename)

        def save_cache(cache_obj):
            with SimilaritySampler._file_locks[self.cache_filename]:
                    torch.save(cache_obj, cache_file_path)

        def finalize():
            cache_obj = None
            self._instance_count[self.cache_filename] -= 1
            if SimilaritySampler._force_reload[self.cache_filename]:
                if self._force_reload:
                    cache_obj = self._cached_features
            else:
                if self._instance_count[self.cache_filename] == 0:
                    cache_obj = SimilaritySampler._hash_to_cache[self.cache_filename]

            if not cache_obj is None:
                """TODO"""

        self._finalizer = weakref.finalize(self, finalize)

        new_feat_cache = {"fingerprint": self._fingerprint}
        if not os.path.exists(cache_file_path) or force_reload:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._features = new_feat_cache
        else:
            with SimilaritySampler._file_locks[self.cache_filename]:
                SimilaritySampler._hash_to_cache[self.cache_filename] = torch.load(
                    cache_file_path
                )
            if self._validate_fingerprint() == False:
                SimilaritySampler._hash_to_cache[self.cache_filename] = (
                    new_feat_cache
                )

        def build_index(name, dataset):
            id_to_idx = {}
            uncached_digests = []
            uncached_items = []
            for i in tqdm(
                range(len(dataset)),
                desc=f"Extracting features and building indices for {name}",
            ):
                feat_digest = self._hash(dataset[feat_field])
                id_to_idx[dataset[i][id_field]] = {
                    "feat_diguest": feat_digest,
                    "index": i,
                }
                if feat_digest not in self._cached_features:
                    uncached_digests.append(feat_digest)
                    uncached_items.append(dataset[feat_field])

            uncached_features = extract_feat_fn(uncached_items)
            assert isinstance(uncached_features, list)
            self._cached_features.update(
                {
                    digest: feature
                    for digest, feature in zip(uncached_digests, uncached_features)
                }
            )

            return id_to_idx

        self.qid_to_idx = build_index("query set", query_set)
        self.sid_to_idx = (
            build_index("support set", support_set)
            if query_set != support_set
            else self.qid_to_idx
        )

    def __iter__(self) -> Iterator:
        pass

    def __len__(self) -> int:
        return len(self.query_set)  # type: ignore[arg-type]

    @cached_property
    def _fingerprint(self):
        """
        Extracts the feature of the first sample in query set or cached sample as fingerprint.
        """

        sample = self.query_set[0][self.feat_field]
        feat = self.extract_feat_fn(sample)
        return self._hash(feat)

    def _validate_fingerprint(self):
        key = f"{type(self.query_set[0][self.feat_field])}_fingerprint"
        return key not in self._cached_features or self._fingerprint == self._cached_features[key]

    def _hash(self, data):
        def create_xxh_object():
            algorithms = {
                "xxh32": xxhash.xxh32,
                "xxh64": xxhash.xxh64,
                "xxh3_64": xxhash.xxh3_64,
                "xxh128": xxhash.xxh128,
                "xxh3_128": xxhash.xxh3_128,
            }

            return algorithms[self.hash_method]()

        if not isinstance(data, bytes):
            data = dill.dumps(data)

        hasher = (
            hashlib.new(self.hash_method, usedforsecurity=False)
            if self.hash_method in hashlib.algorithms_available
            else create_xxh_object()
        )

        hasher.update(data)
        return hasher.hexdigest()

    @cached_property
    def cache_filename(self):
        return str(self._hash(self.hash_method)[:32])

    @property
    def _cached_features(self):
        return (
            self._features
            if not self._features is None
            else SimilaritySampler._hash_to_cache[self.cache_filename]
        )
