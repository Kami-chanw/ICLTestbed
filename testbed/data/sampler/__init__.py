from typing import Iterable, Iterator, List
from torch.utils.data.sampler import Sampler, BatchSampler


class ConcatSampler(Sampler[List[int]]):
    """
    Concatenates multiple samplers, each corresponding to a different dataset.

    This sampler is useful when working with a `ConcatDataset`, allowing you to sample from multiple datasets
    while adjusting the indices so that they are correctly mapped to the corresponding dataset.

    Args:
        samplers (`Iterable[Sampler]`):
            An iterable of samplers, each associated with a different dataset.
        cumulative_dataset_sizes (`List[int]`):
            A list of cumulative sizes of the datasets, used to adjust the indices.
            This list should contain the cumulative sum of dataset sizes.
    """

    def __init__(
        self, samplers: Iterable[Sampler], cumulative_dataset_sizes: List[int]
    ):
        self.samplers = list(samplers)
        sample_indices = [next(iter(sampler)) for sampler in self.samplers]
        self.batch_size = sum(
            len(idx) if isinstance(idx, list) else 1 for idx in sample_indices
        )
        self.cumulative_indices = [0] + cumulative_dataset_sizes[:-1]

    def __iter__(self) -> Iterator[List[int]]:
        sampler_iters = [iter(sampler) for sampler in self.samplers]
        while True:
            try:
                sample_indices = [next(it) for it in sampler_iters]
                batch = []
                for i, mini in enumerate(sample_indices):
                    if isinstance(mini, list):
                        batch.extend(idx + self.cumulative_indices[i] for idx in mini)
                    else:
                        batch.append(mini + self.cumulative_indices[i])

                yield batch
            except StopIteration:
                break

    def __len__(self):
        return min(len(sampler) for sampler in self.samplers)


class MultiBatchSampler(BatchSampler):
    """
    Repeat sampling from BatchSampler multiple times to yield a larger, merged batch of indices.

    This sampler is designed to take an existing `BatchSampler` and yield larger batches by merging
    smaller batches together. It is particularly useful when you need to create batches that are larger
    than what the underlying `BatchSampler` would normally yield.

    Args:
        sampler (`BatchSampler`):
            The original batch sampler that yields smaller batches of indices.
        merge_size (`int`):
            The number of smaller batches to merge together.
        drop_last (`bool`):
            If `True`, drop the last incomplete merged batch; if `False`, return it.
    """

    def __init__(self, sampler: BatchSampler, merge_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if (
            not isinstance(merge_size, int)
            or isinstance(merge_size, bool)
            or merge_size <= 0
        ):
            raise ValueError(
                "merge_size should be a positive integer value, "
                "but got merge_size={}".format(merge_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got "
                "drop_last={}".format(drop_last)
            )
        if isinstance(next(iter(sampler)), int):
            raise ValueError("batch_sampler should yield a list of int.")

        self.sampler = sampler
        self.batch_size = merge_size * sampler.batch_size
        self.merge_size = merge_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.merge_size)]
                    # flatten the merged batches
                    yield [idx for mini in batch for idx in mini]
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for indices in self.sampler:
                for idx in indices:
                    batch[idx_in_batch] = idx
                    idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.merge_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.merge_size - 1) // self.merge_size  # type: ignore[arg-type]
