import re
from typing import Any, Callable, Dict, List, Optional, Union
import inspect
from torch.utils.data import (
    Sampler,
    DataLoader,
    BatchSampler,
    SequentialSampler,
    RandomSampler,
    Dataset,
    ConcatDataset,
)
from .sampler import MultiBatchSampler, ConcatSampler
from .common import (
    register_dataset_retriever,
    register_postprocess,
    POSTPROCESS_MAPPING,
    DATASET_RETRIEVER_MAPPING,
)
from . import coco, vqav2, ok_vqa, hateful_memes


def prepare_input(
    dataset_sources: Union[str, List[str]],
    batch: List[List[Dict[str, Any]]],
    instruction: Optional[str] = None,
) -> List[List[Any]]:
    """
    Prepares a batch of data by using dataset-specific retriever functions based on the source of the data.
    If a dataset source is provided, the corresponding retriever function for that dataset will be used to process
    the items in the batch. If no dataset source is provided, the function will raise an error.

    Args:
        dataset_sources (`Union[str, List[str]]`):
            A string or a list of strings representing the dataset source for each context in the batch.
            If a single string is provided, all contexts are assumed to come from that dataset. If a list is
            provided, its length must match the number of contexts in the batch. Each dataset-specific retriever function
            is called accordingly for each item in the context.
        batch (`List[List[Dict[str, Any]]]`):
            A batch of data where each element is a list of dictionaries, representing a context.
        instruction (`Optional[str]`, *optional*):
            A string instruction to prepend to each context, typically used to guide the task
            or provide a hint. Defaults to `None`.

    Returns:
        `List[List[Any]]`: A batch of processed data where each context has been formatted using
        the dataset-specific retriever functions, and optionally prepended with an instruction.

    Raises:
        `ValueError`: If the length of the dataset_sources list does not match the batch size.
        `KeyError`: If the dataset_source is not recognized.
    """
    if isinstance(dataset_sources, str):
        dataset_sources = [dataset_sources] * len(batch)
    elif isinstance(dataset_sources, list):
        if len(dataset_sources) != len(batch[0]):
            raise ValueError(
                "Length of dataset_sources list must match the number of items in the context."
            )

    for source in dataset_sources:
        if source not in DATASET_RETRIEVER_MAPPING:
            raise ValueError(
                f"Dataset source '{source}' not found in the internal mapping."
            )

    return_annotations = [
        inspect.signature(DATASET_RETRIEVER_MAPPING[source]).return_annotation
        for source in dataset_sources
    ]
    if len(set(return_annotations)) > 1:
        raise ValueError("All dataset retrievers must return the same type.")

    batch_context, batch_additional_outputs = [], []

    for source, context in zip(dataset_sources, batch):
        messages, additional_outputs = [], []
        if instruction is not None:
            messages.append({"role": "instruction", "content": instruction})

        for item in context:
            retriever = DATASET_RETRIEVER_MAPPING[source]
            prepared_item = retriever(item, item == context[-1])

            if isinstance(prepared_item, tuple):
                msg, *rest = prepared_item
                messages.extend(msg)
                additional_outputs.append(tuple(rest))
            else:
                messages.extend(prepared_item)

        batch_context.append(messages)
        batch_additional_outputs.append(additional_outputs)

    if batch_additional_outputs[0] and all(
        isinstance(output, tuple) for output in batch_additional_outputs[0]
    ):
        if (
            len(
                set(
                    len(output)
                    for outputs in batch_additional_outputs
                    for output in outputs
                )
            )
            != 1
        ):
            raise RuntimeError(
                "Inconsistent number of additional outputs across different contexts."
            )

        return batch_context, *[
            list(outputs)
            for outputs in zip(
                *[
                    [list(i) for i in zip(*additional_outputs)]
                    for additional_outputs in batch_additional_outputs
                ]
            )
        ]

    return batch_context


def postprocess_generation(
    dataset: str,
    predictions: Union[str, List[str]],
    stop_words: Optional[List[str]] = None,
) -> Union[str, List[str]]:
    """
    Post-processes generated predictions by applying dataset-specific text normalization techniques.

    This function processes a single prediction or a list of predictions, allowing for optional truncation
    based on stop words. It returns the processed prediction(s) either as a string or a list, depending on the input type.


    Args:
        dataset (`str`, *optional*):
            The name of the dataset to apply dataset-specific postprocessing.
        predictions (`Union[str, List[str]]`):
            The generated predictions, either as a single string or a list of strings.
        stop_words (`Optional[List[str]]`, *optional*):
            A list of stop words used to trim the predictions. If provided, the predictions
            are split at the first occurrence of any stop word. Defaults to `None`.

    Returns:
        `Union[str, List[str]]`: The post-processed predictions. If the input was a single
        prediction string, a single processed string is returned. If the input was a list of
        predictions, a list of processed strings is returned.

    Raises:
        `KeyError`: If a dataset is provided but is not found in the internal mapping.
    """
    is_batched = True
    if isinstance(predictions, str):
        predictions = [predictions]
        is_batched = False

    def preprocess(pred, stop_words):
        if stop_words is not None:
            pred = re.split("|".join(stop_words), pred, 1)[0]
        return pred.strip()

    process_fn = (
        POSTPROCESS_MAPPING[dataset]
        if dataset in POSTPROCESS_MAPPING
        else lambda pred: pred
    )
    result = [process_fn(preprocess(pred, stop_words)) for pred in predictions]

    if is_batched:
        return result
    else:
        return result[0]


def prepare_dataloader(
    datasets: Union[Dataset, List[Dataset]],
    batch_size: int,
    num_shots: Optional[int] = None,
    num_per_dataset: Optional[Union[int, List[int]]] = None,
    collate_fn: Optional[Callable[[List[List[Any]]], Any]] = None,
    samplers: Optional[Union[Sampler, List[Sampler]]] = None,
    **kwargs,
) -> DataLoader:
    """
    Prepare a DataLoader for in-context learning using single or multiple datasets.

    If `collate_fn` is `None`, the DataLoader will return batches as a list of shape
    `[batch_size, num_shots + 1]`, where each sub-list contains `num_shots` in-context
    examples and one query. The function supports sampling from single or multiple datasets
    according to the specified number of examples per dataset.

    Args:
        datasets (`Dataset` or `List[Dataset]`):
            A `Dataset` object or list of datasets to load data from.
        batch_size (`int`):
            Number of sub-lists (each with `num_shots + 1` items) per batch.
        num_shots (`int`, *optional*):
            Total number of in-context examples per sub-list. It can be None if `num_per_dataset` is specified.
        num_per_dataset (`int` or `List[int]`, *optional*):
            Number of items to sample from each dataset, whose sum should be equal `to num_shots` + 1.
            It can be `None` if only one dataset is provided.
        collate_fn (`Callable`, *optional*):
            If `collate_fn` is `None`, the DataLoader will return batches as a list of shape
            `[batch_size, num_shots + 1]`, where each sub-list contains `num_shots` in-context
            examples and one query. The function supports sampling from single or multiple datasets
            according to the specified number of examples per dataset.
        samplers (`Samper` or `List[Sampler`], *optional*):
            Samplers for each dataset. If not specified, `SequentialSampler` will be applied.
        **kwargs: Additional arguments for DataLoader.

    Returns:
        DataLoader: Configured DataLoader for in-context learning across multiple datasets.

    Example:
        >>> dataset1 = range(5)
        >>> dataset2 = range(5, 10)
        >>> dataloader = prepare_dataloader(
        >>>     datasets=[dataset1, dataset2],
        >>>     batch_size=2,
        >>>     num_shots=2,
        >>>     num_per_dataset=[1, 2]
        >>> )
        >>> for batch in dataloader:
        >>>     print(batch)
        [[0, 5, 6], [1, 7, 8]]
    """
    # extract options tht mutually exclusive with batch_sampler
    drop_last = kwargs.pop("drop_last", False)
    shuffle = kwargs.pop("shuffle", False)
    sampler = kwargs.pop("sampler", None)
    if not sampler is None:
        if not samplers is None:
            raise ValueError("Cannot specify sampler and samplers at the same time.")
        else:
            samplers = [sampler]

    def batchilize_sampler(dataset, sampler, minibatch_size):
        if sampler is None:
            sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        sample_idx = next(iter(sampler))
        if isinstance(sample_idx, int):
            return BatchSampler(sampler, minibatch_size, True)
        elif (
            isinstance(sample_idx, list)
            and all(isinstance(idx, int) for idx in sample_idx)
            and len(sample_idx) == minibatch_size
        ):
            return sampler
        else:
            raise ValueError(
                f"Unable to get correct index from sampler {sampler}, "
                f"it should yield an `int` or `list` of `int` of length {minibatch_size}."
            )

    def collate_fn_wrapper(batch):
        batch_list = [
            batch[i * (num_shots + 1) : (i + 1) * (num_shots + 1)]
            for i in range(batch_size)
            if i * (num_shots + 1) < len(batch)
        ]
        if collate_fn:
            return collate_fn(batch_list)
        return batch_list

    def check_consistent(name, obj, default_value):
        old_value = obj
        if obj is None:
            obj = default_value
        if isinstance(obj, list):
            if len(obj) != len(datasets):
                raise ValueError(
                    f"{name} should be a list of the same length as datasets, got {old_value}."
                )
            return obj
        else:  # single object
            return [obj]

    if not isinstance(datasets, list):
        datasets = [datasets]
    if num_shots is None:
        if num_per_dataset is not None:
            num_shots = (
                num_per_dataset - 1
                if isinstance(num_per_dataset, int)
                else sum(num_per_dataset) - 1
            )
        else:
            raise ValueError(
                "num_shot and num_per_dataset can't be None at the same time."
            )

    num_per_dataset = check_consistent(
        "num_per_dataset", num_per_dataset, [num_shots + 1]
    )
    samplers = check_consistent(
        "samplers", samplers, [SequentialSampler(dataset) for dataset in datasets]
    )

    if sum(num_per_dataset) != num_shots + 1:
        raise ValueError("The sum of num_per_dataset should be equal to num_shots + 1.")

    samplers = [
        batchilize_sampler(dataset, sampler, minibatch_size)
        for dataset, sampler, minibatch_size in zip(datasets, samplers, num_per_dataset)
    ]
    concat_dataset = ConcatDataset(datasets)
    concat_sampler = ConcatSampler(samplers, concat_dataset.cumulative_sizes)

    return DataLoader(
        concat_dataset,
        collate_fn=collate_fn_wrapper,
        batch_sampler=MultiBatchSampler(concat_sampler, batch_size, drop_last),
        **kwargs,
    )
