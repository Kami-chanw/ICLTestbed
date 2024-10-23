from typing import Any, Callable, Dict, List, Optional, Union
from torch.utils.data import (
    Sampler,
    DataLoader,
    BatchSampler,
    SequentialSampler,
    RandomSampler,
    Dataset,
    ConcatDataset,
)
from testbed.data.sampler import MultiBatchSampler, ConcatSampler


def prepare_input(
    batch: List[List[Dict[str, Any]]],
    retriever: Callable[[Any, bool], List[Dict]],
    instruction: Optional[str] = None,
):
    """
    Prepares a batch of input data for various tasks such as visual question answering (VQA)
    or image captioning by formatting the context and optionally adding an instruction.

    Args:
        batch (List[List[Dict[str, Any]]]):
            A batch of data where each element is a list of dictionaries,
            representing a context (for example, previous questions/answers or captions).
        retriever (Callable[[Any, bool], List[Dict]]):
            A function that takes an item and a boolean flag `is_last` (indicating if the item is the last in the context)
            and returns a list of dictionaries representing the formatted context.
        instruction (Optional[str], optional):
            An optional instruction to prepend to the context. Defaults to None.

    Returns:
        List[List[Dict[str, Any]]]: A list of formatted contexts, where each context includes the messages retrieved
        by the retriever function, optionally preceded by the instruction.
    """
    batch_context = []
    for context in batch:
        messages = []
        if instruction is not None:
            messages.append({"role": "instruction", "content": instruction})

        for item in context:
            messages.extend(retriever(item, item == context[-1]))

        batch_context.append(messages)

    return batch_context


def prepare_vqa_input(
    batch: List[List[Dict[str, Any]]],
    instruction: Optional[str] = None,
    image_field: str = "image",
    question_field: str = "question",
    answer_field: str = "answer",
):
    """
    Prepares inputs for a Visual Question Answering (VQA) task by splitting a batch of question-answer pairs into
    images, texts, and contexts for processing. Note that this method is especially designed for default prompt template.
    If you customized prompt template, you should write your own prepare_*_input method.

    This function takes a batch of data containing question-answer pairs and splits it into separate components:
    - A list of images associated with each question.
    - A list of textual contexts formatted for further processing, such as feeding into a model.

    Args:
        batch (`List[List[Dict[str, Any]]]`):
            A batch of question-answer pairs sampled from a dataloader. The expected shape of the batch is
            `[batch_size, num_shots + 1]`, where `num_shots` refers to the number of context pairs preceding the
            target question.
        instruction (`str`, *optional*):
            An optional instruction that is prepended to each conversation.
        image_field (`str`, defaults to "image"):
            The field used to access the image in each question-answer pair dictionary.
        question_field (`str`, defaults to "question"):
            The field used to access the question in each question-answer pair dictionary.
        answer_field (`str`, defaults to "answer"):
            The field used to access the answer in each question-answer pair dictionary.

    Returns:
        - A list of lists containing contexts formatted as dictionaries. These contexts include the instruction
            (if provided), questions, and answers, ready to be fed into a model for processing. The shape of this
            list is `[batch_size, num_shots + 1]`.
        - A list of lists containing images associated with each question-answer pair. The shape of this list
            is `[batch_size, num_shots + 1]`.
    """

    def retriever(item: Dict[str, Any], is_last: bool):
        return [
            {"role": "image", "content": [{"type": "image"}]},
            {
                "role": "question",
                "content": [{"type": "text", "text": item[question_field]}],
            },
            (
                {"role": "answer"}
                if is_last
                else {
                    "role": "answer",
                    "content": [{"type": "text", "text": item[answer_field]}],
                }
            ),
        ]

    return prepare_input(batch, retriever, instruction), [
        [qa[image_field] for qa in context] for context in batch
    ]


def prepare_caption_input(
    batch: List[List[Dict[str, Any]]],
    instruction: Optional[str] = None,
    image_field: str = "image",
    caption_field: str = "caption",
):
    """
    Prepares the input data for a image captioning task by extracting images and formatting
    contextual information with optional instructions.Note that this method is especially designed for default prompt template.
    If you customized prompt template, you should write your own prepare_*_input method.

    Args:
        batch (`List[List[Dict[str, Any]]]`):
            A batch of data where each element is a list
            of dictionaries containing image and caption data. Each dictionary should
            have at least the keys specified by `image_field` and `caption_field`.
        instruction (`str`, *optional*): An optional instruction string to include in the
            context messages. Default is None.
        image_field (`str`, defaults to "image"):
            The field used to extract images from each dictionary in the batch.
        caption_field (`str`, defaults to "caption"):
            The field used to extract captions from each dictionary in the batch.

    Returns:
        - A list of lists, where each inner list contains a sequence of dictionaries representing
            the contextual information formatted with roles and content, including images and captions. The shape of this list
            is `[batch_size, num_shots + 1]`.
        - A list of lists, where each inner list contains the images for a batch. The shape of this list
            is `[batch_size, num_shots + 1]`.
    """

    def retriever(item: Dict[str, Any], is_last: bool):
        return [
            {"role": "image", "content": [{"type": "image"}]},
            {"role": "caption"},
            (
                {"role": "caption"}
                if is_last
                else {
                    "role": "caption",
                    "content": [{"type": "text", "text": item[caption_field]}],
                }
            ),
        ]

    return prepare_input(batch, retriever, instruction), [
        [item[image_field] for item in context] for context in batch
    ]


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
