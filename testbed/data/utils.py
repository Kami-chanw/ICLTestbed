from typing import Any, Callable, Dict, List, Optional
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler


def prepare_vqa_input(
    batch: List[List[Dict[str, Any]]],
    instruction: Optional[str] = None,
    image_key: str = "image",
    question_key: str = "question",
    answer_key: str = "answer",
):
    """
    Prepares inputs for a Visual Question Answering (VQA) task by splitting a batch of question-answer pairs into
    images, texts, and contexts for processing.

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
        image_key (`str`, defaults to "image"):
            The key used to access the image in each question-answer pair dictionary.
        question_key (`str`, defaults to "question"):
            The key used to access the question in each question-answer pair dictionary.
        answer_key (`str`, defaults to "answer"):
            The key used to access the answer in each question-answer pair dictionary.

    Returns:
        `Tuple[List[List[Image]], List[List[Dict[str, Any]]]]`:
            - A list of lists containing images associated with each question-answer pair. The shape of this list
              is `[batch_size, num_shots + 1]`.
            - A list of lists containing contexts formatted as dictionaries. These contexts include the instruction
              (if provided), questions, and answers, ready to be fed into a model for processing. The shape of this
              list is `[batch_size, num_shots + 1]`.
    """

    batch_images, batch_context = [], []
    for context in batch:
        batch_images.append([qa[image_key] for qa in context])
        messages = []
        if instruction is not None:
            messages.append({"role": "instruction", "content": instruction})

        for qa in context[:-1]:
            messages.extend(
                [
                    {
                        "role": "question",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": qa[question_key]},
                        ],
                    },
                    {
                        "role": "short answer",
                        "content": [
                            {
                                "type": "text",
                                "text": qa[answer_key],
                            },
                        ],
                    },
                ]
            )
        messages.extend(
            [
                {
                    "role": "question",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": context[-1][question_key]},
                    ],
                },
                {"role": "short answer"},
            ]
        )
        batch_context.append(messages)

    return batch_images, batch_context


def prepare_caption_input(
    batch: List[List[Dict[str, Any]]], instruction: Optional[str] = None,
    image_key:str="image", caption_key:str="caption"
):
    """
    Split inputs from a batch of question-answer pairs for VQA task to images, texts and answers of last question.

    Args:
        batch (`List[List[Dict[str, Any]]]`):
            The batch sampled from dataloader. The shape of it should be [batch_size, num_shots + 1].
        instruction (`str`, *optional*):
            Instruction used to prepend to each conversation.

    Returns:
        [`List[List[PIL.Images]]`]:
            Images corresponding to each question-answer pair with shape [batch_size, num_shots + 1].
        [`List[List[Dict[str, Any]]]`]:
            Context text with shape [batch_size, num_shots + 1], which can be fed into `testbed.models` directly.
    """

    batch_images, batch_context = [], []
    for context in batch:
        batch_images.append([qa["image"] for qa in context])
        messages = []
        if instruction is not None:
            messages.append({"role": "instruction", "content": instruction})

        for qa in context[:-1]:
            messages.extend(
                [
                    {
                        "role": "question",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": qa["question"]},
                        ],
                    },
                    {
                        "role": "short answer",
                        "content": [
                            {
                                "type": "text",
                                "text": qa["answer"],
                            },
                        ],
                    },
                ]
            )
        messages.extend(
            [
                {
                    "role": "question",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": context[-1]["question"]},
                    ],
                },
                {"role": "short answer"},
            ]
        )
        batch_context.append(messages)

    return batch_images, batch_context


def prepare_dataloader(dataset, batch_size, num_shots, sampler=None):
    if sampler is None:
        sampler = SequentialSampler(dataset)

    def collate_fn(batch):
        return [
            batch[i * (num_shots + 1) : (i + 1) * (num_shots + 1)]
            for i in range(batch_size)
        ]

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_sampler=BatchSampler(
            sampler, batch_size * (num_shots + 1), drop_last=True
        ),
    )
