from typing import Any, Callable, Dict, List, Optional
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler


def prepare_vqa_input(
    batch: List[List[Dict[str, Any]]],
    instruction: Optional[str] = None,
    answer_selector: Optional[Callable[[Dict[str, Any]], str]] = None,
):
    """
    Split inputs from a batch of question-answer pairs for VQA task to images, texts and answers of last question.

    Args:
        batch (`List[List[Dict[str, Any]]]`):
            The batch sampled from dataloader. The shape of it should be [batch_size, num_shots + 1].
        instruction (`str`, *optional*):
            Instruction used to prepend to each conversation.
        answer_selector (`Callable[[Dict[str, Any]], str]`, *optional*):
            A method used to select answer presented in in-context examples. It takes answers dict as input, \
            and returns answer that selected as ground truth. If set to `None`, a majority policy will be applied.
    
    Returns:
        [`List[List[PIL.Images]]`]:
            Images corresponding to each question-answer pair with shape [batch_size, num_shots + 1].
        [`List[List[Dict[str, Any]]]`]:
            Context text with shape [batch_size, num_shots + 1], which can be fed into `testbed.models` directly.
    """

    def most_common_from_dict(dct):
        lst = [x["answer"] for x in dct]
        return max(set(lst), key=lst.count)

    answer_selector = answer_selector if answer_selector is not None else most_common_from_dict

    batch_images, batch_context = [], []
    for context in batch:
        batch_images.append([qa["image"] for qa in context])
        messages = []
        if instruction is not None:
            messages.append({"role": "instruction", "content": instruction})

        for qa in context[:-1]:
            messages.append(
                {
                    "role": "example",
                    "query": [
                        {"type": "image"},
                        {"type": "text", "text": qa["question"]},
                    ],
                    "answer": [
                        {
                            "type": "text",
                            "text": answer_selector(qa["answers"]),
                        },
                    ],
                }
            )
        messages.append(
            {
                "role": "question",
                "query": [
                    {"type": "image"},
                    {"type": "text", "text": context[-1]["question"]},
                ],
            }
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
