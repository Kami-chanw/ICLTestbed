import re
import nltk
from testbed.data.common import register_postprocess, register_dataset_retriever

register_dataset_retriever(
    __name__.split(".")[-1],
    lambda item, is_last: (
        [
            {
                "role": "",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": f'is an image with written "{item["text"]}" on it. Is it hateful?',
                    },
                ],
            },
            (
                {"role": "answer"}
                if is_last
                else {
                    "role": "answer",
                    "content": [
                        {"type": "text", "text": "yes" if item["label"] == 1 else "no"}
                    ],
                }
            ),
        ],
        item["img"],
    ),
)


def postprocess(pred, stop_words):
    hateful_keywords = ["yes", "y", "hateful", "hate"]
    non_hateful_keywords = ["no", "n", "non-hateful", "not hateful", "benign"]
    if stop_words is not None:
        pred = re.split("|".join(stop_words), pred, 1)[0]

    pred = pred.strip().lower()
    tokens = nltk.word_tokenize(pred)

    for token in tokens:
        if token in hateful_keywords:
            return 1
        elif token in non_hateful_keywords:
            return 0

    return 0


register_postprocess(__name__.split(".")[-1], postprocess)
