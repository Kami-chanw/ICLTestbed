from testbed.data.common import register_dataset_retriever, register_postprocess

register_dataset_retriever(
    __name__.split(".")[-1],
    lambda item, is_last: (
        [
            {"role": "image", "content": [{"type": "image"}]},
            {
                "role": "question",
                "content": [{"type": "text", "text": item["question"]}],
            },
            (
                {"role": "answer"}
                if is_last
                else {
                    "role": "answer",
                    "content": [{"type": "text", "text": item["answer"]}],
                }
            ),
        ],
        item["image"],
    ),
)

# mroe post process will be done in evaluate procedure
register_postprocess(__name__.split(".")[-1], lambda pred: pred)
