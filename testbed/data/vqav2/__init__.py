from testbed.data.common import register_dataset_retriever

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

# no need to register post process for vqav2, just use default post process
