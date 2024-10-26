from testbed.data.common import register_dataset_retriever

register_dataset_retriever(
    __name__.split(".")[-1],
    lambda item, is_last: (
        [
            {"role": "image", "content": [{"type": "image"}]},
            (
                {"role": "caption"}
                if is_last
                else {
                    "role": "caption",
                    "content": [{"type": "text", "text": item["caption"]}],
                }
            ),
        ],
        item["image"],
    ),
)
