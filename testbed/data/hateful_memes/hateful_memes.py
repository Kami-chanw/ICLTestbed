import json
import os
from pathlib import Path

from testbed.data.common import split_generators
import datasets


_CITATION = """@inproceedings{kiela2020hateful,
  title={The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes},
  author={Kiela, Douwe and Firooz, Hamed and Mohan, Ashwin Paranjape and Goswami, Vedanuj and Singh, Amanpreet and Ringshia, Piyush and Testuggine, Davide},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
"""

_DESCRIPTION = """\
The Hateful Memes dataset consists of 10,000+ new multimodal examples created by Facebook AI.
These memes have been carefully designed to fool unimodal classifiers, and contain both benign confounders and hateful speech.
The dataset includes both the images and OCR-extracted text.
"""

_HOMEPAGE = "https://ai.facebook.com/datasets/hateful-memes/"

_LICENSE = "CC BY-NC 4.0"

_URLS = ["https://huggingface.co/datasets/limjiayi/hateful_memes_expanded"]

_FEATURES = datasets.Features(
    {
        "id": datasets.Value("string"),
        "img": datasets.Image(),
        "label": datasets.Value("int32"),
        "text": datasets.Value("string"),
    }
)

_SUB_FOLDER_OR_FILE_NAME = {
    "annotations": {
        "train": "train.jsonl",
        "val": ["dev_seen.jsonl", "dev_unseen.jsonl"],
        "test": ["test_seen.jsonl", "test_unseen.jsonl"],
    },
    "images": {
        "train": "img",
        "val": "img",
        "test": "img",
    },
}


class HatefulMemesConfig(datasets.BuilderConfig):

    def __init__(self, verbose=True, **kwargs):
        self.verbose = verbose

        super().__init__(**kwargs)


class HatefulMemes(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIG_CLASS = HatefulMemesConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        def expand_path(file_type, split_name):
            basename = _SUB_FOLDER_OR_FILE_NAME[file_type][split_name]
            if isinstance(basename, list):
                return [Path(self.config.data_dir) / name for name in basename]
            return Path(self.config.data_dir) / basename

        return split_generators(
            expand_path, _SUB_FOLDER_OR_FILE_NAME, self.config.verbose
        )

    def _generate_examples(self, split, annotations_path, images_path):
        if not isinstance(annotations_path, list):
            annotations_path = [annotations_path]

        memo = set()
        for ann in annotations_path:
            with open(ann, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    if data["id"] in memo:
                        continue
                    memo.add(data["id"])
                    record = {
                        "id": str(data["id"]),
                        "img": str(images_path / os.path.basename(data["img"])),
                        "label": int(data["label"]),
                        "text": data["text"],
                    }

                    yield data["id"], record
