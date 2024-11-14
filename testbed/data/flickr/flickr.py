import json
from pathlib import Path

import datasets
from testbed.data.common import split_generators

_DESCRIPTION = """
Flickr30k and Flickr8k are popular image captioning datasets, containing images with multiple associated captions. 
Flickr30k contains 30,000 images, while Flickr8k contains 8,000 images, and both datasets are used for training and evaluating image captioning models.
"""

_LICENSE = "CC BY 4.0"

# Flickr30k image URL (requires manual download)
_FLICKR30K_IMAGES_URL = (
    "https://shannon.cs.illinois.edu/DenotationGraph/data/index.html"  # Manual download
)

# Flickr8k image URL (requires manual download from Kaggle)
_FLICKR8K_IMAGES_URL = (
    "https://www.kaggle.com/datasets/adityajn105/flickr8k"  # Manual download
)

# Define the Features based on the provided item structure
_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "filepath": datasets.Value("string"),
        "sentids": [datasets.Value("int32")],
        "filename": datasets.Value("string"),
        "imgid": datasets.Value("int32"),
        "split": datasets.Value("string"),
        "caption": datasets.Value("string"),
        "sentences_tokens": [[datasets.Value("string")]],
        "sentences_raw": [datasets.Value("string")],
        "sentences_sentid": [datasets.Value("int32")],
        "cocoid": datasets.Value("int32"),
    }
)

_SUB_FOLDER_OR_FILE_NAME = {
    "flickr30k": {
        "annotations": {
            "train": "dataset_flickr30k.json",
            "val": "dataset_flickr30k.json",
            "test": "dataset_flickr30k.json",
        },
        "images": {
            "train": "flickr30k-images",
            "val": "flickr30k-images",
            "test": "flickr30k-images",
        },
    },
    "flickr8k": {
        "annotations": {
            "train": "dataset_flickr8k.json",
            "val": "dataset_flickr8k.json",
            "test": "dataset_flickr8k.json",
        },
        "images": {
            "train": "flickr8k-images",
            "val": "flickr8k-images",
            "test": "flickr8k-images",
        },
    },
}


class FlickrConfig(datasets.BuilderConfig):
    def __init__(self, images_dir=None, caption_selector=None, verbose=True, **kwargs):
        self.images_dir = images_dir if images_dir is not None else self.data_dir
        self.verbose = verbose
        self.caption_selector = (
            caption_selector
            if caption_selector is not None
            else lambda dct: dct[0]["raw"]  # select the 1st sentence as ground truth
        )
        super().__init__(**kwargs)


class Flickr(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = FlickrConfig
    DEFAULT_CONFIG_NAME = "flickr30k"
    BUILDER_CONFIGS = [
        FlickrConfig(
            name="flickr30k",
            description="Flickr30k dataset for image captioning.",
            version=datasets.Version("1.0.0"),
        ),
        FlickrConfig(
            name="flickr8k",
            description="Flickr8k dataset for image captioning.",
            version=datasets.Version("1.0.0"),
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        if self.config.data_dir is None or self.config.images_dir is None:
            raise ValueError("Missing arguments for data_dir and images_dir.")

        return split_generators(
            lambda file_type, split_name: Path(
                self.config.data_dir
                if file_type != "images"
                else self.config.images_dir
            )
            / _SUB_FOLDER_OR_FILE_NAME[self.config.name][file_type][split_name],
            _SUB_FOLDER_OR_FILE_NAME[self.config.name],
            self.config.verbose,
        )

    def _generate_examples(self, split, annotations_path, images_path):
        with open(annotations_path, "r", encoding="utf-8") as fi:
            annotations = json.load(fi)
            for image_metadata in annotations["images"]:
                if image_metadata["split"] == split:
                    record = {
                        "image": str(
                            images_path.resolve() / image_metadata["filename"]
                        ),
                        "filepath": str(
                            images_path.resolve() / image_metadata["filename"]
                        ),
                        "sentids": image_metadata["sentids"],
                        "filename": image_metadata["filename"],
                        "imgid": image_metadata["imgid"],
                        "split": image_metadata["split"],
                        "caption": self.config.caption_selector(
                            image_metadata["sentences"]
                        ),
                        "sentences_tokens": [
                            caption["tokens"] for caption in image_metadata["sentences"]
                        ],
                        "sentences_raw": [
                            caption["raw"] for caption in image_metadata["sentences"]
                        ],
                        "sentences_sentid": [
                            caption["sentid"] for caption in image_metadata["sentences"]
                        ],
                    }
                    yield record["imgid"], record
