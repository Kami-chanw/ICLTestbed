import json
from pathlib import Path
import datasets
import os
from urllib import request
from testbed.data.common import split_generators, most_common_from_dict

_CITATION = """\
@InProceedings{mishraICDAR19,
  author    = "Anand Mishra and Shashank Shekhar and Ajeet Kumar Singh and Anirban Chakraborty",
  title     = "OCR-VQA: Visual Question Answering by Reading Text in Images",
  booktitle = "ICDAR",
  year      = "2019",
}
"""


_DESCRIPTION = """\
 OCR-VQA dataset comprises of 207,572 images of book covers and contains more than 1 million question-answer pairs about these images. 
"""


_HOMEPAGE = "https://ocr-vqa.github.io/"


_LICENSE = "Apache License 2.0"

# OCR-VQA dataset URL (requires manual download)
_URLS = "https://ocr-vqa.github.io/"


_SUB_FOLDER_OR_FILE_NAME = {
    "annotations": {
        "train": "dataset.json",
        "val": "dataset.json",
        "test": "dataset.json",
    },
}


class OCRVQAConfig(datasets.BuilderConfig):

    def __init__(self, images_dir=None, verbose=True, **kwargs):
        self.images_dir = images_dir
        self.verbose = verbose

        super().__init__(**kwargs)


class OCRVQA(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = OCRVQAConfig

    def _info(self):
        features = datasets.Features(
            {
                "image": datasets.Image(),
                "question": datasets.Value("string"),
                "question_id": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "image_id": datasets.Value("string"),
                "genre": datasets.Value("string"),
                "authorName": datasets.Value("string"),
                "title": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.data_dir is None or self.config.images_dir is None:
            raise ValueError("Missing arguments for data_dir and images_dir.")

        return split_generators(
            lambda file_type, split_name: Path(self.config.data_dir)
            / _SUB_FOLDER_OR_FILE_NAME[file_type][split_name],
            _SUB_FOLDER_OR_FILE_NAME,
            self.config.verbose,
        )

    def _generate_examples(self, split, annotations_path):
        dataset = json.load(open(annotations_path, "r"))
        split_mapping = {
            "train": 1,
            "val": 2,
            "test": 3,
        }
        for image_id, info in dataset.items():
            if split_mapping[split] != info["split"]:
                continue
            if self.config.images_dir is None:
                image_path = info["imageURL"]
            else:
                image_path = os.path.join(
                    self.config.images_dir,
                    f"{image_id}{os.path.splitext(info["imageURL"])[1]}",
                )
                if not os.path.exists(image_path):
                    try:
                        request.urlretrieve(info["imageURL"], image_path)
                    except Exception as e:
                        print(f"Failed to download from {info["imageURL"]}, because {e}")
                        continue
            for idx, (question, answer) in enumerate(
                zip(info["questions"], info["answers"])
            ):

                record = {
                    "image": image_path,
                    "image_id": image_id,
                    "question_id": f"{image_id}{idx:02d}",
                    "question": question,
                    "answer": answer,
                    "genre": info["genre"],
                    "title": info["title"],
                    "authorName": info["authorName"],
                }

                yield record["question_id"], record
