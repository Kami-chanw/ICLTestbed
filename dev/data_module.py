import pytorch_lightning as pl
import datasets
from torch.utils.data import (
    DistributedSampler,
    BatchSampler,
    SequentialSampler,
    RandomSampler,
)
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from testbed.data import prepare_caption_input, prepare_dataloader, prepare_vqa_input
import config


class DataModule(pl.LightningDataModule):

    def __init__(self, cfg, lmm) -> None:
        super().__init__()
        self.cfg = cfg
        self.lmm = lmm

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            if self.cfg.data.name == "vqav2":
                self.dataset = datasets.load_dataset(
                    os.path.join(config.testbed_dir, "data", "vqav2"),
                    split="train",
                    data_dir=config.vqav2_dir,
                    images_dir=config.coco_dir,
                    trust_remote_code=True,
                )
            elif self.cfg.data.name == "ok_vqa":
                self.dataset = datasets.load_dataset(
                    os.path.join(config.testbed_dir, "data", "ok_vqa"),
                    split="train",
                    data_dir=config.ok_vqa_dir,
                    images_dir=config.coco_dir,
                    trust_remote_code=True,
                )
            elif self.cfg.data.name == "coco_cap":
                self.dataset = datasets.load_dataset(
                    os.path.join(config.testbed_dir, "data", "coco"),
                    split="train",
                    data_dir=config.karpathy_coco_caption_dir,
                    images_dir=config.coco_dir,
                    trust_remote_code=True,
                )
            self.query_set = self.dataset.shuffle().select(
                range(self.cfg.data.num_query_samples)
            )

    def collate_fn(self, batch):
        """
        Split batch into full context, in-context examples, query and answer, and process them into model inputs.
        """
        if self.cfg.data.name in ["vqav2", "ok_vqa"]:
            context, images = prepare_vqa_input(
                batch, instruction=self.cfg.data.vqa_instruction
            )
            # we use the first answer as grounding truth
            answers = [item[-1]["answers"][0]["answer"] for item in batch]

            # the last 3 items:
            # [
            #   { "role" : "image",
            #     "content" :  ... },
            #   { "role" : "question",
            #     "content" : ... },
            #   { "role" : "answer" }
            # ]
            ice_texts = self.lmm.apply_prompt_template([ctx[:-3] for ctx in context])
            query_texts = self.lmm.apply_prompt_template([ctx[-3:] for ctx in context])
        elif self.cfg.data.name == "caption":
            context, images = prepare_caption_input(
                batch, instruction=self.cfg.data.caption_instruction
            )
            answers = [item[-1]["sentences_raw"][0] for item in batch]
            # the last 2 items:
            # [
            #   { "role" : "image"
            #     "content" :  ... },
            #   { "role" : "caption" }
            # ]
            ice_texts = self.lmm.apply_prompt_template([ctx[:-2] for ctx in context])
            query_texts = self.lmm.apply_prompt_template([ctx[-2:] for ctx in context])

        return {
            "ice_texts": ice_texts,
            "query_texts": query_texts,
            "answers": answers,
            "images": images,
        }

    def train_dataloader(self):
        if self.trainer.world_size > 1:
            samplers = [
                BatchSampler(
                    (
                        DistributedSampler(self.dataset, shuffle=True)
                        if self.cfg.data.name == "vqav2"
                        else RandomSampler(
                            self.dataset,
                            replacement=True,
                            num_samples=self.cfg.data.num_shot
                            * self.cfg.data.num_query_samples,
                        )
                    ),
                    batch_size=self.cfg.data.num_shot,
                    drop_last=True,
                ),
                DistributedSampler(self.query_set, shuffle=False),
            ]
        else:
            samplers = [
                BatchSampler(
                    (
                        RandomSampler(self.dataset)
                        if self.cfg.data.name == "vqav2"
                        else RandomSampler(
                            self.dataset,
                            replacement=True,
                            num_samples=self.cfg.data.num_shot
                            * self.cfg.data.num_query_samples,
                        )
                    ),
                    batch_size=self.cfg.data.num_shot,
                    drop_last=True,
                ),
                SequentialSampler(self.query_set),
            ]

        return prepare_dataloader(
            [self.dataset, self.query_set],
            self.cfg.data.batch_size,
            num_per_dataset=[self.cfg.data.num_shot, 1],
            collate_fn=self.collate_fn,
            samplers=samplers,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            shuffle=True,
        )
