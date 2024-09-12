import pytorch_lightning as pl
import datasets
from torch.utils.data import DistributedSampler
import os
import sys

sys.path.insert(0, "..")
from testbed.data import prepare_caption_input, prepare_dataloader, prepare_vqa_input
import config
import exp_settings as setting


class ICVDataModule(pl.LightningDataModule):

    def __init__(self, lmm) -> None:
        super().__init__()
        self.lmm = lmm

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            if setting.task == "vqa":
                self.dataset = datasets.load_dataset(
                    os.path.join(config.testbed_dir, "data", "vqav2"),
                    split="train",
                    data_dir=config.vqav2_dir,
                    images_dir=config.coco_dir,
                    trust_remote_code=True,
                )
            elif setting.task == "caption":
                self.dataset = datasets.load_dataset(
                    os.path.join(config.testbed_dir, "data", "coco"),
                    data_dir=config.karpathy_coco_caption_dir,
                    images_dir=config.coco_dir,
                    trust_remote_code=True,
                )
            self.dataset = self.dataset.shuffle().select(range(setting.num_train_samples))

    def collate_fn(self, batch):
        """
        Split batch into full context, in-context examples, query and answer, and process them into model inputs.
        """
        if setting.task == "vqa":
            context, images = prepare_vqa_input(
                batch, instruction=setting.vqa_instruction
            )
            # we use the first answer as grounding truth
            answers = [item[-1]["answers"][0]["answer"] for item in batch]
        elif setting.task == "caption":
            context, images = prepare_caption_input(
                batch, instruction=setting.caption_instruction
            )
            answers = [item[-1]["sentences_raw"][0] for item in batch]

        # the last two items (take vqa as an example):
        # [
        #   { "role" : "question"
        #     "content" :  ... },
        #   { "role" : "short answer" }
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
            sampler = DistributedSampler(self.dataset)
        else:
            sampler = None
        return prepare_dataloader(
            self.dataset,
            setting.batch_size,
            setting.num_shot,
            collate_fn=self.collate_fn,
            samplers=sampler,
            num_workers=setting.num_workers,
            pin_memory=True,
            shuffle=True,
        )
