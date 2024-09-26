import torch
import os
import sys
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "..")
from dev.shift_model import Stratety
from testbed.data import prepare_dataloader
import config
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler


from transformers import BitsAndBytesConfig
from testbed.models import Idefics
from shift_encoder import AttnFFNShift, ShiftStrategy
import hydra

from testbed.models.model_base import HookType
from tqdm import tqdm
import evaluate

from testbed.data import prepare_vqa_input
from testbed.data.vqav2 import postprocess_generation
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="exp_settings.yaml", version_base=None)
def main(cfg: DictConfig):
    hparams = {
        "batch_size": 16,
        "num_shots": 0,
        "dtype": torch.float16,
        "generate_args": dict(cfg.generation_args),
    }
    # ice_set = load_dataset(
    #     os.path.join(config.testbed_dir, "data", "vqav2"),
    #     split="train",
    #     data_dir=config.vqav2_dir,
    #     images_dir=config.coco_dir,
    #     trust_remote_code=True,
    # )
    query_set = load_dataset(
        os.path.join(config.testbed_dir, "data", cfg.data.name),
        split="validation",
        data_dir="." if cfg.data.name == "vqav2" else config.ok_vqa_dir,
        images_dir=config.coco_dir,
        trust_remote_code=True,
    )

    dataloader = prepare_dataloader(
        query_set,
        batch_size=hparams["batch_size"],
        num_shots=hparams["num_shots"],
    )
    device = torch.device("cuda")
    lmm = Idefics(
        config.idefics_9b_path,
        torch_dtype=torch.float16,
    ).to(device)
    lmm.eval()
    sd = torch.load(
        f"/home/jyc/ICLTestbed/results/ckpt/{cfg.runname}.pth", weights_only=True
    )
    sd = {
        k.removeprefix("shift_encoder."): v.squeeze()
        for k, v in sd.items()
        if k.startswith("shift_encoder.")
    }
    icv_encoder = AttnFFNShift(
        4096, 32, attn_strategy=ShiftStrategy.USE_VECTOR_IMPL
    ).to(device, dtype=hparams["dtype"])
    icv_encoder.load_state_dict(sd)

    hooks = icv_encoder.register_shift_hooks(lmm)

    total_acc = evaluate.load("Kamichanw/vqa_accuracy")
    result = []
    for _, batch in zip(
        range(10000), tqdm(dataloader, desc=f"Evaluating {lmm.model_name} ...")
    ):
        text, images = prepare_vqa_input(
            batch,
            instruction="Provide an answer to the question. Use the image to answer.",
        )
        predictions = lmm.generate(text, images, **hparams["generate_args"])
        for pred, context in zip(predictions, batch):
            last_qa = context[-1]
            gt_answer = [item["answer"] for item in last_qa["answers"]]
            prediction = postprocess_generation(pred)
            total_acc.add(
                prediction=prediction,
                reference=gt_answer,
                question_types=last_qa["question_type"],
                answer_types=last_qa["answer_type"],
            )
            result.append(
                {
                    "question_id": last_qa["question_id"],
                    "raw_output": pred,
                    "question": last_qa["question"],
                    "question_type": last_qa["question_type"],
                    "answer_type": last_qa["answer_type"],
                    "prediction": prediction,
                    "answers": last_qa["answers"],
                }
            )

    eval_result = total_acc.compute()
    print(eval_result)

    hparams["dtype"] = str(hparams["dtype"])
    evaluate.save(
        f"./{cfg.runname}.json",
        eval_result=eval_result,
        hparams=hparams,
        records=result,
    )


if __name__ == "__main__":
    main()
