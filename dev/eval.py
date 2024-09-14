# %%
import torch
import os
import sys
from datasets import load_dataset

sys.path.insert(0, "..")
from testbed.data import prepare_dataloader
import config
import exp_settings as setting

dataset = load_dataset(
    os.path.join(config.testbed_dir, "data", "vqav2"),
    split="validation",
    data_dir=".",
    images_dir=config.coco_dir,
    trust_remote_code=True,
)

hparams = {
    "batch_size": 8,
    "num_shots": 0,
    "dtype": torch.float16,
    "generate_args": setting.generate_args,
}

dataloader = prepare_dataloader(
    dataset,
    batch_size=hparams["batch_size"],
    num_shots=hparams["num_shots"],
)

# %%
from transformers import BitsAndBytesConfig
from testbed.models import Idefics
from dev.icv_encoder import GlobalICVEncoder, AttnAwareEncoder

device = torch.device("cuda:1")
lmm = Idefics(
    config.idefics_9b_path,
    torch_dtype=torch.float16,
).to(device)
lmm.eval()
icv_encoder = AttnAwareEncoder(4096, 32).to(device)

# %%
sd = torch.load("../results/ckpt/icv_cpk-attn-aware.pth")
sd = {
    k.removeprefix("icv_encoder."): v.squeeze()
    for k, v in sd.items()
    if k.startswith("icv_encoder.")
}
icv_encoder.load_state_dict(sd, strict=False)

# %%
from testbed.models.model_base import HookType


hooks = lmm.register_forward_pre_hook(
    HookType.TEXT_MODEL_LAYER,
    icv_encoder.pre_hook,
)

# %%
from tqdm import tqdm
import evaluate

from testbed.data import prepare_vqa_input
from testbed.data.vqav2 import postprocess_generation

total_acc = evaluate.load("Kamichanw/vqa_accuracy")
result = []

# for simplicity, just run 10 batches
for _, batch in zip(range(100), tqdm(dataloader, desc=f"Evaluating {lmm.model_name} ...")):
    text, images = prepare_vqa_input(
        batch, instruction="Provide an answer to the question. Use the image to answer."
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

# %%
hparams["dtype"] = str(hparams["dtype"])
evaluate.save("./", eval_result=eval_result, hparams=hparams, records=result)
