# %%
import torch
import os
import sys
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "..")
from dev.shift_model import Stratety
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
    "batch_size": 16,
    "num_shots": 0,
    "dtype": torch.float16,
    "generate_args": setting.generate_args,
}

dataloader = prepare_dataloader(
    dataset,
    batch_size=hparams["batch_size"],
    num_shots=hparams["num_shots"],
    shuffle=True,
)

# %%
from transformers import BitsAndBytesConfig
from testbed.models import Idefics
from dev.shift_encoder import AttnFFNShift, ShiftConfig, ShiftConfig

sd = torch.load("../results/ckpt/attn-ffn-lora.pth")
sd = {
    k.removeprefix("icv_encoder."): v.squeeze()
    for k, v in sd.items()
    if k.startswith("icv_encoder.")
}
device = torch.device("cuda:0")
icv_encoder = AttnFFNShift(ShiftConfig(4096, 32, strategy=ShiftConfig.FFN_SHIFT)).to(
    device, dtype=hparams["dtype"]
)
icv_encoder.load_state_dict(sd)


# %%
lmm = Idefics(
    config.idefics_9b_path,
    torch_dtype=torch.float16,
).to(device)
lmm.eval()
# %%
from testbed.models.model_base import HookType


hooks = icv_encoder.register_hook_for(lmm)

# %%
from tqdm import tqdm
import evaluate

from testbed.data import prepare_vqa_input
from testbed.data.vqav2 import postprocess_generation

total_acc = evaluate.load("Kamichanw/vqa_accuracy")
result = []

# for simplicity, just run 10 batches
for batch in tqdm(dataloader, desc=f"Evaluating {lmm.model_name} ..."):
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
