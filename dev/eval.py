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
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler

hparams = {
    "batch_size": 8,
    "num_shots": 0,
    "dtype": torch.bfloat16,
    "generate_args": {"num_beams": 3, "max_new_tokens": 5},
}
# ice_set = load_dataset(
#     os.path.join(config.testbed_dir, "data", "vqav2"),
#     split="train",
#     data_dir=config.vqav2_dir,
#     images_dir=config.coco_dir,
#     trust_remote_code=True,
# )
query_set = load_dataset(
    os.path.join(config.testbed_dir, "data", "vqav2"),
    split="validation",
    data_dir=".",
    images_dir=config.coco_dir,
    trust_remote_code=True,
)

dataloader = prepare_dataloader(
    query_set,
    batch_size=hparams["batch_size"],
    num_shots=hparams["num_shots"],
)

# %%
from transformers import BitsAndBytesConfig
from testbed.models import Idefics
from shift_encoder import AttnFFNShift, ShiftStrategy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--runname', type=str, required=True, help='Name of the run')
args = parser.parse_args()

sd = torch.load(f"../results/ckpt/{args.runname}.pth")
sd = {
    k.removeprefix("shift_encoder."): v.squeeze()
    for k, v in sd.items()
    if k.startswith("shift_encoder.")
}
device = torch.device("cuda:2")
icv_encoder = AttnFFNShift(4096, 32, attn_strategy=ShiftStrategy.USE_VECTOR_IMPL).to(
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


hooks = icv_encoder.register_shift_hooks(lmm)

# %%
from tqdm import tqdm
import evaluate

from testbed.data import prepare_vqa_input
from testbed.data.vqav2 import postprocess_generation

total_acc = evaluate.load("Kamichanw/vqa_accuracy")
result = []
# lmm.prompt_template = """{% if messages[0]['role'] == 'instruction' -%}
# Instruction: {{ messages[0]['content'] }}
# {%- set messages = messages[1:] %}
# {%- endif %}
# {%- for message in messages %}
# {%- if 'content' in message and message['content'] and message['content'][0]['type'] == 'image' %}
# \n<image>
# {%- endif %} {{ message['role'].capitalize() }}: {%- if 'content' in message %}
# {%- for line in message['content'] %}
# {%- if line['type'] == 'text' %} {{ line['text'] }}{%- endif %}
# {%- endfor %}{%- endif %}
# {%- if not loop.last %}\n{%- endif %}
# {%- endfor %}"""
# import re

# for simplicity, just run 10 batches
for _, batch in zip(
    range(1000), tqdm(dataloader, desc=f"Evaluating {lmm.model_name} ...")
):
    text, images = prepare_vqa_input(
        batch, instruction="Provide an answer to the question. Use the image to answer."
    )
    predictions = lmm.generate(text, images, **hparams["generate_args"])
    for pred, context in zip(predictions, batch):
        last_qa = context[-1]
        gt_answer = [item["answer"] for item in last_qa["answers"]]
        prediction = postprocess_generation(pred)
        # prediction = re.split("Short|Question|Long", prediction)[0]
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
