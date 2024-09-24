# dataset config
num_query_samples = 500

# training config
alpha_lr = 5e-3
icv_lr = 5e-4
weight_decay = 1e-4
warmup_step = 0.1
ce_loss_weight = 0.5
strategy = "deepspeed_stage_2_offload"  # "deepspeed_stage_2_offload" / "ddp"

# data module config
batch_size = 2
accumulate_grad_batches = 2
num_shot = 32
num_workers = 5
grad_clip_val = 1.0

# generation config
generate_args = dict(num_beams=3, max_new_tokens=10, length_penalty=0.0)

dataset = "vqav2"  # [vqav2, ok_vqa, coco_cap]
vqa_instruction = "Provide an answer to the question. Use the image to answer."
caption_instruction = ""

# in current tasks (vqa and caption), query only has 1 images.
# this is useful in splitting images of in-context examples and query.
num_image_in_query = 1
