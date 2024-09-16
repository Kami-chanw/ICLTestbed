# dataset config
num_query_samples = 300

# training config
alpha_lr = 1e-2
icv_lr = 1e-3
weight_decay = 1e-3 
warmup_step = 0.1
ce_loss_weight = 0.0
strategy = "deepspeed_stage_2_offload"  # "deepspeed_stage_2_offload" / "ddp"

# data module config
batch_size = 1
num_shot = 32
num_workers = 5

# generation config
generate_args = dict(num_beams=3, max_new_tokens=10, length_penalty=0.0)

task = "vqa"  # one of "vqa" and "caption"
vqa_instruction = "Provide an answer to the question. Use the image to answer."
caption_instruction = ""

# in current tasks (vqa and caption), query only has 1 images.
# this is useful in splitting images of in-context examples and query.
num_image_in_query = 1
