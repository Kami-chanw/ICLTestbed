import torch

if torch.cuda.is_available():
    import os

    os.environ["CUDA_VISIBLE_DEVICE"] = ",".join(
        [str(i) for i in range(torch.cuda.device_count())]
    )

# path
coco_dir = "/data/share/pyz/data/mscoco/mscoco2014"
vqav2_dir = "/data1/pyz/dataset/vqav2"
ok_vqa_dir = "/data1/pyz/dataset/okvqa"

# model weight
idefics_9b_path = "/data1/pyz/model_weight/idefics-9b"


idefics2_8b_path = (
    "/data1/pyz/model_weight/idefics2-8b"  # you'd better not use idefics2-8b to run icl
)
idefics2_8b_base_path = "/data1/pyz/model_weight/idefics2-8b-base"
