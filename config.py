import torch

if torch.cuda.is_available():
    import os

    os.environ["CUDA_VISIBLE_DEVICE"] = ",".join(
        [str(i) for i in range(torch.cuda.device_count())]
    )

# path
coco_dir = "/home/share/pyz/dataset/mscoco/mscoco2014"
vqav2_dir = "/home/share/pyz/dataset/vqav2"
ok_vqa_dir = "/home/share/pyz/dataset/okvqa"
karpathy_coco_caption_dir = "/home/share/karpathy-split"

# model weight
idefics_9b_path = "/home/share/pyz/model_weight/idefics-9b"


idefics2_8b_path = "/home/share/pyz/model_weight/idefics2-8b"  # you'd better not use idefics2-8b to run icl
idefics2_8b_base_path = "/home/share/pyz/model_weight/idefics2-8b-base"
