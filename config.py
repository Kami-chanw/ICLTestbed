import torch
from pathlib import Path
import subprocess

network_node_name = subprocess.run(
    ["uname", "-n"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
).stdout
# path
testbed_dir = str(Path(__file__).parent / "testbed")
result_dir = "/home/jyc/ICLTestbed/results"

if "ii" in network_node_name:
    # 8x3090
    coco_dir = "/data/pyz/data/mscoco"
    vqav2_dir = "/data/pyz/data/vqav2"
    ok_vqa_dir = "/data/share/pyz/okvqa"
    karpathy_coco_caption_dir = "/data/share/karpathy-split"

    idefics_9b_path = "/data1/share/idefics-9b"

    idefics2_8b_path = (
        "/data1/share/idefics2-8b"  # you'd better not use idefics2-8b to run icl
    )
    idefics2_8b_base_path = "/data1/share/idefics2-8b-base"

elif "a6000" in network_node_name:
    # 4xa6000
    coco_dir = "/home/share/pyz/dataset/mscoco/mscoco2014"
    vqav2_dir = "/home/share/pyz/dataset/vqav2"
    ok_vqa_dir = "/home/share/pyz/dataset/okvqa"
    karpathy_coco_caption_dir = "/home/share/karpathy-split"
    idefics_9b_path = "/home/share/pyz/model_weight/idefics-9b"

    idefics2_8b_path = "/home/share/pyz/model_weight/idefics2-8b"  # you'd better not use idefics2-8b to run icl
    idefics2_8b_base_path = "/home/share/pyz/model_weight/idefics2-8b-base"
elif "ubuntu" in network_node_name:
    # 4x3090
    coco_dir = "/data/share/pyz/data/mscoco/mscoco2014"
    vqav2_dir = "/data/share/pyz/vqav2"
    ok_vqa_dir = "/data/share/pyz/okvqa"
    karpathy_coco_caption_dir = "/data/share/karpathy-split"
    idefics_9b_path = "/data1/pyz/model_weight/idefics-9b"

    idefics2_8b_path = "/data1/pyz/model_weight/idefics2-8b"  # you'd better not use idefics2-8b to run icl
    idefics2_8b_base_path = "/data1/pyz/model_weight/idefics2-8b-base"
else:
    raise RuntimeError("Unknow host")
