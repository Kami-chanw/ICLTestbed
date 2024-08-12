import evaluate
import pytest
import os
import sys

sys.path.insert(0, os.getcwd())

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def standard_evaluate(annotations_path, result_path):
    coco = COCO(annotations_path)
    coco_result = coco.loadRes(result_path)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params["image_id"] = coco_result.getImgIds()
    coco_eval.evaluate()
    return coco_eval.eval["CIDEr"]


def custom_evaluate(annotations_path, result_path):
    # same as standard evaluation procedure
    coco = COCO(annotations_path)
    coco_result = coco.loadRes(result_path)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params["image_id"] = coco_result.getImgIds()
    gts = {}
    res = {}
    for imgId in coco_eval.params["image_id"]:
        gts[imgId] = coco_eval.coco.imgToAnns[imgId]
        res[imgId] = coco_eval.cocoRes.imgToAnns[imgId]

    # custom evaluation procedure
    CIDEr = evaluate.load("testbed/evaluate/metrics/CIDEr")
    references = [[y['caption'] for y in x]for x in gts.values()]
    predicitons = [x[0]['caption'] for x in res.values()]

    return CIDEr.compute(predictions=predicitons, references=references)["CIDEr"]


@pytest.mark.parametrize(
    "annotations_path,result_path",
    [
        (
            os.path.join("tests", "CIDEr", "captions_val2014.json"),
            os.path.join("tests", "CIDEr", "captions_val2014_fakecap_results.json"),
        ),
        # you can add more test case to validate correctness of `CIDEr`
    ],
)
def test_CIDEr(annotations_path, result_path):
    standard = standard_evaluate(annotations_path, result_path)
    testbed = custom_evaluate(annotations_path, result_path)

    assert standard == testbed
