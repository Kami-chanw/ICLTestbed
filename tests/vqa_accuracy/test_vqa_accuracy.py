import evaluate
import pytest
import os
import sys

sys.path.insert(0, "..")
import config

from testbed.data import vqav2
from tests.vqa_accuracy.vqa import VQA
from tests.vqa_accuracy.vqaEval import VQAEval

VQAv2_FILE_NAME = {
    "questions": {
        "train": "v2_OpenEnded_mscoco_train2014_questions.json",
        "val": "v2_mscoco_val2014_question_subdata.json",
        "test-dev": "v2_OpenEnded_mscoco_test-dev2015_questions.json",
        "test": "v2_OpenEnded_mscoco_test2015_questions.json",
    },
    "annotations": {
        "train": "v2_mscoco_train2014_annotations.json",
        "val": "v2_mscoco_val2014_annotations_subdata.json",
    },
}


def standard_evaluate(questions_path, annotations_path, result_path):
    vqa = VQA(annotations_path, questions_path)
    vqaRes = vqa.loadRes(result_path, questions_path)
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    return vqaEval.evaluate()


def custom_evaluate(questions_path, annotations_path, result_path):
    vqa = VQA(annotations_path, questions_path)
    vqaRes = vqa.loadRes(result_path, questions_path)
    vqaEval = VQAEval(vqa, vqaRes, n=2)

    vqa_acc = evaluate.load("testbed/evaluate/metrics/vqa_accuracy")

    accQA, accQuesType, accAnsType = [], {}, {}
    quesIds = [quesId for quesId in vqaEval.params["question_id"]]
    gts, res = {}, {}
    for quesId in quesIds:
        gts[quesId] = vqa.qa[quesId]
        res[quesId] = vqaRes.qa[quesId]
    for quesId in quesIds:
        prediction, reference = vqav2.postprocess_generation(
            res[quesId]["answer"], [ans["answer"] for ans in gts[quesId]["answers"]]
        )
        acc = vqa_acc.compute(predictions=prediction, references=reference)

        quesType = gts[quesId]["question_type"]
        ansType = gts[quesId]["answer_type"]
        avgGTAcc = acc["vqa_accuracy"]
        accQA.append(avgGTAcc)
        if quesType not in accQuesType:
            accQuesType[quesType] = []
        accQuesType[quesType].append(avgGTAcc)
        if ansType not in accAnsType:
            accAnsType[ansType] = []
        accAnsType[ansType].append(avgGTAcc)
        vqaEval.setEvalQA(quesId, avgGTAcc)
        vqaEval.setEvalQuesType(quesId, quesType, avgGTAcc)
        vqaEval.setEvalAnsType(quesId, ansType, avgGTAcc)
    vqaEval.setAccuracy(accQA, accQuesType, accAnsType)

    return vqaEval.accuracy


@pytest.mark.parametrize(
    "questions_path,annotations_path,result_path",
    [
        (
            os.path.join(config.vqav2_dir, VQAv2_FILE_NAME["questions"]["val"]),
            os.path.join(config.vqav2_dir, VQAv2_FILE_NAME["annotations"]["val"]),
            os.path.join(
                "tests", "vqa_accuracy", "v2_OpenEnded_mscoco_val2014_results.json"
            ),
        ),
        # you can add more test case to validate correctness of `vqa_accuracy`
    ],
)
def test_vqa_accuracy(questions_path, annotations_path, result_path):
    standard = standard_evaluate(questions_path, annotations_path, result_path)
    testbed = custom_evaluate(questions_path, annotations_path, result_path)

    assert standard == testbed
