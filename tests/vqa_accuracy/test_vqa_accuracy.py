import evaluate
import pytest
import os
import sys

sys.path.insert(0, os.getcwd())

from testbed.data import vqav2
from vqa import VQA
from vqaEval import VQAEval


def round_to(dct, precision):
    for key, value in dct.items():
        if isinstance(value, dict):
            dct[key] = round_to(value, precision)
        else:
            dct[key] = round(value, precision)
    return dct


def standard_evaluate(questions_path, annotations_path, result_path, precision):
    vqa = VQA(annotations_path, questions_path)
    vqaRes = vqa.loadRes(result_path, questions_path)
    vqaEval = VQAEval(vqa, vqaRes, precision)
    vqaEval.evaluate(vqaRes.getQuesIds())
    return vqaEval.accuracy


def custom_evaluate(questions_path, annotations_path, result_path, precision):
    vqa = VQA(annotations_path, questions_path)
    vqaRes = vqa.loadRes(result_path, questions_path)

    vqa_acc = evaluate.load("Kamichanw/vqa_accuracy")

    predictions, references = [], []
    question_types, answer_types = [], []

    quesIds = vqaRes.getQuesIds()

    for quesId in quesIds:
        gt = vqa.qa[quesId]
        res = vqaRes.qa[quesId]

        prediction = vqav2.postprocess_generation(res["answer"])

        predictions.append(prediction)
        references.append([item["answer"] for item in gt["answers"]])
        question_types.append(gt["question_type"])
        answer_types.append(gt["answer_type"])

    result = vqa_acc.compute(
        predictions=predictions,
        references=references,
        question_types=question_types,
        answer_types=answer_types,
    )

    return round_to(result, precision)


@pytest.mark.parametrize(
    "questions_path,annotations_path,result_path",
    [
        (
            os.path.join(
                "tests",
                "vqa_accuracy",
                "v2_OpenEnded_mscoco_val2014_questions_subset.json",
            ),
            os.path.join(
                "tests",
                "vqa_accuracy",
                "v2_mscoco_val2014_annotations_subset.json",
            ),
            os.path.join("tests", "vqa_accuracy", "v2_OpenEnded_mscoco_val2014_results.json"),
        ),
        # you can add more test case to validate correctness of `vqa_accuracy`
    ],
)
def test_vqa_accuracy(questions_path, annotations_path, result_path):
    standard = standard_evaluate(questions_path, annotations_path, result_path, 2)
    testbed = custom_evaluate(questions_path, annotations_path, result_path, 2)

    assert standard == testbed
