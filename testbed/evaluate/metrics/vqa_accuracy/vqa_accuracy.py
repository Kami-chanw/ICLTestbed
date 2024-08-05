import datasets
import evaluate

_DESCRIPTION = """
VQA accuracy is a evaluation metric which is robust to inter-human variability in phrasing the answers:
Acc(`ans`) = min{ # humans that said `ans` / 3, 1 }
Where `ans` is answered by machine. In order to be consistent with 'human accuracies', machine accuracies are averaged over all 10 choose 9 sets of human annotators.
Note that to obtain results consistent with offical VQA evaluation, all inputs should be processed with `postprocess_generation` from testbed.data.vqav2.
"""


_KWARGS_DESCRIPTION = """
Args:
    predictions (`list` of `str`): Predicted answers.
    references (`list` of `str` lists): Ground truth answers. 
    answer_types (`list` of `str`, *optional*): Answer types corresponding to each questions.
    questions_type (`list` of `str`, *optional*): 

Returns:
    visual question answering accuracy (`float` or `int`): Accuracy accuracy. Minimum possible value is 0. Maximum possible value is 1.0, or the number of examples input, if `normalize` is set to `True`.. A higher accuracy means higher accuracy.

"""


_CITATION = """
@InProceedings{{VQA},
author      = {Stanislaw Antol and Aishwarya Agrawal and Jiasen Lu and Margaret Mitchell and Dhruv Batra and C. Lawrence Zitnick and Devi Parikh},
title       = {{VQA}: {V}isual {Q}uestion {A}nswering},
booktitle   = {International Conference on Computer Vision (ICCV)},
year        = {2015},
}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class VQAaccuracy(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Sequence(
                        datasets.Value("string", id="sequence"), id="references"
                    ),
                }
            ),
            reference_urls=[
                "https://visualqa.org/evaluation.html",
                "https://github.com/GT-Vision-Lab/VQA/blob/master",
            ],
        )

    def _compute(
        self,
        predictions,
        references,
        answer_types=None,
        question_types=None,
        precision=2,
    ):
        if answer_types is None:
            answer_types = [None] * len(predictions)

        if question_types is None:
            question_types = [None] * len(predictions)

        if not len(predictions) == len(answer_types) == len(question_types):
            raise ValueError(
                "The length of predictions, answer_types and question_types doesn't match."
            )

        total, ans_type_dict, ques_type_dict = [], {}, {}

        for pred, gts, ans_type, ques_type in zip(
            predictions, references, answer_types, question_types
        ):
            accuracy = []
            for i in range(len(gts)):
                other_gt = gts[:i] + gts[i + 1 :]
                matching_ans = [item for item in other_gt if item == pred]
                accuracy.append(min(1, len(matching_ans) / 3))

            vqa_acc = sum(accuracy) / len(accuracy)
            total.append(vqa_acc)

            if ans_type is not None:
                if ans_type not in ans_type_dict:
                    ans_type_dict[ans_type] = []
                ans_type_dict[ans_type].append(vqa_acc)

            if ques_type is not None:
                if ques_type not in ques_type_dict:
                    ques_type_dict[ques_type] = []
                ques_type_dict[ques_type].append(vqa_acc)

        # the following key names follow the naming of the official evaluation results
        result = {"overall": round(100 * sum(total) / len(total), precision)}

        if len(ans_type_dict) > 0:
            result["perAnswerType"] = {
                ans_type: round(
                    100 * sum(accuracy_list) / len(accuracy_list), precision
                )
                for ans_type, accuracy_list in ans_type_dict.items()
            }

        if len(ques_type_dict) > 0:
            result["perQuestionType"] = {
                ques_type: round(
                    100 * sum(accuracy_list) / len(accuracy_list), precision
                )
                for ques_type, accuracy_list in ques_type_dict.items()
            }

        return result
