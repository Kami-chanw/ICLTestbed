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
    predictions (`str`): A predicted answer.
    references (`list` of `str`): Ground truth answers. 

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

    def _compute(self, predictions, references):
        total = []
        for pred, gts in zip(predictions, references):
            accuracy = []
            for i in range(len(gts)):
                other_gt = gts[:i] + gts[i + 1 :]
                matching_ans = [item for item in other_gt if item == pred]
                accuracy.append(min(1, len(matching_ans) / 3))
            total.append(sum(accuracy) / len(accuracy))

        return {"vqa_accuracy": sum(total) / len(total)}
