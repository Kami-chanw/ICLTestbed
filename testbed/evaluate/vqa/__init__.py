from .vqa import VQA, VQAEval

def compute_vqa_accuracy(result_json_path, question_json_path, annotation_json_path):
    """Compute the VQA accuracy metric.

    Args:
        result_json_path (str): Path to the json file with model outputs
        question_json_path (str): Path to the json file with questions
        annotation_json_path (str): Path to the json file with annotations

    Returns:
        float: VQA accuracy
    """

    # create vqa object and vqaRes object
    vqa = VQA(annotation_json_path, question_json_path)
    vqaRes = vqa.loadRes(result_json_path, question_json_path)

    # create vqaEval object by taking vqa and vqaRes
    # n is precision of accuracy (number of places after decimal), default is 2
    vqaEval = VQAEval(vqa, vqaRes, n=2)

    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate()

    return vqaEval.accuracy
