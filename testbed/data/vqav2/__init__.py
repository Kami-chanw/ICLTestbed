from typing import List, Union


def postprocess_generation(predictions: Union[str, List[str]]):
    is_batched = True
    if isinstance(predictions, str):
        predictions = [predictions]
        is_batched = False

    def process(pred):
        if pred:
            return pred.split("\n")[0]
        return ""
    # more postprocess will be applied in official VQA evaluation procedure
    result = [process(pred) for pred in predictions]

    if is_batched:
        return result
    else:
        return result[0]
