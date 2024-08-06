import re
from typing import List, Union


def postprocess_generation(predictions: Union[str, List[str]]):
    is_batched = True
    if isinstance(predictions, str):
        predictions = [predictions]
        is_batched = False

    def process(pred):
        pred = re.split("Question|Answer|Short", pred, 1)[0]
        pred = re.split(", ", pred, 1)[0]
        return pred

    # more postprocess will be applied in official VQA evaluation procedure
    result = [process(pred) for pred in predictions]

    if is_batched:
        return result
    else:
        return result[0]
