import re
from typing import List, Union


def postprocess_generation(predictions: Union[str, List[str]]):
    if isinstance(predictions, str):
        predictions = [predictions]

    def process(pred):
        pred = re.split("Question|Answer|Short", pred, 1)[0]
        pred = re.split(", ", pred, 1)[0]
        return pred

    # more postprocess will be applied in official VQA evaluation procedure
    return [process(pred) for pred in predictions]
