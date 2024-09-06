from typing import List, Union


def postprocess_generation(predictions: Union[str, List[str]]):
    is_batched = True
    if isinstance(predictions, str):
        predictions = [predictions]
        is_batched = False

    def process(pred):
        return pred.split("Caption", 1)[0]

    result = [process(pred) for pred in predictions]

    if is_batched:
        return result
    else:
        return result[0]