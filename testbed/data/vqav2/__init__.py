from typing import List, Union


def postprocess_generation(predictions: Union[str, List[str]]):
    is_batched = True
    if isinstance(predictions, str):
        predictions = [predictions]
        is_batched = False

    # more postprocess will be applied in official VQA evaluation procedure
    result = [pred.split()[0] for pred in predictions]

    if is_batched:
        return result
    else:
        return result[0]
