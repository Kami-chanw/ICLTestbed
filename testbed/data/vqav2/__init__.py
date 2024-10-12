import re
from typing import List, Optional, Union


def postprocess_generation(predictions: Union[str, List[str]], stop_words: Optional[List[str]] = None):
    """
    Post-processes generated predictions by applying text normalization techniques.

    This function processes a single prediction or a list of predictions, allowing for optional truncation 
    based on stop words. It returns the processed prediction(s) either as a string or a list, depending on the input type.

    Args:
        predictions (Union[str, List[str]]): The generated text prediction(s) to be processed.
        stop_words (Optional[List[str]], *optional*): A list of stop words to truncate predictions. If provided, the 
            prediction will be truncated at the first occurrence of any stop word.

    Returns:
        Union[str, List[str]]: The post-processed prediction(s). Returns a string if a single prediction is given, 
        or a list if multiple predictions are provided.
    """
    is_batched = True
    if isinstance(predictions, str):
        predictions = [predictions]
        is_batched = False

    def process(pred):
        if stop_words is not None:
            pred = re.split("|".join(stop_words), pred, 1)[0]
        return pred

    # more postprocess will be applied in official VQA evaluation procedure
    result = [process(pred) for pred in predictions]

    if is_batched:
        return result
    else:
        return result[0]
