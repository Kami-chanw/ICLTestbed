import re
from typing import List, Optional, Union
import nltk

def postprocess_generation(predictions: Union[str, List[str]], stop_words: Optional[List[str]] = None):
    """
    Post-processes generated predictions by normalizing 'hateful' and 'non-hateful' predictions to binary values.

    This function converts variations of 'hateful' and 'non-hateful' predictions into binary labels.
    - 'yes', 'hateful', etc. are mapped to 1.
    - 'no', 'non-hateful', etc. are mapped to 0.

    Args:
        predictions (Union[str, List[str]]): The generated text prediction(s) to be processed.
        stop_words (Optional[List[str]], *optional*): A list of stop words to truncate predictions. If provided, the 
            prediction will be truncated at the first occurrence of any stop word.

    Returns:
        Union[int, List[int]]: The post-processed binary prediction(s). Returns an integer if a single prediction is given, 
        or a list if multiple predictions are provided.
    """
    is_batched = True
    if isinstance(predictions, str):
        predictions = [predictions]
        is_batched = False

    hateful_keywords = ["yes", "y", "hateful", "hate"]
    non_hateful_keywords = ["no", "n", "non-hateful", "not hateful", "benign"]

    def process(pred):
        if stop_words is not None:
            pred = re.split("|".join(stop_words), pred, 1)[0]
        
        pred = pred.strip().lower()
        tokens = nltk.word_tokenize(pred)

        for token in tokens:
            if token in hateful_keywords:
                return 1 
            elif token in non_hateful_keywords:
                return 0 

        return 0

    result = [process(pred) for pred in predictions]

    if is_batched:
        return result
    else:
        return result[0]