from pathlib import Path
from typing import Callable, Dict, List, Union
import warnings
import datasets

DATASET_RETRIEVER_MAPPING = {}
POSTPROCESS_MAPPING = {}


def register_dataset_retriever(dataset_name, retriever):
    """
    Registers a dataset retriever function for a given dataset.

    The retriever function is responsible for converting an individual item from the dataset
    into the format required for the model's input. The retriever function takes
    two arguments: an item from the dataset and a boolean flag indicating whether the item
    is the last one in the context.

    Args:
        dataset_name (str):
            The name of the dataset for which the retriever function is being registered.
            It must be a non-empty string.
        retriever (Callable[[Any, bool], Any]):
            A callable function that processes each dataset item. It transforms an item into
            the format needed for the model. The boolean flag indicates if the item is the
            last in the current context.
    """
    DATASET_RETRIEVER_MAPPING[dataset_name] = retriever


def register_postprocess(dataset_name: str, postprocess: Callable):
    """
    Registers a post-process generation function for a given dataset.

    postprocess process model generation by applying text normalization techniques.
    It processes a single prediction or a list of predictions, allowing for optional truncation
    based on stop words.

    Args:
        dataset_name (str):
            The name of the dataset for which the retriever function is being registered.
            It must be a non-empty string.
        postprocess (Callable:
            The user defined process function to be registered. The function should have the following signature::

                postprocess(predictions, stop_words) -> str
    """
    POSTPROCESS_MAPPING[dataset_name] = postprocess


def most_common_from_dict(dct):
    lst = [x["answer"] for x in dct]
    return max(set(lst), key=lst.count)


def split_generators(
    expand_path_fn: Callable[[str, str], Path],
    type_split_file_dict: Dict[str, Dict[str, Union[str, List[str]]]],
    verbose: bool = True,
):
    """
    This method is used in `_split_generatiors` of hugging face style datasets that derived from `datasets.GeneratorBasedBuilder`.
    It will form and check existence of data paths passed in `_generate_examples`, 
    then pass split and file type into the `generate_examples` method as parameters `split` and `{file_type}_path` respectively.

    Args:
        expand_path_fn (`Callable[[str, str], Union[Path, List[Path]]]`): 
            A method requires `file_type` and `split_name` as input and returns a `pathlib.Path` object \
            or a list of `pathlib.Path` object pointing to the full path of `file_type` (e.g. questions, annotations and images).
        type_split_file_dict (`Dict[str, Dict[str, Union[str, List[str]]]]`):
            A multi dict that contains file type -> split name -> file name(s).
        verbose (`bool`, defaults to `True`):
            Whether to warn if a split files are not exist.
    
    Returns:
        [`List[datasets.SplitGenerator]`]: a list of `datasets.SplitGenerator`, whose length is depends on valid dataset files. \
            `{file_type}_path`(e.g. `questions_path`, `annotations_path` depends on the keys of `type_split_file_dict`) \
            will be passed to `_generate_examples` as input.
    """
    splits = set(
        [
            file_name
            for item in type_split_file_dict.values()
            for file_name in item.keys()
        ]
    )

    data_path = {
        split_name: {
            file_type: expand_path_fn(file_type, split_name)
            for file_type in type_split_file_dict.keys()
            if split_name in type_split_file_dict[file_type]
        }
        for split_name in splits
    }

    missing_files = []

    for split_name, dct in data_path.items():
        for path in dct.values():
            if not isinstance(path, list):
                path = [path]
            for p in path:
                if not p.exists():
                    missing_files.append(str(p))
                    if split_name in splits:
                        splits.remove(split_name)
                    if verbose:
                        warnings.warn(
                            f"{str(p)} is not exists. The {split_name} split will not be loaded."
                            "You can suppress this warning by set verbose to False when loading dataset."
                        )
                    break

    if len(splits) == 0:
        raise FileNotFoundError(
            f"Unable to load any splits because following files are not found: {', '.join(missing_files)}"
        )

    gen_kwargs = {
        split_name: {
            **{"split": split_name},
            **{
                f"{file_type}_path": data_path[split_name][file_type]
                for file_type in type_split_file_dict.keys()
            },
        }
        for split_name in splits
    }

    gen_name = {
        "train": datasets.Split.TRAIN,
        "test": datasets.Split.TEST,
        "val": datasets.Split.VALIDATION,
        # aliasing
        "valid": datasets.Split.VALIDATION,
        "validation": datasets.Split.VALIDATION,
    }

    return [
        datasets.SplitGenerator(
            name=gen_name[split] if split in gen_name else split,
            gen_kwargs=gen_kwargs[split],
        )
        for split in splits
    ]
