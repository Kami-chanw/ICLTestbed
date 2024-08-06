from pathlib import Path
from typing import Callable, Dict
import warnings
import datasets


def split_generators(
    expand_path_fn: Callable[[str, str], Path],
    type_split_file_dict: Dict[str, Dict[str, str]],
    verbose: bool = True,
):
    """
    This method is used in `_split_generatiors` of hugging face style datasets that derived from `datasets.GeneratorBasedBuilder`.
    It will form and check existence of data paths passed in `_generate_examples`.

    Args:
        expand_path_fn (`Callable[[str, str], Path]`): 
            A method requires `file_type` and `split_name` as input and returns a `pathlib.Path` object \
            pointing to the full path of `file_type` (e.g. questions, annotations and images).
        type_split_file_dict (`Dict[str, Dict[str, str]]`):
            A multi dict that contains file type -> split name -> file name.
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
            if not path.exists():
                missing_files.append(str(path))
                splits.remove(split_name)
                if verbose:
                    warnings.warn(
                        f"{str(path)} is not exists. The {split_name} split will not be loaded."
                    )
                break

    if len(splits) == 0:
        raise FileNotFoundError(
            f"Unable to load any splits because following files are not found: {', '.join(missing_files)}"
        )

    gen_kwargs = {
        split_name: {
            f"{file_type}_path": data_path[split_name][file_type]
            for file_type in type_split_file_dict.keys()
        }
        for split_name in splits
    }

    gen_name = {
        "train": datasets.Split.TRAIN,
        "test": datasets.Split.TEST,
        "val": datasets.Split.VALIDATION,
    }

    return [
        datasets.SplitGenerator(
            name=gen_name[split] if split in gen_name else split,
            gen_kwargs=gen_kwargs[split],
        )
        for split in splits
    ]
