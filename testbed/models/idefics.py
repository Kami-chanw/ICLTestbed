from functools import partial
import os
import re
from typing import Any, Dict, List, Union
import warnings
import transformers
from packaging import version
from PIL.Image import Image
from transformers import (
    IdeficsForVisionText2Text,
    AutoProcessor,
)

from testbed.models.model_base import ModelBase


class Idefics(ModelBase):
    def __init__(
        self,
        model_root,
        processor_class=AutoProcessor,
        model_class=IdeficsForVisionText2Text,
        processor_args=None,
        model_args=None,
        **common_args,
    ):
        super().__init__(
            model_root=model_root,
            processor_class=processor_class,
            model_class=model_class,
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
        )

        self._model_name = os.path.basename(model_root).lower()
        if not re.fullmatch(r"^idefics-\d+b[a-zA-Z-]*$", self._model_name):
            warnings.warn(
                "The model type cannot be detected automatically in `model_root`, which may lead to unexpected behaviors."
            )
            self._model_name = None

    @property
    def default_prompt_template(self):
        # see https://arxiv.org/pdf/2306.16527
        # fmt: off
        return (
            "{% if messages[0]['role'] == 'instruction' %}"
                "Instruction: {{ messages[0]['content'] }}\n"
                "{% set messages = messages[1:] %}"
            "{% endif %}"
            "{% set last_role = messages[0]['role'] %}"
            "{% for message in messages %}"
                "{% if message['role'] != '' %}"
                    "{{ message['role'].capitalize() }}"
                    "{% if not 'content' in message or message['content'][0]['type'] == 'image' %}"
                        "{{':'}}"
                    "{% else %}"
                        "{{': '}}"
                    "{% endif %}" 
                "{% endif %}"
                "{% if 'content' in message %}"
                    "{% for line in message['content'] %}"
                        "{% if line['type'] == 'text' %}"
                            "{{ line['text'] }}"
                        "{% elif line['type'] == 'image' %}"
                            "{{- '<image>' }}"
                        "{% endif %}"
                        "{% if not loop.last %}"
                            " "
                        "{%+ endif %}"
                    "{% endfor %}"
                    "{% set is_end_of_round = loop.nextitem is not defined or loop.nextitem['role'] == last_role %}"
                    "{% if is_end_of_round %}"
                        "\n\n"
                    "{% else %}"
                        " "
                    "{%+ endif %}"
                "{% endif %}"
            "{% endfor %}"
        )
        # fmt: on

    def process_input(
        self,
        text: Union[str, List[Union[str, Dict[str, Any]]], List[List[Dict[str, Any]]]],
        images: Union[List[Image], List[List[Image]]],
        prompt_template: str = None,
        **kwargs,
    ):
        """
        Processes text and image inputs for the model.

        Args:
            text (str, List[Union[str, Dict[str, Any]]], List[List[Dict[str, Any]]]):
                A single string, a list of strings or dictionaries, or a nested list (batch) of strings/dictionaries.
                For unbatched input (single text), this should be a string or a list of dict, where each item is
                either a string or a doct (following the transformers' conversation format with
                keys like "role" and "content").
                For batched input, this should be a nested list (list of lists) or a list of strings

            images (Union[List[Image], List[List[Image]]]):
                A list of images or a list of lists of images. For unbatched input, this should be a single-level list
                of images. For batched input, this should be a nested list where each inner list represents a batch of images.
                Each image should be an instance of the `Image` class.

            prompt_template (str, optional):
                An optional template string used to format the input texts if they are provided as dictionaries.

            **kwargs:
                Additional keyword arguments passed to the `processor`.

        Returns:
            The output of the `processor` function, which is the processed input ready for the model.
        """
        if isinstance(text, str) or (
            isinstance(text, list) and isinstance(text[0], dict)
        ):
            text = [text]
            images = [images]

        if isinstance(text[0][0], dict):
            text = self.apply_prompt_template(text, prompt_template=prompt_template)

        if version.parse(transformers.__version__) < version.parse("4.46.0"):
            assert len(text) == len(images)
            inputs = []
            for i, (ctx, image_list) in enumerate(zip(text, images)):
                ctx = ctx.split("<image>")

                if len(ctx) - 1 != len(image_list):
                    raise ValueError(
                        f"In the {i}-th input, the number of images {len(image_list)} does not match the number of image tokens {len(text) - 1} in the text."
                    )
                result = []
                for seg, image in zip(ctx, image_list):
                    if seg != "":
                        result.append(seg)
                    result.append(image)
                if ctx[-1] != "":  # the last question without answer
                    result.append(ctx[-1])
                inputs.append(result)

            process = partial(self.processor, prompts=inputs)
        else:
            process = partial(self.processor, text=text, images=images)

        return process(
            padding=kwargs.pop("padding", True),
            return_tensors=kwargs.pop("return_tensors", "pt"),
            **kwargs,
        )
