from typing import Any, Dict, List, Union
from PIL.Image import Image
from transformers import (
    IdeficsForVisionText2Text,
    IdeficsProcessor,
)

from testbed.models.model_base import HookType, ModelBase


class Idefics(ModelBase):
    def __init__(
        self,
        model_root,
        processor_class=IdeficsProcessor,
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

    @property
    def model_name(self):
        return "idefics-9b"

    def _register_hook(self, register_fn_name, module_name_or_type, hook, **kwargs):
        pattern_prefix = None
        if module_name_or_type == HookType.TEXT_MODEL_LAYER:
            pattern_prefix = ""
        elif module_name_or_type == HookType.VISION_MODEL_LAYER:
            pattern_prefix = r"vision_model\.encoder\."
        if pattern_prefix is not None:
            module_name_or_type = r"model\." + pattern_prefix + r"layers\.\d+$"

        return super()._register_hook(
            register_fn_name,
            module_name_or_type,
            hook,
            use_regex=kwargs.pop("use_regex", False) or pattern_prefix is not None,
            **kwargs,
        )

    @property
    def default_prompt_template(self):
        # see https://arxiv.org/pdf/2306.16527
        # fmt: off
        return (
            "{% if messages[0]['role'] == 'instruction' %}"
                "Instruction: {{ messages[0]['content'] }}\n"
                "{% set messages = messages[1:] %}"
            "{% endif %}"
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
                        "{% if loop.last %}"
                            "{% if message['role'] == 'answer' or message['role'] == 'caption' %}"
                                "\n\n"
                            "{% else %}"
                                " "
                            "{%+ endif %}"
                        "{% endif %}"
                    "{% endfor %}"
                "{% endif %}"
            "{% endfor %}"
        )
        # fmt: on

    def process_input(
        self,
        texts: Union[
            List[Union[str, Dict[str, Any]]], List[List[Union[str, Dict[str, Any]]]]
        ],
        images: Union[List[Image], List[List[Image]]],
        padding: bool = True,
        return_tensors: str = "pt",
        prompt_template: str = None,
        **kwargs,
    ):
        """
        Processes text and image inputs for the model.

        Args:
            texts (Union[List[Union[str, Dict[str, Any]]], List[List[Union[str, Dict[str, Any]]]]]):
                A list of texts or a list of lists of texts. For unbatched input, this should be a single-level list
                where each item is either a string or a dictionary. For batched input, this should be a nested list
                (list of lists) where each inner list represents a batch of texts. Dictionaries can follow the
                transformers' conversation format, with keys like "role" and "content".

            images (Union[List[Image], List[List[Image]]]):
                A list of images or a list of lists of images. For unbatched input, this should be a single-level list
                of images. For batched input, this should be a nested list where each inner list represents a batch of images.
                Each image should be an instance of the `Image` class.

            padding (bool, optional):
                Whether to pad the inputs to the same length. Defaults to True.

            return_tensors (str, optional):
                The type of tensors to return. Defaults to "pt" for PyTorch tensors.
                Can be set to other formats depending on the framework (e.g., "tf" for TensorFlow).

            prompt_template (str, optional):
                An optional template string used to format the input texts if they are provided as dictionaries.

            **kwargs:
                Additional keyword arguments passed to the `processor`.

        Returns:
            The output of the `processor` function, which is the processed input ready for the model.
        """
        if isinstance(texts[0], dict) or (
            isinstance(texts[0], list) and isinstance(texts[0][0], dict)
        ):
            texts = self.apply_prompt_template(texts, prompt_template=prompt_template)
        inputs = []
        for i, (text, image_list) in enumerate(zip(texts, images)):
            text = text.split("<image>")
            result = []
            if len(text) - 1 != len(image_list):
                raise ValueError(
                    f"In the {i}-th input, the number of images {len(image_list)} does not match the number of image tokens {len(text) - 1} in the text."
                )
            for seg, image in zip(text, image_list):
                if seg != "":
                    result.append(seg)
                result.append(image)
            if text[-1] != "":  # the last question without answer
                result.append(text[-1])
            inputs.append(result)

        return self.processor(
            prompts=inputs, padding=padding, return_tensors=return_tensors, **kwargs
        )