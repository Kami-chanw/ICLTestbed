import os
import re
import warnings
from typing import Any, Dict, List, Union
from PIL.Image import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)

from testbed.models.model_base import HookType, ModelBase


class Idefics2(ModelBase):
    def __init__(
        self,
        model_root,
        processor_class=AutoProcessor,
        model_class=AutoModelForVision2Seq,
        processor_args=None,
        model_args=None,
        **common_args,
    ):
        self._model_name = os.path.basename(model_root).lower()
        if not re.fullmatch(r"^idefics2-\d+b[a-zA-Z-]*$", self._model_name):
            warnings.warn(
                "The model type cannot be detected automatically in `model_root`, which may lead to unexpected behaviors."
            )
            self._model_name = None

        processor_args = (
            processor_args
            if processor_args
            else dict(
                chat_template=self.default_prompt_template, do_image_splitting=False
            )
        )

        super().__init__(
            model_root=model_root,
            processor_class=processor_class,
            model_class=model_class,
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
        )

    def _register_hook(self, register_fn_name, module_name_or_type, hook, **kwargs):
        pattern_prefix = None
        if module_name_or_type == HookType.TEXT_MODEL_LAYER:
            pattern_prefix = r"text_model\."
        elif module_name_or_type == HookType.VISION_MODEL_LAYER:
            pattern_prefix = r"vision_model\.encoder\."
        if pattern_prefix is not None:
            module_name_or_type = pattern_prefix + r"layers\.\d+$"

        return super()._register_hook(
            register_fn_name, module_name_or_type, hook, **kwargs
        )

    @property
    def default_prompt_template(self):
        # adopt idefics1 prompt template, see https://arxiv.org/pdf/2306.16527
        # fmt: off
        template = (
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
                                "<end_of_utterance>\n"
                            "{% else %}"
                                " "
                            "{%+ endif %}"
                        "{% else %}"
                            " "
                        "{%+ endif %}"
                    "{% endfor %}"
                "{% endif %}"
            "{% endfor %}"
        )
        # fmt: on
        if "base" in self._model_name:
            # base model doesn't have <end_of_utterance> token
            return template.replace("<end_of_utterance>", "\n")
        return template

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
        return self.processor(
            text=texts,
            images=images,
            padding=padding,
            return_tensors=return_tensors,
            **kwargs,
        )
