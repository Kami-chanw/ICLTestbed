from testbed.models.model_base import ModelBase

from transformers import (
    IdeficsForVisionText2Text,
    IdeficsProcessor,
)


class Idefics(ModelBase):
    def __init__(self, model_root, precision, device=None):
        super().__init__(precision, device)

        self.processor = IdeficsProcessor.from_pretrained(
            model_root,
            torch_dtype=self.cast_dtype,
            local_files_only=True,
        )

        self.model = IdeficsForVisionText2Text.from_pretrained(
            model_root,
            torch_dtype=self.cast_dtype,
            local_files_only=True,
            device_map=device,
        )

    @property
    def model_name(self):
        return "idefics-9b"

    @property
    def default_prompt_template(self):
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
                    "{% endfor %}"
                "\n\n"
                "{% endif %}"
            "{% endfor %}"
        )
        # fmt: on

    def generate(self, texts, images, **kwargs):
        texts = self.apply_prompt_template(texts)
        inputs = []
        for i, (text, image_list) in enumerate(zip(texts, images)):
            text = text.split("<image>")
            result = []
            if len(text) - 1 != len(image_list):
                raise ValueError(
                    f"In the {i}-th input, the number of images does not match the number of image tokens in the text."
                )
            for seg, image in zip(text, image_list):
                if seg != "":
                    result.append(seg)
                result.append(image)
            if text[-1] != "":  # the last question without answer
                result.append(text[-1])
            inputs.append(result)
        return self._generate(
            {"prompts": inputs, "padding": True, "return_tensors": "pt"}, kwargs
        )
