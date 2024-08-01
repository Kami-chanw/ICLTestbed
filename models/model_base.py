import abc
from functools import lru_cache
from typing import Dict, List

import torch
from contextlib import suppress

from packaging import version


class ModelBase(abc.ABC):
    def __init__(self, precision, device=None):
        precision_dict = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "amp": torch.cuda.amp.autocast,
            "amp_bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
        }
        
        self.autocast = precision_dict.get(precision, suppress)
        self.cast_dtype = precision_dict.get(precision, None)

        self.processor = None
        self.model = None
        self.device = device

        self.prompt_template = None

    @property
    def default_prompt_template(self):
        raise NotImplementedError(
            "The property `default_prompt_template` should be specified by derived class."
        )

    @property
    def model_name(self):
        raise NotImplementedError(
            "The property `model_name` should be specified by derived class."
        )

    def generate(self, text, images, **kwargs):
        raise NotImplementedError("The method `generate` should be override by derived class.")

    def _generate(self, args_to_processor, args_to_generate):
        if not isinstance(args_to_processor, dict) or not isinstance(args_to_generate, dict):
            raise ValueError(
                "Implement erorr: arguments pass to `ModelBase._generate` should all be dict."
            )

        args_to_processor["padding"] = True
        args_to_processor["return_tensors"] = "pt"
        inputs = self.processor(**args_to_processor).to(self.device)
        seq_len = inputs.input_ids.shape[-1]

        generated_ids = self.model.generate(**inputs, **args_to_generate)
        generated_ids = generated_ids[:, seq_len:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def apply_prompt_template(
        self, conversation: List[Dict[str, str]], prompt_template=None, tokenize=False, **kwargs
    ):
        if prompt_template is not None:
            template = prompt_template
        else:
            template = (
                self.default_prompt_template
                if self.prompt_template is None
                else self.prompt_template
            )
            
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                conversation, chat_template=template, tokenize=tokenize, **kwargs
            )
        elif hasattr(self.processor.tokenizer, "apply_chat_template"):
            return self.processor.tokenizer.apply_chat_template(
                conversation, chat_template=template, tokenize=tokenize, **kwargs
            )
        
        compiled_template = self._compile_jinja_template(template)
        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple))
            or hasattr(conversation[0], "messages")
        ):
            conversations = conversation
            is_batched = True
        else:
            conversations = [conversation]
            is_batched = False

        if not "add_generation_prompt" in kwargs:
            kwargs["add_generation_prompt"] = False

        rendered = []
        for chat in conversations:
            if hasattr(chat, "messages"):
                # Indicates it's a Conversation object
                chat = chat.messages
            rendered_chat = compiled_template.render(messages=chat, **kwargs)
            rendered.append(rendered_chat)

        if not is_batched:
            rendered = rendered[0]

        if tokenize:
            return self.processor.tokenizer(rendered)["input_ids"]
        return rendered

    @lru_cache
    def _compile_jinja_template(self, template):
        try:
            import jinja2
            from jinja2.exceptions import TemplateError
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError:
            raise ImportError("apply_template requires jinja2 to be installed.")

        if version.parse(jinja2.__version__) < version.parse("3.0.0"):
            raise ImportError(
                "apply_template requires jinja2>=3.0.0 to be installed. Your version is "
                f"{jinja2.__version__}."
            )

        def raise_exception(message):
            raise TemplateError(message)

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = raise_exception
        return jinja_env.from_string(template)
