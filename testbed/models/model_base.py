import abc
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import torch
from contextlib import suppress

from packaging import version


class ModelBase(abc.ABC):
    def __init__(self, precision, device="auto"):
        """
        Args:
            precision (string): precision used to load model. It can be one of "bf16", "fp16", "fp32", "amp" and "amp_bf16", \
            which represents loading with torch.bfloat16, torch.float16, torch.float32 or auto mixture precision, respectively.
            device (torch.device or str, defaults to None): device that tensors, models, etc. should be moved to. If it is set to \
            "auto", the model should load with `device_map="auto"`, in this case, any `tensor.to(device)` operation will have no effect.
        """
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
        self.device = device if device != "auto" else None

        self.prompt_template = None

    @property
    def default_prompt_template(self):
        raise NotImplementedError

    @property
    def model_name(self):
        raise NotImplementedError

    def generate(self, **kwargs):
        raise NotImplementedError

    def _generate(self, args_to_processor, args_to_generate):
        if not isinstance(args_to_processor, dict) or not isinstance(
            args_to_generate, dict
        ):
            raise ValueError(
                "Implement erorr: arguments pass to `ModelBase._generate` should all be dict."
            )

        args_to_processor["padding"] = True
        args_to_processor["return_tensors"] = "pt"
        inputs = self.processor(**args_to_processor).to(
            self.device if self.device is not None else "cuda"
        )

        seq_len = inputs.input_ids.shape[-1]

        generated_ids = self.model.generate(**inputs, **args_to_generate)
        generated_ids = generated_ids[:, seq_len:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def apply_prompt_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        prompt_template: Optional[str] = None,
        tokenize=False,
        **kwargs
    ):
        if prompt_template is None:
            prompt_template = (
                self.default_prompt_template
                if self.prompt_template is None
                else self.prompt_template
            )

        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                conversation, chat_template=prompt_template, tokenize=tokenize, **kwargs
            )

        # all tokenziers have apply_chat_template method
        return self.processor.tokenizer.apply_chat_template(
            conversation, chat_template=prompt_template, tokenize=tokenize, **kwargs
        )
