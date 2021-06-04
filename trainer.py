# coding:utf-8
"""
Filename : trainer.py
Role : trainer overwrite

@author : Sunwaee
"""

from typing import Any, Dict, Union

import torch
from torch import nn
from transformers import Trainer as HFTrainer
from transformers.file_utils import is_apex_available

if is_apex_available():
    from apex import amp

from utils import label_smoothed_nll_loss


class Trainer(HFTrainer):
    """
    Trainer.
    """

    def __init__(self, label_smoothing: float = 0, **kwargs):
        """
        Initializes Trainer.

        :param label_smoothing: label smoothing rate
        :param kwargs: other args
        """

        super().__init__(**kwargs)
        self.label_smoothing = label_smoothing

    def _training_step(self, model: nn.Module,
                       inputs: Dict[str, Union[torch.Tensor, Any]],
                       optimizer: torch.optim.Optimizer) -> float:
        """
        Overrides training step to support label smoothing.

        :param model: model
        :param inputs: inputs
        :param optimizer: optimizer
        :return: loss items
        """

        # Training the model
        model.train()

        # Sending keys to device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        # Forcing return tuple
        if isinstance(model, nn.DataParallel):
            inputs["return_tuple"] = True

        # Getting output
        if self.label_smoothing == 0:
            outputs = model(**inputs)
            loss = outputs[0]
        else:
            labels = inputs.pop("labels")
            labels[labels == -100] = model.config.pad_token_id
            outputs = model(**inputs)
            lprobs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.label_smoothing, ignore_index=model.config.pad_token_id
            )

        # Calculating loss
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # Taking fp16 in account
        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()
