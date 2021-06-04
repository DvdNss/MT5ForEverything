# coding:utf-8
"""
Filename : data_collector.py
Role : data collection

@author : Sunwaee
"""

from typing import Dict, List

import torch
from transformers import (
    MT5Tokenizer,
)


def trim_batch(input_ids, pad_token_id, attention_mask=None):
    """
    Remove columns that are populated exclusively by pad_token_id.

    :param input_ids: input ids
    :param pad_token_id: pad token id
    :param attention_mask: attention mask
    :return: input_ids and eventually attention mask
    """

    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask]


class DataCollector:
    """
    Data collection.
    """

    def __init__(self, tokenizer: MT5Tokenizer, mode: str = 'training', using_tpu: bool = False) -> None:
        """
        Initiliazes DataCollector.

        :param tokenizer: tokenizer
        :param mode: mode
        :param using_tpu: whether to use tpu or not
        """

        self.tokenizer = tokenizer
        self.using_tpu = using_tpu
        self.mode = mode

    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Takes a list of samples and collates them into a batch.

        :param batch: list of samples
        :return: dictionary of tensors
        """

        # Stacking elements
        input_ids = torch.stack([example['source_ids'] for example in batch])
        target_ids = torch.stack([example['target_ids'] for example in batch])
        attention_mask = torch.stack([example['attention_mask'] for example in batch])

        # Setting pad token id
        pad_token_id = self.tokenizer.pad_token_id

        # Preventing trim on TPU use
        if not self.using_tpu:
            input_ids, attention_mask = trim_batch(input_ids, pad_token_id, attention_mask=attention_mask)
            target_ids = trim_batch(target_ids, pad_token_id)

        # Shifting decoder inputs to the right
        lm_labels = target_ids.clone()
        decoder_input_ids = self._shift_right_mt5(lm_labels)
        if self.mode == 'training':
            lm_labels[lm_labels[:, :] == pad_token_id] = -100

        # Creating dictionary with results
        params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_input_ids": decoder_input_ids
        }

        return params

    def _shift_right_mt5(self, input_ids):
        """
        Shifts inputs to the right.

        :param input_ids:
        :return:
        """

        # Setting pad token id
        pad_token_id = self.tokenizer.pad_token_id

        # Assertions to prevent bugs
        assert pad_token_id is not None, \
            "self.model.config.pad_token_id has to be defined."

        # Shifting inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = pad_token_id

        # Replacing possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        # Assertions to make sure output is bugfree
        assert torch.all(shifted_input_ids >= 0).item(), \
            "Verify that `labels` has only positive values and -100"

        return shifted_input_ids
