# coding:utf-8
"""
Filename : utils.py
Role : tools necessary for the project

@author : Sunwaee
"""

import json
import os


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """
    Label smoothing to prevent overconfidence for classification tasks
    (from https://fairseq.readthedocs.io/en/latest/_modules/fairseq/criterions/label_smoothed_cross_entropy.html).
    """

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        bs = pad_mask.long().sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        bs = lprobs.shape[0]

    nll_loss = nll_loss.sum()
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss / bs, nll_loss / bs


def dict_to_json(args_dict: dict, filename: str) -> str:
    """
    Saves a dictionnary as a json file.

    :param args_dict: model dictionnary
    :param filename: json file name
    """

    assert filename[-5:] == '.json', \
        "filename must be a .json file. "

    # Create intermediate folders
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Saving the databuilder config as json file
    with open(filename, 'w') as config:
        json.dump(args_dict, config)

    return filename
