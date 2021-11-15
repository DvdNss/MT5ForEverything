# coding:utf-8
"""
Filename: utils.py
Author: @DvdNss

Created on 11/15/2021
"""

import json
import os


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

    # Saving the config as json file
    with open(filename, 'w') as config:
        json.dump(args_dict, config, indent=2)

    return filename
