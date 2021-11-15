# coding:utf-8
"""
Filename : utils.py
Role : tools necessary for the project

@author : Sunwaee
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

    # Saving the databuilder config as json file
    with open(filename, 'w') as config:
        json.dump(args_dict, config)

    return filename
