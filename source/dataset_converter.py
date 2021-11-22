# coding:utf-8
"""
Filename: dataset_converter.py
Author: @DvdNss

Created on 11/21/2021
"""
import os

import pandas
from datasets import load_dataset
from tqdm import tqdm


def mlsum_summarization(dict: dict):
    """
    Map function for mlsum summarization task.

    :param dict: set of features
    :return: input/output list of size 2
    """

    input = dict['text']
    output = dict['summary']

    return [input, output]


def dataset_to_csv(name: str, path: str, map_function, dataset: dict = {}, size: int = -1, option: str = '', **kwargs):
    """
    Convert dataset to input/output tsv file.

    :param path: tsv path
    :param name: dataset name
    :param option: dataset option
    :param map_function: function used to map dataset
    :param dataset: custom dataset
    :param size: max size of dataset
    """

    # Load specified dataset with given options
    if name != 'custom':
        dataset = load_dataset(path=name, name=option) if option != "" else load_dataset(path=name)
    else:
        dataset = dataset

    # Create output directory if needed
    try:
        os.makedirs(path + option)
    except FileExistsError:
        pass

    for key in dataset.keys():
        csv_path = f'{path}{option}/{key}.csv'
        subset = pandas.DataFrame(list(map(map_function, tqdm(dataset[key]))), columns=['source_text', 'target_text'])
        subset = subset[0:size]
        subset.to_csv(csv_path, columns=['source_text', 'target_text'], **kwargs)
        print(f'Successfully converted {name}:{option}[{key}] to {csv_path}. ')


if __name__ == '__main__':
    dataset_to_csv(name='mlsum', option='fr', path='../data/mlsum/', map_function=mlsum_summarization, sep='\t',
                   size=100000)
