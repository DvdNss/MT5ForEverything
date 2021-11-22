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


def squad_converter():
    """
    Convert squad to a better format.
    """

    dataset = load_dataset('squad')
    transformed_dataset = {}

    # Convert dataset for each subset
    for subset in dataset:

        # Generate new dict
        transformed_subset = {}

        # Transform subset
        for row in tqdm(dataset[subset]):
            context = row['context']
            question = row['question']
            index = row['answers']['answer_start'][0]
            answer = row['answers']['text'][0]

            if row['context'] not in transformed_subset:
                transformed_subset[f"{context}"] = []
            else:
                transformed_subset[f"{context}"].append({
                    'question': question,
                    'index': index,
                    'answer': answer
                })

        new_subset = []
        for key in transformed_subset.keys():
            new_subset.append([key, transformed_subset[key]])

        transformed_dataset[subset] = new_subset

    return transformed_dataset


def squad_qaqg(tab):
    """
    Map function for squad qa-qg task.

    :param tab: set of features

    :return: input/output list of size 2
    """

    context = tab[0]
    items = tab[1]  # question, index, answer

    input_ae = context
    for answer in items:
        question = answer['question']
        index = answer['index']
        answer = answer['answer']
        # TODO: generate input/output dataset for squad here

        context[index: index + len(answer['answer'])].replace(answer['answer'])

    return 'ok'


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
    # # SQUAD
    # dataset = squad_converter()
    # print(dataset)
    # dataset_to_csv(name='custom', dataset=dataset, path='../data/squad-reworked/', map_function=squad_qaqg, sep='\t')

    # # MLSUM
    # dataset_to_csv(name='mlsum', option='fr', path='../data/mlsum/', map_function=mlsum_summarization, sep='\t')
