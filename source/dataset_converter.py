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
                transformed_subset[f"{context}"] = sorted(transformed_subset[f"{context}"], key=lambda d: d['index'])

        new_subset = []
        for key in transformed_subset.keys():
            new_subset.append([key, transformed_subset[key]])

        transformed_dataset[subset] = new_subset

    return transformed_dataset


def squad_to_csv(dataset, path: str = 'dataframe.csv', size: int = -1, **kwargs):
    """
    Convert squad to qa-qg-ae csv.

    :param dataset: transformed squad dataset
    :param size: max size of transformed dataset
    :param path: path to generated csv files
    """

    for subset in dataset.keys():  # train, validation
        # Init empty list
        data = []
        csv_path = f"{path}{subset}.csv"

        for text in tqdm(dataset[subset]):
            # Pull context
            context = text[0]

            # Generate ae input
            input_ae = f"extract answer: {context}"
            output_ae = context

            count = 0
            for content in text[1]:
                # Pull content
                question = content['question']
                index = content['index']
                answer = content['answer']

                # Generate qg input/output
                input_qg = f"generate question: {context[0:index]}<hl> {answer} <hl>{context[index + len(answer):-1]}"
                output_qg = question
                data.append([input_qg, output_qg])

                # Generate qa input/output
                input_qa = f"context: {context} <sep> question: {question}"
                output_qa = answer
                data.append([input_qa, output_qa])

                # Generate ae output
                output_ae = f"{output_ae[0:index + count]}<hl> {answer} <hl>{output_ae[index + count + len(answer):]}"

                count += 10

            # Add transformed example in data
            data.append([input_ae, output_ae])

        dataframe = pandas.DataFrame(data, columns=['source_text', 'target_text'])
        dataframe = dataframe[:size] if size != -1 else dataframe
        dataframe.to_csv(csv_path, columns=['source_text', 'target_text'], **kwargs)
        print(f'Successfully converted squad:{subset}[{len(dataframe)}] to {csv_path}. ')


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
    dataset = squad_converter()
    squad_to_csv(dataset, path='../data/squad/', sep='\t')
