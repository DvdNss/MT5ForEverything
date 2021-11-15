# coding:utf-8
"""
Filename: use_case.py
Author: @DvdNss

Created on 11/15/2021
"""

import pandas as pd

from source import train, databuilder, pipelines

# Build datasets
train_data = [['first source', 'first_target'], ['second source', 'second_target']]
eval_data = [['first source', 'first_target'], ['second source', 'second_target']]

# Build dataframes
train_df = pd.DataFrame(train_data, columns=['source_text', 'target_text'])
eval_df = pd.DataFrame(eval_data, columns=['source_text', 'target_text'])

# Save dataframes
train_df.to_csv("data/train.tsv", sep='\t')
eval_df.to_csv("data/valid.tsv", sep='\t')

# Build databuilder config
databuilder_args = dict(
    source_max_length=100,
    target_max_length=20,
    tokenizer_save_path='model/tokenizer'
)

# Run databuilder
databuilder.run(args_dict=databuilder_args)

train_args = dict(
    output_dir="model",  # output directory of model & tokenizer
    model_config_save_path="model/config/config.json",  # output path of model config
    wandb_project_name='mt5-project',  # wandb project name for training tracking
    overwrite_output_dir=True,  # whether to overwrite output_dir or not
    seed=42,  # seed to use
    learning_rate=1e-4,  # learning rate
    num_train_epochs=10,  # number of epochs
    per_device_train_batch_size=1,  # train batch size
    per_device_eval_batch_size=1,  # eval batch size
    evaluation_strategy="epoch"  # evaluation strategy
)

# Starting training
train.run(args_dict=train_args)

pipeline_args = dict(
    pipeline='classic',
    pipeline_config_path='model/config/classic.json'
)

# Loading the pipeline
pipeline = pipelines.run(args_dict=pipeline_args)

# Inference
inference = pipeline(inputs='This is some text')
print(inference)
