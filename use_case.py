# coding:utf-8
"""
Filename: use_case.py
Author: @DvdNss

Created on 11/15/2021
"""

from source import train, databuilder, pipelines

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
    learning_rate=1e-3,  # learning rate
    num_train_epochs=2,  # number of epochs
    per_device_train_batch_size=1,  # train batch size
    per_device_eval_batch_size=1,  # eval batch size
    evaluation_strategy="epoch",  # evaluation strategy
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
inference = pipeline(inputs='Where are my grades ?')
print(inference)
