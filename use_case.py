# coding:utf-8
"""
Filename: use_case.py
Author: @DvdNss

Created on 11/15/2021
"""

from source import train, databuilder, pipelines

databuilder_args = dict(
   source_max_length=1024,  # Maximum length of source text
   target_max_length=100,  # Maximum length of target text
   tokenizer_name_or_path='google/mt5-base',  # Tokenizer path
   tokenizer_save_path='model/mlsum/tokenizer',  # Tokenizer save path
   train_csv_path='data/mlsum/fr/train.csv',  # Training file path
   valid_csv_path='data/mlsum/fr/validation.csv',  # Validation file path
   source_column='source_text',  # Source column
   target_column='target_text',  # Target column
   train_data_save_path='data/mlsum/fr/train.pt',  # Training data save path
   valid_data_save_path='data/mlsum/fr/valid.pt',  # Validation data save path
   databuilder_config_save_path='model/mlsum/config/config.json'  # Save path of databuilder config
)

# Run databuilder
databuilder.run(args_dict=databuilder_args)

# train_args = dict(
#     output_dir="model",  # output directory of model & tokenizer
#     model_config_save_path="model/config/config.json",  # output path of model config
#     wandb_project_name='mt5-project',  # wandb project name for training tracking
#     overwrite_output_dir=True,  # whether to overwrite output_dir or not
#     seed=42,  # seed to use
#     learning_rate=1e-3,  # learning rate
#     num_train_epochs=2,  # number of epochs
#     per_device_train_batch_size=1,  # train batch size
#     per_device_eval_batch_size=1,  # eval batch size
#     evaluation_strategy="epoch",  # evaluation strategy
# )
#
# # Starting training
# train.run(args_dict=train_args)
#
# pipeline_args = dict(
#     pipeline='classic',
#     pipeline_config_path='model/config/classic.json'
# )
#
# # Loading the pipeline
# pipeline = pipelines.run(args_dict=pipeline_args)
#
# # Inference
# inference = pipeline(inputs='Where are my grades ?')
# print(inference)
