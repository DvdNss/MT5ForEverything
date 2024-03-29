<!-- PROJECT LOGO -->
<h3 align="center">MT5ForEverything by @DvdNss</h3>
<p align="center">
Making Google MT5 Transformers models fast and easy to use.
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

This project aims to help people use [Google/MT5](https://huggingface.co/models?search=google%2Fmt5-) model for
everything (easily).

### Built With

* [Python 3.9](https://www.python.org/)

<!-- GETTING STARTED -->

### Installation

1. Clone the repo
    ```sh
    path/to/folder$ git clone https://github.com/DvdNss/MT5ForEverything.git
    ```
2. Install necessary packages
    ```sh
    path/to/repo$ pip install -r requirements.txt
    ```

<!-- USAGE EXAMPLES -->

## Usage

> To build training and evaluation data, we use the `databuilder` class. It
> takes two **.tsv** files and gives you two **.pt** files which will be used for
> training and validation.
>
> **Input:**
> - training and validation **csv** files with source and target columns (**csv** format is **.csv** or **.tsv**
    with **"\t"** as separator). Source cells should be in the **"task: text"** format.
> - databuilder arguments of your choice (see in python script below)
>
> **example :** source_text -> "task: apply task to this text."  || target_text -> "output of the task on source text"

1. The first step is to convert **.tsv** files to data files using `databuilder.py`
   - Using a Python script :
        ```python
        from source import databuilder 
        
        databuilder_args = dict(
           source_max_length=512,  # Maximum length of source text
           target_max_length=30,  # Maximum length of target text
           tokenizer_name_or_path='google/mt5-small',  # Tokenizer path
           tokenizer_save_path='tokenizer',  # Tokenizer save path
           train_csv_path='data/train.tsv',  # Training file path
           valid_csv_path='data/valid.tsv',  # Validation file path
           source_column='source_text',  # Source column
           target_column='target_text',  # Target column
           train_data_save_path='data/train.pt',  # Training data save path
           valid_data_save_path='data/valid.pt',  # Validation data save path
           databuilder_config_save_path='model/config/config.json'  # Save path of databuilder config
        )
        
        # Running databuilder
        databuilder.run(args_dict=databuilder_args)
        ```
     
      as this is the default configuration, it is the same as:
        ```python
        from source import databuilder
        
        # Running databuilder
        databuilder.run()
        ```
   - Using command line :
       ```bash
       path/to/repo$ python source/databuilder.py --source_max_length 512 --target_max_length 30 --tokenizer_name_or_path google/mt5-small --tokenizer_save_path tokenizer --train_csv_path data/train.tsv --valid_csv_path data/valid.tsv --source_column source_text --target_column target_text --train_data_save_path data/train.pt --valid_data_save_path data/valid.pt --databuilder_config_save_path data/config/config.json  
       ```
     or:
       ```bash
       path/to/repo$ python source/databuilder.py
       ```

2. The next step is the training
   - Using a Python script:
        ```python
        from source import train
        
        train_args = dict(
           output_dir="model",  # output directory of model & tokenizer
           model_config_save_path="model/config/config.json",  # output path of model config
           wandb_project_name='mt5-project',  # wandb project name for training tracking
           overwrite_output_dir=True,  # whether to overwrite output_dir or not
           do_train=True,  # whether to train or not
           do_eval=True,  # whether to evaluate or not
           seed=42,  # seed to use
           learning_rate=1e-4,  # learning rate
           num_train_epochs=1,  # number of epochs
           per_device_train_batch_size=1,  # train batch size
           per_device_eval_batch_size=1,  # eval batch size
           evaluation_strategy="epoch"  # evaluation strategy
        )
        
        # Starting training
        train.run(args_dict=train_args)
        ```
     > Note that you can add plenty of other arguments that fits Hugging Face [TrainingArguments](https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments).
   - Using command line:
        ```shell
        path/to/repo$ python source/train.py --output_dir model --seed 42 --learning_rate 1e-4 --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --evaluation_strategy epoch
        ```

3. The final step is the inference
   - This time, it's only available using Python Script:
        ```python
        from source import pipelines
        
        pipeline_args = dict(
           pipeline='classic',
           pipeline_config_path='model/config/classic.json'
        )
        
        # Loading the pipeline
        pipeline = pipelines.run(args_dict=pipeline_args)
        
        # Inference
        inference = pipeline(inputs='This is some text')
        print(inference)
        ```

<!-- LICENSE -->

## License

Distributed under the Apache-2.0 License. See `License` for more information.

<!-- CONTACT -->

## Contact

@DvdNss - private.david.naisse@gmail.com - [LinkedIn](https://www.linkedin.com/in/dvdnss/)