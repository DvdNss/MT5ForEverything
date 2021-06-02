<!-- PROJECT LOGO -->
<h3 align="center">MT5 for everything by Sunwaee</h3>
<p align="center">
This project aims to let anybody use Google MT5 models for traning/inference.
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

- [ ] About The Project
- [ ] Getting Started
- [ ] Usage
- [ ] Contact

<!-- ABOUT THE PROJECT -->

## About The Project

Here is the project description.

### Built With

* [Hugging Face](https://huggingface.co/)

<!-- GETTING STARTED -->

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

Here are the prerequisites.

### Installation

1. Clone the repo
    ```sh
    git clone https://mydavi@dev.azure.com/mydavi/BU%20Recherche/_git/Forthem
    ```
2. Install necessary packages
    ```sh
    pip install -r requirements.txt
    ```

<!-- USAGE EXAMPLES -->

## Usage

> To build training and evaluation data, we use the _Databuilder_ class. It
> takes two **csv** files and gives you two **pt** files which will be used for
> training and validation.
>
> **Input:**
> - training and validation **csv** files with source and target columns (**csv** format is **.csv** or **.tsv**
    with **"\t"** as separator). Source cells should be in the **"task: text"** format.
> - databuilder arguments of your choice (see in python script below)
>
> **example :** source_text -> "paraphrasing: paraphrase this."  || target_text "paraphrase that."

1. The first step is to convert .tsv files to data files
    - Using a Python script :
        ```python
        import databuilder 
      
        databuilder_args = dict(
            source_max_length=512,  # Maximum length of source text
            target_max_length=30,  # Maximum length of target text
            highlight_token='<hl>',  # Highlight token
            separation_token='<sep>',  # Separation token
            tokenizer_name_or_path='google/mt5-small',  # Tokenizer path
            tokenizer_save_path='tokenizer',  # Tokenizer save path
            train_csv_path='data/train.tsv',  # Training file path
            valid_csv_path='data/valid.tsv',  # Validation file path
            source_column='source_text',  # Source column
            target_column='target_text',  # Target column
            train_file_path='data/train.pt',  # Training data save path
            valid_file_path='data/valid.pt',  # Validation data save path
            databuilder_config_path='data/config.json'  # Save path of databuilder config
        )
        
        # Running databuilder
        databuilder.run(args_dict=databuilder_args)
        ```
      as this is the default configuration, it is the same as:
        ```python
        import databuilder
        
        databuilder_args = dict()
        
        # Running databuilder
        databuilder.run()
        ```
    - Using command line :
        ```bash
        python databuilder.py --source_max_length 512 \
            --target_max_length 30 \
            --highlight_token <hl> \
            --separation_token <sep> \
            --tokenizer_name_or_path google/mt5-small \
            --tokenizer_save_path tokenizer \
            --train_file_path data/train.tsv \
            --valid_file_path data/valid.tsv \
            --source_column source_text \
            --target_column target_text \
            --train_save_path data/train.pt \
            --valid_save_path data/valid.pt \
            --config_save_path data/config.json  
        ```
      or:
        ```bash
        cd system
        python databuilder.py
        ```

2. The next step is the training
   - Using a Python script:
       ```python
       import train
      
       model_args = dict(
           output_dir="model",
           do_train=True,
           do_eval=True,
           seed=42,
           learning_rate=1e-4,
           num_train_epochs=1,
           per_device_train_batch_size=1,
           per_device_eval_batch_size=1,
           evaluate_during_training=True
       )
       
       train.run(args_dict=model_args)
       ```
   - Using command line:
        ```shell
        cd system
        python train.py --output_dir model --do_train True --do_eval True --seed 42 --learning_rate 1e-4 --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --evaluation_strategy epoch
        ```
     > Note that you can add plenty of other arguments that fits Hugging Face [TrainingArguments](https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments).

3.

<!-- CONTACT -->

## Contact

David NAISSE - [LinkedIn](https://www.linkedin.com/in/dvdnss) - david_naisse@etu.u-bourgogne.fr