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
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#aknowledgments">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

This project aims to help people use mt5 model for everything (easily).

### Built With

* [Python](https://www.python.org/)
* [PyCharm](https://www.jetbrains.com/pycharm/)



<!-- GETTING STARTED -->

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

There are no prerequisites other than those of requirements.txt.

### Installation

1. Clone the repo
    ```sh
    path/to/folder$ git clone https://github.com/Sunwaee/sunwaee-mt5-template.git
    ```
2. Install necessary packages
    ```sh
    path/to/folder/sunwaee-mt5-template$ pip install -r requirements.txt
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
> **example :** source_text -> "task: apply task to this text."  || target_text "output of the task on source text"

1. The first step is to convert .tsv files to data files using `databuilder.py`
    - Using a Python script :
        ```python
        import databuilder 
      
        databuilder_args = dict(
            source_max_length=512,  # Maximum length of source text
            target_max_length=30,  # Maximum length of target text
            tokenizer_name_or_path='google/mt5-small',  # Tokenizer path
            tokenizer_save_path='tokenizer',  # Tokenizer save path
            train_csv_path='data/train.tsv',  # Training file path
            valid_csv_path='data/valid.tsv',  # Validation file path
            source_column='source_text',  # Source column
            target_column='target_text',  # Target column
            train_file_path='data/train.pt',  # Training data save path
            valid_file_path='data/valid.pt',  # Validation data save path
            databuilder_config_path='model/config/config.json'  # Save path of databuilder config
        )
        
        # Running databuilder
        databuilder.run(args_dict=databuilder_args)
        ```
      as this is the default configuration, it is the same as:
        ```python
        import databuilder
        
        # Running databuilder
        databuilder.run()
        ```
    - Using command line :
        ```bash
        path/to/sunwaee-mt5-template$ databuilder.py --source_max_length 512 --target_max_length 30 --tokenizer_name_or_path google/mt5-small --tokenizer_save_path tokenizer --train_file_path data/train.tsv --valid_file_path data/valid.tsv --source_column source_text --target_column target_text --train_save_path data/train.pt --valid_save_path data/valid.pt --databuilder_config_path data/config/config.json  
        ```
      or:
        ```bash
        path/to/sunwaee-mt5-template$ databuilder.py
        ```

2. The next step is the training
   - Using a Python script:
        ```python
        import train
        
        train_args = dict(
            output_dir="model",
            do_train=True,
            do_eval=True,
            seed=42,
            learning_rate=1e-4,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            evaluation_strategy="epoch"
        )
        
        train.run(args_dict=train_args)
        ```
   - Using command line:
        ```shell
        path/to/sunwaee-mt5-template$ python train.py --output_dir model --do_train True --do_eval True --seed 42 --learning_rate 1e-4 --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --evaluation_strategy epoch
        ```
     > Note that you can add plenty of other arguments that fits Hugging Face [TrainingArguments](https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments).

3. The final step is the pipeline:
    - This time it's only available using Python Script:
        ```python
        import pipelines
        
        # Loading the pipeline
        pipeline = pipelines.run()
        
        inference = pipeline(inputs='This is some text')
        print(inference)
        ```



<!-- LICENSE -->

## License

Distributed under the MIT License. See `License` for more information.



<!-- CONTACT -->

## Contact

@Sunwaee - sunwaee.contact@gmail.com - [LinkedIn](https://www.linkedin.com/in/dvdnss/)

Project Link: [https://github.com/Sunwaee/sunwaee-mt5-template](https://github.com/Sunwaee/sunwaee-mt5-template)



<!-- AKNOWLEDGEMENTS -->

## Aknowledgements

- [HuggingFace](https://huggingface.co/) (no need to talk about it)
- [GitHub](https://github.com/) (sometimes we forget to mention it)
- [othneildrew/Best-README-Template](https://github.com/othneildrew/Best-README-Template) (should be set by default in any git)
- [patil-suraj/question_generation](https://github.com/patil-suraj/question_generation) (learnt fromt his repo so big thanks to him)