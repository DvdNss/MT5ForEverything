# coding:utf-8
"""
Filename : pipelines.py
Role : model pipelines

@author : Sunwaee
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    HfArgumentParser,
    TrainingArguments,
)

from databuilder import DatabuilderArguments
from train import (
    ModelArguments,
    DEFAULT_ARGS as model_config
)
from utils import dict_to_json

# Tracking modules & packages hierarchy
logger = logging.getLogger(__name__)

DEFAULT_ARGS = dict(
    pipeline='requesting',
    pipeline_config_save_path='model/config/config.json'
)


@dataclass
class PipelineArguments:
    """
    Pipeline Arguments.
    """

    pipeline: Optional[str] = field(default=DEFAULT_ARGS['pipeline'],
                                    metadata={"help": "Pipeline to use. "})

    pipeline_config_save_path: Optional[str] = field(default=DEFAULT_ARGS['pipeline_config_save_path'],
                                                     metadata={'help': 'Model config path'})


class Pipeline:
    """
    Pipeline class.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, **kwargs):
        """
        Initializes Pipeline class.

        :param model: pretrained model used for inference
        :param tokenizer: pretrained tokenizer used for inference
        """

        # Initializing class attributes
        self.model = model
        self.tokenizer = tokenizer

        # Using GPU if CUDA available, otherwise using CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Sending model to GPU
        self.model.to(self.device)

        # Checking model is MT5 model
        assert self.model.__class__.__name__ in ["MT5ForConditionalGeneration"]

    def _tokenize(self, inputs: List[str], padding: bool = True, truncation: bool = True,
                  add_special_tokens: bool = True,
                  max_length: int = 512):
        """
        Tokenizes given input.

        :param inputs: text to tokenize
        :param padding: whether to apply padding or not
        :param truncation: whether to apply truncation or not
        :param add_special_tokens: whether to add special tokens or not
        :param max_length: input max length (tokens)
        :return: tokenized input
        """

        # Tokenizing input
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )

        # Returning tokenized input
        return inputs


class ClassicPipeline(Pipeline):
    """
    Classic Pipeline
    """

    def __init__(self, **kwargs):
        """
        Initializes RequestingPipeline class.
        """

        super().__init__(**kwargs)

    def __call__(self, inputs: str, *args, **kwargs):
        """
        Calls ClassicPipeline.

        :param inputs: inputs that must go through the pipeline
        :param args: other args
        :param kwargs: other kwargs
        :return: pipeline output
        """

        # Removing uneeded space characters
        inputs = " ".join(inputs.split())

        # Readying input for task
        inputs = self._ready_for_task(inputs)

        # Encoding inputs using tokenizer
        encoded_inputs = self._tokenize(inputs=[inputs])

        # Generating encoded outputs with model
        encoded_outputs = self.model.generate(input_ids=encoded_inputs['input_ids'].to(self.device),
                                              attention_mask=encoded_inputs['attention_mask'].to(self.device))

        # Decoding outputs
        outputs = self.tokenizer.decode(encoded_outputs[0], skip_special_tokens=True)

        return outputs

    @staticmethod
    def _ready_for_task(text: str) -> str:
        return f"task: {text} </s>"


# Supported pipelines
PIPELINES = {
    "classic": {
        "impl": ClassicPipeline,
        "default": {
            "model": "model",
        }
    }
}


def main(from_json: bool = True, filename: str = DEFAULT_ARGS['pipeline_config_save_path']):
    """
    Calls the specified pipeline.

    :param filename: json filename
    :param from_json: whether to run pipeline from json file or not
    :return: pipeline call function
    """

    # Parsing arguments
    parser = HfArgumentParser((ModelArguments, DatabuilderArguments, TrainingArguments, PipelineArguments))
    model_args, databuilder_args, training_args, pipeline_args = parser.parse_json_file(
        json_file=filename) if from_json else parser.parse_args_into_dataclasses()

    # Asserting specified pipeline does exist
    assert pipeline_args.pipeline in PIPELINES, \
        "Unknown pipeline {}, available pipelines are {}".format(pipeline_args.pipeline, list(PIPELINES.keys()))

    # Loading model & tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(training_args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)

    # Getting specified pipeline
    task_pipeline = PIPELINES[pipeline_args.pipeline]["impl"]

    return task_pipeline(model=model, tokenizer=tokenizer)


def run(args_dict: dict = {}, model_config_path: str = model_config['model_config_save_path']):
    """
    Run pipeline from dict.

    :param args_dict: pipeline arguments dict
    :param model_config_path: json path to model arguments
    """

    # Asserting config paths exist
    assert os.path.isfile(model_config_path), \
        f"Invalid filename for {model_config_path}, file doesn't exist. "

    # Opening databuilder config path and merging it with pipeline dict
    with open(model_config_path, "r") as cfg:
        args_dict = {**json.load(fp=cfg), **DEFAULT_ARGS, **args_dict}

    # Writing file to json
    file = dict_to_json(args_dict=args_dict, filename=args_dict['pipeline_config_save_path'])

    # Calling pipeline using json as source
    pipeline = main(filename=file)

    return pipeline


if __name__ == "__main__":
    main(from_json=False)
