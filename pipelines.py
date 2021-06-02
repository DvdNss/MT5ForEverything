# coding:utf-8
"""
Filename : pipelines.py
Role : TO-DO: Change role of pipelines.py

@author : Sunwaee
"""

import itertools
import logging
import torch
import json

from tqdm import tqdm
from typing import Optional, Dict, Union
from dataclasses import dataclass, field
from utils import dict_to_json
from databuilder import DatabuilderArguments
from train import ModelArguments

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    HfArgumentParser,
    TrainingArguments,
)

# Tracking modules & packages hierarchy
logger = logging.getLogger(__name__)

DEFAULT_ARGS = dict(
    pipeline='classic',
    cfg_path='model/params.json'
)


@dataclass
class PipelineArguments:
    """
    Pipeline arguments.
    """

    pipeline: Optional[str] = field(default=DEFAULT_ARGS['pipeline'],
                                    metadata={"help": "Pipeline to use in pipelines. "})


class ClassicPipeline:
    """
    Classic Pipeline
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Initializes ClassicPipeline class.

        :param model: pretrained model used for inference
        :param tokenizer: pretrained tokenizer used for inference
        :param use_cuda: whether to use GPU or not
        """

        # Initializing class attributes
        self.model = model
        self.tokenizer = tokenizer

        # Using GPU if CUDA available, otherwise using CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Sending model to GPU
        self.model.to(self.device)

        # Checking model is MT5 model
        assert self.model.__class__.__name__ in ["MT5ForConditionalGeneration"], 'Model should be MT5 model. '

    def __call__(self, inputs: str, *args, **kwargs):
        """
        Calls ClassicPipeline.

        :param inputs: inputs that must go through the pipeline
        :param args: other args
        :param kwargs: other kwargs
        :return: pipeline output
        """

        # TO-DO: Fill the call function with pipeline processes

        return self._tokenize(inputs=[inputs])

    def _tokenize(self, inputs, padding=True, truncation=True, add_special_tokens=True, max_length=512):
        """
        Tokenizes given input.

        :param inputs: text to tokenize
        :param padding: whether to apply padding or not
        :param truncation: whether to apply truncation or not
        :param add_special_tokens: whether to add special tokens or not
        :param max_length: input max length (tokens)
        :return: tokenized input
        """

        # TO-DO: Make the params callable from global dict

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

    # TO-DO: Function to prepare input for Requesting Pipeline


# Supported pipelines
PIPELINES = {
    "classic": {
        "impl": ClassicPipeline,
        "default": {
            "model": "model",
        }
    }
}


def main(from_json: bool = True, filename: str = DEFAULT_ARGS['cfg_path']):
    """
    Calls the specified pipeline.

    :param filename:
    :param from_json:
    :return: pipeline result
    """

    # Parsing arguments
    parser = HfArgumentParser((ModelArguments, DatabuilderArguments, TrainingArguments, PipelineArguments))
    model_args, databuilder_args, training_args, pipeline_args = parser.parse_json_file(
        json_file=filename) if from_json else parser.parse_args_into_dataclasses()

    assert pipeline_args.pipeline in PIPELINES, "Unknown pipeline {}, available pipelines are {}".format(pl, list(
        PIPELINES.keys()))

    model = AutoModelForSeq2SeqLM.from_pretrained(databuilder_args.global_output_dir)
    tokenizer = AutoTokenizer.from_pretrained(databuilder_args.global_output_dir)

    task_pipeline = PIPELINES[pipeline_args.pipeline]["impl"]
    return task_pipeline(model=model, tokenizer=tokenizer)


def run(args_dict: dict = {}) -> ClassicPipeline:
    args_dict = {**DEFAULT_ARGS, **args_dict}
    with open(args_dict['cfg_path'], "r") as config:
        args_dict = {**json.load(fp=config), **args_dict}
    file = dict_to_json(args_dict=args_dict, filename=args_dict['cfg_path'])
    pipeline = main(filename=file)
    return pipeline


if __name__ == "__main__":
    main(from_json=False)
