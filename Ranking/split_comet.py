#!/usr/bin/env python3

"""
Use the COMET model to score all the INSTRUCTIONS and sort them by SCORE
===============================

optional arguments:
  -h, --help            Show this help message and exit.
  --batch_size BATCH_SIZE
                        (type: int, default: 16)
  --gpus GPUS           (type: int, default: 1)
  --disable_cache       Disables sentence embeddings caching. This makes inference
                        slower but saves memory. (default: False)
  --disable_length_batching
                        Disables length batching. This makes inference slower. 
                        (default: False)
"""
import itertools
import json
import logging
import os

import numpy as np
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr
from pytorch_lightning import seed_everything
from sacrebleu.utils import get_reference_files, get_source_file

from comet import download_model, load_from_checkpoint
import csv


def split_command() -> None:
    parser = ArgumentParser(description="Command for sort instruction tuning dataset by COMET score.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--input",
        help=(
            "Path to the directory where intruction dataset will be stored. "
            + "By default its saved in data/alpaca_data.json"
        ),
        default=None,
    )
    parser.add_argument(
        "--num_workers",
        help="Number of workers to use when loading data.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
        help="Disables sentence embeddings caching. This makes inference slower but saves memory.",
    )
    parser.add_argument(
        "--disable_length_batching",
        action="store_true",
        help="Disables length batching. This makes inference slower.",
    )
    cfg = parser.parse_args()

    seed_everything(1)
    
    # Constructing input format of COMET model 
    if cfg.input is not None:
        with open(cfg.input, "r") as f:
            origent_data = json.load(f) 
    else:
        with open("alpaca_data.json", "r") as f:
            origent_data = json.load(f) 
        
    model_path = 'lightning_logs/instruction_comet/checkpoints/epoch=3-step=508-val_kendall=0.331.ckpt'
    model = load_from_checkpoint(model_path)
    model.eval()

    if not cfg.disable_cache:
        model.set_embedding_cache()

    instruct_data = []
    response = []

    for item in origent_data:
        instruct_data.append("Instruction: "+ item["instruction"] + ' Input: ' + item["input"])
        response.append('Response: ' + item["output"])
    data = {"src": instruct_data, 'mt':response}

    # COMET model evaluates the SCORE of each instruction
    seg_scores = []
    new_data = []    

    sys_data = {k: v for k, v in data.items()}
    sys_data = [dict(zip(sys_data, t)) for t in zip(*sys_data.values())]
    new_data.append(np.array(sys_data))
    outputs = model.predict(
        samples=sys_data,
        batch_size=cfg.batch_size,
        gpus=cfg.gpus,
        accelerator="cpu" if cfg.gpus == 0 else "auto",
        num_workers=cfg.num_workers,
        length_batching=(not cfg.disable_length_batching),
    )
    seg_scores = outputs.scores

    i = 0
    for instruction_pair in origent_data:
        instruction_pair['score'] = seg_scores[i]
        i += 1
    
    sorted_data = sorted(origent_data, key=lambda x: x['score'], reverse=True)
    json.dump(sorted_data, open('Comet_ranking_result.json', 'w'))    



if __name__ == "__main__":
    split_command()
