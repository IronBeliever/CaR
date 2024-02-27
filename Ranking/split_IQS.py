#!/usr/bin/env python3


"""
Use the IQS model to score all the INSTRUCTIONS and sort them by SCORE
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
import json
import os

import numpy as np
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr
from pytorch_lightning import seed_everything

from comet import download_model, load_from_checkpoint
import csv


def split_command() -> None:
    parser = ArgumentParser(description="Command for sort instruction tuning dataset by IQS score.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--input",
        help=(
            "Path to the directory where intruction dataset will be stored. "
            + "By default its saved in ../data/alpaca_data.json"
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

    # Constructing input format of IQS model  
    if cfg.input is not None:
        with open(cfg.input, "r") as f:
            origent_data = json.load(f) 
    else:
        with open("./data/alpaca_data.json", "r") as f:
            origent_data = json.load(f) 

    # load IQS model    
    model_path = 'Ranking/lightning_logs/instruction_score/checkpoints/epoch=3-step=1016-val_kendall=0.600.ckpt'
    model = load_from_checkpoint(model_path)
    model.eval()

    if not cfg.disable_cache:
        model.set_embedding_cache()

    instruct_data = []
    for item in origent_data:
        instruct_data.append("Instruction: "+ item["instruction"] + ' Input: ' + item["input"] + ' Response: ' + item["output"])
    data = {"src": instruct_data}

    # IQS model evaluates the SCORE of each instruction
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
    json.dump(sorted_data, open('./data/ranking_IQS_result.json', 'w'))    


if __name__ == "__main__":
    split_command()
