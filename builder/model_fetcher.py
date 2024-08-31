'''
RunPod | serverless-ckpt-template | model_fetcher.py

Downloads the model from the URL passed in.
'''

import shutil
import argparse
from pathlib import Path
from urllib.parse import urlparse
import torch
from diffusers import FluxPipeline

MODEL_CACHE_DIR = "diffusers-cache"


def download_model():
    '''
    Downloads the model from the URL passed in.
    '''
    model_cache_path = Path(MODEL_CACHE_DIR)
    if model_cache_path.exists():
        shutil.rmtree(model_cache_path)
    model_cache_path.mkdir(parents=True, exist_ok=True)

    model_id = "black-forest-labs/FLUX.1-schnell"

    pipe = FluxPipeline.from_pretrained(
        model_id,
        cache_dir=model_cache_path,
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload() 


# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    download_model()