"""Create a Qwen3 model config and save it to model_configs/.

Edit ``kwargs`` below for the desired architecture, then run:
    uv run python scripts/create/create_config.py --output model_configs/<name>
"""

import argparse

import torch
import transformers
from transformers import Qwen3Config

kwargs = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 512,
    "initializer_range": 0.02,
    "intermediate_size": 1534,
    "max_position_embeddings": 1000,
    "max_window_layers": 64,
    "model_type": "qwen3",
    "num_attention_heads": 8,
    "num_hidden_layers": 64,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "dtype": torch.bfloat16,
    "transformers_version": transformers.__version__,
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 1972,
    "pad_token_id": 1969,
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="model_configs/qwen3_200M_512_64", help="Output directory")
args = parser.parse_args()

config = Qwen3Config(**kwargs)
config.save_pretrained(args.output)
print(f"Config saved to {args.output}")
