"""Create and save the Plynder chess tokenizer (word-level, UCI moves as tokens).

The tokenizer is not needed at runtime (rollout skips tokenizer init and
training works on raw token IDs) but is required to publish the model on the
Hub and is handy for debugging.
"""

import argparse
import os

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from plynder.core import TOKEN_ID_TO_UCI

# flip: id -> string → string -> id
custom_vocab = {v: k for k, v in TOKEN_ID_TO_UCI.items()}


custom_vocab["<bos>"] = 1968
custom_vocab["<draw>"] = 1969
custom_vocab["<white_win>"] = 1970
custom_vocab["<black_win>"] = 1971

custom_vocab["<unk>"] = max(custom_vocab.values()) + 1

# build tokenizer
tokenizer = Tokenizer(models.WordLevel(vocab=custom_vocab, unk_token="<unk>"))

# important: split on whitespace
tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)


# wrap into HF fast tokenizer
hf_tok = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="<unk>",
)

# save
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="models/plynder_tokenizer", help="Output directory")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)
hf_tok.save_pretrained(args.output)

print(f"Tokenizer saved to {args.output}")
