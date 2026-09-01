# Model Configurations

Pre-built HuggingFace-compatible `config.json` files for the Qwen3 architecture
variants used in the Plynder experiments. Each subdirectory contains a
`config.json` ready for `AutoConfig.from_pretrained()`.

The model name encodes the architecture: `qwen3_<params>_<hidden>_<layers>`.

| Config | Hidden | Layers | ~Params | Architecture | Notes |
|--------|--------|--------|---------|--------------|-------|
| `qwen3_50M_512_16` | 512 | 16 | 50 M | Qwen3ForCausalLM | **Released model**, trained with CISPO |

All configs share a custom vocabulary of **1972 tokens** (1968 chess moves +
4 special tokens: BOS, DRAW, WHITE_WIN, BLACK_WIN) and
`max_position_embeddings = 8000`.

## Usage

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("model_configs/qwen3_32M_512_8")
```

Or via Hydra config (`configs/config.yaml`):

```yaml
paths:
  model_config: "model_configs/qwen3_50M_512_16"
```

## Notes

- `pad_token_id` is set to `1969` (the `DRAW` token). Any token ID would do.
  Padding positions are excluded by the attention mask, but the collate
  function reads this value, so keep it consistent with the vocabulary.
The special tokens are the only non-move tokens in the vocabulary.
