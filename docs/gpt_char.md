# GPT Char

`gpt_char.c` is a minimal GPT-like character language model demo built around a
small stack of causal self-attention blocks.

The demo keeps the design intentionally small:

- raw character vocabulary derived directly from the input text
- learned token embeddings and learned positional embeddings
- explicit Q/K/V attention with a causal prefix mask
- configurable attention head count and transformer block depth
- residual connections plus a small feed-forward block
- next-character cross-entropy training with Adam

At a high level:

- input chars -> token embeddings + position embeddings
- repeat `blocks` times: layernorm -> Q/K/V projections -> split across `heads`
  attention heads -> causal attention over the visible context prefix ->
  residual add -> layernorm -> feed-forward MLP -> residual add
- vocab logits at every position

Build and run:

```bash
make gpt-char
./gpt_char \
  --text=data/shakespeare/shakespeare_gutenberg.txt \
  --steps=2000 \
  --context=32 \
  --dim=32 \
  --hidden=64 \
  --heads=4 \
  --blocks=2 \
  --lr=0.003 \
  --prompt="To be,"
```

`dim` must be divisible by `heads`. A small starting point like `--dim=32
--heads=4 --blocks=2` keeps each head width manageable while still exercising
both the multi-head path and a deeper residual stack.

Useful outputs:

- `out/gpt_char_training_log.csv`
- `out/gpt_char_report.txt`

The report writes a few fixed prompts plus generated continuations so the demo
has a human-friendly artifact in the same spirit as the other showcase models.
