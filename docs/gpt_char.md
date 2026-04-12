# GPT Char

`gpt_char.c` is a minimal GPT-like character language model demo built around a
single causal self-attention block.

The first version keeps the design intentionally small:

- raw character vocabulary derived directly from the input text
- learned token embeddings and learned positional embeddings
- explicit single-head Q/K/V attention with a causal prefix mask
- residual connections plus a small feed-forward block
- next-character cross-entropy training with Adam

At a high level:

- input chars -> token embeddings + position embeddings
- layernorm -> Q/K/V projections
- causal attention over earlier positions in the current context window
- residual add
- layernorm -> feed-forward MLP -> residual add
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
  --lr=0.003 \
  --prompt="To be,"
```

Useful outputs:

- `out/gpt_char_training_log.csv`
- `out/gpt_char_report.txt`

The report writes a few fixed prompts plus generated continuations so the demo
has a human-friendly artifact in the same spirit as the other showcase models.
