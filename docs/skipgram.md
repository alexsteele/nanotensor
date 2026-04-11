# SkipGram

`skipgram.c` is a minimal word2vec-style skip-gram embedding demo.

It tokenizes a text corpus into lowercase words, trains a center-word to
context-word predictor with a softmax loss, prints nearest-neighbor words from
the learned embedding table, and saves a snapshot plus retained vocab.

Build and run:

```bash
make skipgram
./skipgram \
  --text=data/shakespeare/shakespeare_gutenberg.txt \
  --steps=2000 \
  --batch=64 \
  --window=2 \
  --lr=0.05 \
  --embed=32 \
  --vocab=800 \
  --snapshot=out/skipgram_snapshot.bin \
  --vocab-out=out/skipgram_vocab.txt \
  --report=out/skipgram_neighbors.txt
```

The human-friendly nearest-neighbor artifact used in `examples/` is written to
`out/skipgram_neighbors.txt`.
