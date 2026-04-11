# Overview

`ngram.c` is a small neural word-level n-gram language model built on top of
[tensor.c](/Users/alex/Code/nanotensor/tensor.c) and the shared vocabulary helper in
[vocab.c](/Users/alex/Code/nanotensor/vocab.c).

The current model does:

- tokenize a text corpus into lowercase word ids
- keep the top `N` vocabulary entries by frequency
- take a fixed ordered context of previous words
- project each context position through a shared embedding table and per-position weight matrix
- sum those projected context features, apply `tanh`, then predict the next word with softmax

This differs from [skipgram.c](/Users/alex/Code/nanotensor/skipgram.c), which trains embeddings from
single center-word / context-word pairs rather than modeling ordered multi-word next-word prediction.

# Knobs

The current CLI exposes:

- `--text=PATH`: source text corpus
- `--steps=N`: number of training updates
- `--batch=N`: minibatch size
- `--context=N`: number of previous words used to predict the next word
- `--lr=FLOAT`: learning rate
- `--embed=N`: embedding width
- `--hidden=N`: hidden layer width
- `--vocab=N`: retained vocabulary size
- `--snapshot=PATH`: tensor snapshot output path
- `--vocab-out=PATH`: exported vocabulary path

Defaults in [ngram.c](/Users/alex/Code/nanotensor/ngram.c) are:

- `steps=2000`
- `batch=64`
- `context=3`
- `lr=0.03`
- `embed=32`
- `hidden=64`
- `vocab=1000`

# Limitations

Current limitations to keep in mind:

- the context window is fixed-width and small
- generation is greedy, so repetition is common
- the eval split is a simple contiguous tail split of the encoded corpus
- we are training from scratch, not initializing from skip-gram embeddings
- the implementation currently builds one-hot context tensors, which is simple but not memory-efficient

# Experiment Log

Track one short entry per meaningful run keyed by git commit.

### `5212a56` default baseline

- config: assumed default `ngram` settings
  `steps=2000 batch=64 context=3 lr=0.03 embed=32 hidden=64 vocab=1000`
- final train loss: `5.563024`
- final eval loss: `5.887973`
- final perplexity: `360.67`
- prediction sample:
  `to be or -> and(0.04) the(0.03) i(0.03) to(0.03) a(0.02)`
- generation sample:
  `sample: to be or and and and and and and and and and and and and`
- notes: model collapsed toward a few high-frequency words and repetitive generation; this is a
  useful baseline showing the current setup is training, but not yet producing good conditional
  predictions

# Next Ideas

- try a larger hidden size and longer training run
- compare `context=3` against `context=4` or `5`
- add temperature or sampling instead of pure greedy decoding
- test whether skip-gram embedding initialization improves early perplexity
