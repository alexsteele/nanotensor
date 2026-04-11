# Seq2Seq

`seq2seq.c` will be a minimal encoder-decoder demo built on the core tensor
library.

## Goal

The first version should demonstrate sequence-to-sequence learning clearly,
without depending on a real text dataset.

The task will be synthetic:

- input: a short digit sequence
- output: the reversed digit sequence

Examples:

- `123 -> 321`
- `5087 -> 7805`
- `90012 -> 21009`

This gives us unlimited clean training data and a very obvious success
criterion.

## Vocabulary

The initial token set should stay tiny:

- digits `0-9`
- `BOS`
- `EOS`

That keeps the model and the decoding logic easy to inspect.

## V1 Architecture

The first model should be a plain recurrent encoder-decoder with no attention:

- token embedding lookup
- tanh RNN encoder over the input sequence
- final encoder hidden state becomes the decoder initial hidden state
- tanh RNN decoder with teacher forcing during training
- decoder output projection to vocabulary logits

At a high level:

- encoder: `token -> embed -> rnn hidden`
- bridge: `encoder_final_hidden -> decoder_hidden_0`
- decoder: `prev_token + hidden -> next hidden -> vocab logits`

## Loss And Metrics

Training should use cross-entropy over each decoder output position.

Useful metrics:

- token accuracy
- exact-sequence accuracy
- periodic checkpoint logging to CSV for baseline comparison

The current baseline should log:

- `step`
- `seq_len`
- `curriculum_max_len`
- `train_loss`
- `train_tok`
- `eval_tok`
- `eval_seq`

The trainer now uses a simple length curriculum:

- early steps sample shorter sequences
- middle steps widen the sampled length range
- final steps use the full configured `max_len`

This is meant to make the plain fixed-context encoder-decoder easier to train
before trying a larger architectural upgrade like attention.

## Experiment Log

Track one short entry per meaningful run keyed by git commit.

### `4e73919` curriculum baseline smoke run

- config:
  `steps=9 batch=4 embed=8 hidden=16 min_len=3 max_len=8 lr=0.03`
- curriculum:
  early checkpoint sampled with `curriculum_max_len=4`, final checkpoint with
  `curriculum_max_len=8`
- checkpoint metrics:
  step `1`: `train_loss=2.484628 train_tok=0.250000 eval_tok=0.168269 eval_seq=0.000000`
- checkpoint metrics:
  step `9`: `train_loss=2.464005 train_tok=0.125000 eval_tok=0.181818 eval_seq=0.000000`
- sample:
  `290 -> 0000 (target 092)`
- sample:
  `0513915 ->  (target 5193150)`
- notes:
  this is only a very short sanity-check run, so exact-sequence accuracy stayed
  at zero; the main value is confirming the current curriculum + logging
  baseline and output format

## Scope Boundaries

The first version should stay intentionally small:

- no attention yet
- no beam search
- no padding/masking system
- no real-language corpus
- no batching of variable-length mixed sequences unless it stays simple

## Follow-Up Path

Once the fixed-context encoder-decoder works, the next upgrade is:

- add attention over encoder states

That gives a clean before/after comparison on the same synthetic reversal task.
