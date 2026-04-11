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

## Baseline Architecture

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

## Attention Upgrade

The file now also supports an optional attention variant enabled with
`--attention=1`.

That variant keeps the same encoder and decoder RNNs, but each decoder step
also:

- scores the current decoder hidden state against each encoder hidden state
- softmaxes those scores over source positions
- builds a context vector as a weighted sum of encoder states
- projects `[decoder_hidden, context]` to vocabulary logits

This keeps the plain RNN baseline available while giving us a clean
before/after comparison on the exact same synthetic reversal task.

## Loss And Metrics

Training should use cross-entropy over each decoder output position.

Useful metrics:

- token accuracy
- exact-sequence accuracy
- periodic checkpoint logging to CSV for baseline comparison

The current trainer logs:

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

This is meant to make the plain fixed-context encoder-decoder easier to train,
and it can also remain enabled for the attention variant so we compare the two
models under the same data schedule.

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

### attention run at 2000 steps

- config:
  `attention=1 steps=2000 len=8 cur_max=8`
- checkpoint metrics:
  step `2000`: `train_loss=2.080554 train_tok=0.219000 eval_tok=0.331000 eval_seq=0.000000`
- sample:
  `18955654 -> 8888000 (target 45655981)`
- notes:
  attention is learning something measurable at the token level, but sequence
  reversal is still far from solved at 2000 steps; exact-sequence accuracy
  remained at zero and the sample output still collapses toward repeated digits
  instead of a faithful reversed copy

## Scope Boundaries

The first version should stay intentionally small:

- no beam search
- no padding/masking system
- no real-language corpus
- no batching of variable-length mixed sequences unless it stays simple

## Follow-Up Path

Now that the first attention variant exists, the next upgrades are:

- gather longer baseline vs attention runs
- compare whether curriculum still helps once attention is enabled
- improve logging and sample inspection for qualitative comparison
