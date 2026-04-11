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
  with a small learned additive scorer
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

Evaluation now runs on a fixed synthetic holdout set generated once at startup.
That makes `eval_tok` and `eval_seq` much easier to compare across checkpoints
and across separate runs.

The demo now also supports optimizer selection via `--opt=momentum|adam`.
Momentum remains the simple baseline, while Adam gives us a second training
dynamic to compare on the same fixed eval set.

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
- result:
  `step=9 train_loss=2.464005 train_tok=0.125000 eval_tok=0.181818 eval_seq=0.000000`
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
- result:
  `step=2000 train_loss=2.080554 train_tok=0.219000 eval_tok=0.331000 eval_seq=0.000000`
- sample:
  `18955654 -> 8888000 (target 45655981)`
- notes:
  attention is learning something measurable at the token level, but sequence
  reversal is still far from solved at 2000 steps; exact-sequence accuracy
  remained at zero and the sample output still collapses toward repeated digits
  instead of a faithful reversed copy

### attention run near 10000 steps

- config:
  `attention=1 steps=10000 max_len=8`
- result:
  `step=10000 train_loss=1.021648 train_tok=0.656000 eval_tok=0.560000 eval_seq=0.004000`
- sample:
  `095 -> 590 (target 590)`
- sample:
  `0215168 -> 2226120 (target 8615120)`
- sample:
  `6685 -> 6666 (target 5866)`
- notes:
  by 10K steps the attention model is clearly learning partial reversal and
  often gets prefixes or suffixes right, with eval token accuracy around
  `0.55-0.62`. Exact-sequence accuracy is still low and noisy, which suggests
  the model is not yet consistently binding each output position to the correct
  source position, especially on longer sequences and repeated digits.

### matched optimizer comparison at 2000 steps

- config:
  `attention=1 steps=2000 batch=32 embed=16 hidden=32 min_len=3 max_len=8`
- momentum result:
  `opt=momentum lr=0.03 step=2000 train_loss=2.080554 train_tok=0.219000 eval_tok=0.355000 eval_seq=0.000000`
- Adam result:
  `opt=adam lr=0.003 step=2000 train_loss=1.798076 train_tok=0.399000 eval_tok=0.487000 eval_seq=0.016000`
- notes:
  on the same fixed synthetic eval set, Adam is clearly ahead by 2K steps.
  The gap shows up in both token accuracy and exact-sequence accuracy, so this
  looks like a real optimization win rather than just noisy checkpoint drift.

### attention + Adam run at 10000 steps

- config:
  `attention=1 opt=adam steps=10000 batch=32 embed=16 hidden=32 min_len=3 max_len=8 lr=0.003`
- result:
  `step=10000 train_loss=0.975108 train_tok=0.650000 eval_tok=0.570000 eval_seq=0.027000`
- sample:
  `095 -> 890 (target 590)`
- sample:
  `0215168 -> 1022120 (target 8615120)`
- sample:
  `6685 -> 6866 (target 5866)`
- notes:
  Adam stays clearly ahead of the 2K momentum baseline and reaches sequence
  accuracy that is competitive with the earlier 10K attention run. The model
  is still making alignment mistakes, especially on longer sequences and
  repeated digits, but the optimizer change gives us a much stronger training
  trajectory without changing the architecture.

### learned additive attention + Adam run at 10000 steps

- config:
  `attention=1 opt=adam steps=10000 batch=32 embed=16 hidden=32 min_len=3 max_len=8 lr=0.003`
- result:
  `step=10000 train_loss=1.700396 train_tok=0.347000 eval_tok=0.480000 eval_seq=0.051000`
- sample:
  `35125752 -> 25553533 (target 25752153)`
- notes:
  this learned additive scorer improved exact-sequence accuracy relative to the
  earlier 10K dot-attention run, but it regressed token accuracy sharply and
  still produced visibly unstable long-sequence outputs. As currently
  parameterized, it does not look like a clear upgrade over the simpler dot
  attention baseline.

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
