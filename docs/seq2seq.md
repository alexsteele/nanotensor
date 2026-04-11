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
- `train_loss`
- `train_tok`
- `eval_tok`
- `eval_seq`

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
