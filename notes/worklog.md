# Worklog

Line totals below come from `git log --numstat` and exclude binary file diffs, which Git reports as
`-/-`.

<!-- Update rule: "Hours Worked" should be sessionized. Split a day into separate sessions whenever
adjacent commits are more than 1 hour apart, then sum the first-to-last span within each session. -->

| Date       | Hours Worked | Lines Added | Lines Removed |
| ---------- | -----------: | ----------: | ------------: |
| 2026-02-21 |        0.98h |        2259 |            22 |
| 2026-04-10 |        2.42h |        1586 |           201 |

LOC - 4923

- tensor.c: 1489
- ngram.c: 614
- mnist.c: 611
- autoencoder.c: 559
- tensor_test.c: 477
- skipgram.c: 350
- llm.c: 346
- vocab.c: 333
- demo.c: 144

## 2026-04-10

- Added the MNIST demo, MNIST dataset setup flow, and `.gitignore` coverage for generated files and
  datasets.
- Refactored the MNIST code into a clearer model struct with forward/train/eval helpers and better
  inline architecture comments.
- Added CSV metric logging, elapsed-time reporting, a matplotlib-based loss plot script, and
  refreshed the example chart asset.
- Simplified the MNIST CLI to `--key=value` flags for core knobs like epochs, batch, channels,
  learning rate, momentum, and log path.
- Added MNIST notes, experiment history, shuffled training order, momentum SGD support, and compared
  baseline vs momentum runs.
- Ran and recorded a stronger `channels=32`, `epochs=10`, `momentum=0.9` experiment that reached
  about 45% test accuracy on the current subset/config.
- Reworked the repo layout around named demos like `skipgram.c`, `ngram.c`, and `autoencoder.c`,
  plus shared output paths under `out/`.
- Added a shared `vocab.c` / `vocab.h` helper and refactored both the skip-gram and n-gram demos to
  use the same corpus-building pipeline.
- Built a neural word-level n-gram demo with held-out eval loss, perplexity reporting, greedy text
  generation, notes, and cleaner model-owned forward scratch buffers.
- Moved row-wise one-hot encoding and row argmax helpers into the tensor library and added unit
  coverage for them in `tensor_test.c`.
- Added a minimal MNIST autoencoder demo, documented its architecture in `docs/autoencoder.md`, and
  added a reconstruction export/render flow for visual before-vs-after grids.
- Captured the current n-gram baseline in `notes/ngram.md`, added a broader model roadmap in
  `TODO.md`, and the current HEAD for this stretch is `db73058`.

## 2026-02-21

- Built the initial minimal C tensor library with autodiff and a small demo program.
- Added grad-mode handling, snapshot I/O, tensor copy support, and a broader set of tensor ops
  including broadcast reductions, softmax/cross-entropy, pow/sqrt, and layernorm.
- Added unit tests plus Makefile targets for building the static library and running tests.
- Wrote the initial README and usage examples.
- Added a character-level LLM demo and then a Shakespeare helper script/demo path on top of the core
  tensor library.
