# Worklog

Line totals below come from `git log --numstat` and exclude binary file diffs, which Git reports as
`-/-`.

| Date       | Hours Worked | Lines Added | Lines Removed |
| ---------- | -----------: | ----------: | ------------: |
| 2026-02-21 |        0.98h |        2259 |            22 |
| 2026-04-10 |        0.92h |        1586 |           201 |

LOC - 2985

- tensor.c: 1452
- mnist.c: 599
- tensor_test.c: 455
- llm.c: 346
- demo.c: 133

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

## 2026-02-21

- Built the initial minimal C tensor library with autodiff and a small demo program.
- Added grad-mode handling, snapshot I/O, tensor copy support, and a broader set of tensor ops
  including broadcast reductions, softmax/cross-entropy, pow/sqrt, and layernorm.
- Added unit tests plus Makefile targets for building the static library and running tests.
- Wrote the initial README and usage examples.
- Added a character-level LLM demo and then a Shakespeare helper script/demo path on top of the core
  tensor library.
