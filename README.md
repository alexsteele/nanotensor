# nanotensor

A minimal tensor/autodiff library in C for small machine-learning experiments.

This project was developed with ChatGPT Codex.

## Features

- 1-2D tensors
- Autograd, SGD
- Serialization

Our goal is to support some common model types: multi-layer neural nets, mnist convnets, bigram
language models.

## Build

```bash
make static      # builds libtensor.a
make             # builds demo linked against libtensor.a
make test        # builds and runs unit tests
make run         # runs the demo training program
make rebuild     # forces a clean rebuild of all binaries for this machine
make mnist-data  # downloads and unpacks raw MNIST IDX files into data/mnist
make ngram       # builds the neural word-level n-gram demo
make seq2seq     # builds the seq2seq reversal demo
make autoencoder # builds the MNIST autoencoder demo
make examples    # regenerates the demo artifacts under examples/
```

## Example Artifacts

The repo keeps a few human-friendly demo outputs in `examples/`, including the
autoencoder reconstruction panel, seq2seq qualitative predictions, skip-gram
nearest neighbors, the n-gram text report, and the MNIST loss plot.

To regenerate them end-to-end:

```bash
make examples
```

That target builds the needed binaries and then runs
[`scripts/run_examples.sh`](/Users/alex/Code/nanotensor/scripts/run_examples.sh)
to refresh the checked-in example artifacts from fresh outputs in `out/`.

## Example

```c
#include "tensor.h"
#include <stdio.h>

int main(void) {
    float xv[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float wv[] = {0.5f, -1.0f};

    Tensor *x = tensor_from_array(2, 2, xv, 0);
    Tensor *w = tensor_from_array(2, 1, wv, 1);

    Tensor *y = tensor_matmul(x, w);
    tensor_print(y, "y", 0);

    /* Target and MSE loss */
    float tv[] = {0.0f, 1.0f};
    Tensor *t = tensor_from_array(2, 1, tv, 0);
    Tensor *loss = tensor_mse_loss(y, t);

    tensor_backward(loss);

    Tensor *params[] = {w};
    tensor_sgd_step(params, 1, 0.01f);

    printf("loss: %.6f\n", loss->data[0]);

    tensor_free(x);
    tensor_free(w);
    tensor_free(y);
    tensor_free(t);
    tensor_free(loss);
    return 0;
}
```

Compile manually against the static library:

```bash
cc -O2 -std=c11 -Wall -Wextra -pedantic your_program.c libtensor.a -lm -o your_program
```

## Demo

`demo.c` trains a small 2-layer network on synthetic data and prints training loss.

## LLM Demo

`llm.c` is a minimal character-level language model demo. It trains a small 2-layer MLP to predict
the next character from a text corpus and then generates text from a prompt.

Build and run:

```bash
make llm
./llm shakespeare.txt 3000 "To be" 64 300 0.2 0.9 96
```

To automatically download and prepare a multi-play Shakespeare corpus from Gutenberg and run the
demo:

```bash
make run-shakespeare
```

Or run the script directly with optional args:

```bash
./scripts/run_shakespeare_llm.sh [steps] [prompt] [batch] [gen_len] [lr] [temperature] [hidden]
```

## SkipGram Demo

`skipgram.c` is a minimal word2vec-style skip-gram word embedding demo:

- tokenizes a text corpus into lowercase words
- trains a center-word to context-word predictor with a softmax loss
- prints nearest-neighbor words from the learned embedding table

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
  --vocab-out=out/skipgram_vocab.txt

...
--- nearest neighbors ---
king: she(0.97) by(0.97) upon(0.97) now(0.96) give(0.96)
queen: may(0.85) she(0.84) mine(0.84) these(0.83) than(0.83)
love: thee(0.97) may(0.97) from(0.97) this(0.97) go(0.97)
death: d(0.95) very(0.95) so(0.94) had(0.94) must(0.94)
man: time(0.96) had(0.96) s(0.95) hamlet(0.95) like(0.95)
woman: now(0.93) were(0.93) they(0.92) at(0.92) which(0.91)

saved snapshot: out/skipgram_snapshot.bin
saved vocab: out/skipgram_vocab.txt
```

## NGram Demo

`ngram.c` is a small neural word-level n-gram language model demo:

- builds a retained vocabulary from a text corpus via `vocab.c`
- uses the previous `N` words as fixed context
- learns shared embeddings plus a tanh hidden layer
- predicts the next word with softmax and cross-entropy
- reports held-out eval loss / perplexity and prints a short greedy sample

Build and run:

```bash
make ngram
./ngram \
  --text=data/shakespeare/shakespeare_gutenberg.txt \
  --steps=2000 \
  --batch=64 \
  --context=3 \
  --lr=0.03 \
  --embed=32 \
  --hidden=64 \
  --vocab=1000 \
  --snapshot=out/ngram_snapshot.bin \
  --vocab-out=out/ngram_vocab.txt
```

Named args: `--text=PATH --steps=N --batch=N --context=N --lr=FLOAT --embed=N --hidden=N --vocab=N --snapshot=PATH --vocab-out=PATH`

The demo prints periodic training loss, held-out eval loss / perplexity when enough eval tokens are
available, shows a few next-word predictions plus a short greedy sample after training, and saves:

- `out/ngram_snapshot.bin`
- `out/ngram_vocab.txt`

## Autoencoder Demo

`autoencoder.c` is a minimal MNIST MLP autoencoder:

- flattens each image to `784` inputs
- encodes through a hidden layer and latent bottleneck
- decodes back to pixel space with a sigmoid output layer
- trains with MSE reconstruction loss

Build:

```bash
make autoencoder
```

Run:

```bash
./autoencoder \
  --epochs=10 \
  --batch=32 \
  --hidden=128 \
  --latent=32 \
  --lr=0.01 \
  --log=out/autoencoder_training_log.csv \
  --recon=out/autoencoder_recon.csv
```

Named args: `--epochs=N --batch=N --hidden=N --latent=N --lr=FLOAT --log=PATH --recon=PATH`

The demo expects the raw MNIST IDX files under `data/mnist/` and writes one CSV
row per epoch with `train_loss`, `eval_loss`, and `elapsed_s`. It also saves a
small reconstruction dump CSV that can be rendered to a PNG grid:

```bash
make plot-autoencoder RECON=out/autoencoder_recon.csv RECON_OUT=out/autoencoder_recon.png
```

## Seq2Seq Demo

`seq2seq.c` is a minimal encoder-decoder demo for a synthetic digit-sequence
reversal task.

The current baseline uses:

- tiny vocabulary with digits plus `BOS` / `EOS`
- tanh RNN encoder
- tanh RNN decoder
- no attention yet
- simple length curriculum from shorter to longer sequences
- periodic CSV logging for baseline comparison

Build and run:

```bash
make seq2seq
./seq2seq \
  --steps=2000 \
  --batch=32 \
  --embed=16 \
  --hidden=32 \
  --min-len=3 \
  --max-len=8 \
  --lr=0.03 \
  --log=out/seq2seq_training_log.csv
```

Named args: `--steps=N --batch=N --embed=N --hidden=N --lr=FLOAT --min-len=N --max-len=N --log=PATH`

The demo logs checkpoint rows with `train_loss`, token accuracy, exact-sequence
accuracy, and the active curriculum max length to `out/seq2seq_training_log.csv`.
The design note lives in `docs/seq2seq.md`.

## MNIST Conv Demo (MVP)

`convnet.c` is a minimal conv-like classifier using MNIST and existing 2D tensor ops.
The shared IDX loader and batch helpers now live in `mnist.c` / `mnist.h`:

- `im2col` patch extraction in C
- `matmul + bias + relu` as the convolution stage
- patch logits pooled per image, then softmax + cross entropy

Build:

```bash
make mnist-conv
```

Download the raw IDX dataset files:

```bash
make mnist-data
```

Run (expects raw IDX files, not `.gz`):

```bash
./mnist_conv_demo \
  --epochs=5 \
  --batch=32 \
  --channels=8 \
  --opt=momentum \
  --lr=0.03 \
  --momentum=0.9 \
  --log=out/mnist_training_log.csv
```

Named args: `--epochs=N --batch=N --channels=N --opt=sgd|momentum|adam --lr=FLOAT --momentum=FLOAT --log=PATH`

Dataset paths and the current train/test subset defaults are fixed in the demo for now.

The demo writes one CSV row per epoch with `train_loss`, `train_acc`, `train_error`, `test_acc`, and
`test_error`.

Generate a simple training-loss chart from a metrics CSV:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python3 scripts/plot_training_loss.py out/mnist_training_log.csv
```

Or via `make`:

```bash
make plot-loss LOG=out/mnist_training_log.csv OUT=out/mnist_training_loss.png
```
