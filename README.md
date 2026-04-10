# nanotensor

A minimal tensor/autodiff library in C for small machine-learning experiments.

This project was developed with ChatGPT Codex.

## Features

- 1-2D tensors
- Autograd, SGD
- Serialization

Our goal is to support some common model types: multi-layer neural nets, mnist convnets, bigram language models.

## Build

```bash
make static      # builds libtensor.a
make             # builds demo linked against libtensor.a
make test        # builds and runs unit tests
make run         # runs the demo training program
make rebuild     # forces a clean rebuild of all binaries for this machine
make mnist-data  # downloads and unpacks raw MNIST IDX files into data/mnist
```

## Minimal Example

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

`main.c` trains a small 2-layer network on synthetic data and prints training loss.

## LLM Demo

`llm.c` is a minimal character-level language model demo. It trains a small 2-layer MLP
to predict the next character from a text corpus and then generates text from a prompt.

Build and run:

```bash
make llm
./llm shakespeare.txt 3000 "To be" 64 300 0.2 0.9 96
```

To automatically download and prepare a multi-play Shakespeare corpus from Gutenberg and
run the demo:

```bash
make run-shakespeare
```

Or run the script directly with optional args:

```bash
./scripts/run_shakespeare_llm.sh [steps] [prompt] [batch] [gen_len] [lr] [temperature] [hidden]
```

## MNIST Conv Demo (MVP)

`mnist_conv_demo.c` is a minimal conv-like classifier using MNIST and existing 2D tensor ops:

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
  --lr=0.03 \
  --log=mnist_training_log.csv
```

Named args:
`--epochs=N --batch=N --channels=N --lr=FLOAT --log=PATH`

Dataset paths and the current train/test subset defaults are fixed in the demo for now.

The demo writes one CSV row per epoch with `train_loss`, `train_acc`, `train_error`, `test_acc`, and `test_error`.

Generate a simple training-loss chart from a metrics CSV:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python3 scripts/plot_training_loss.py mnist_training_log.csv
```

Or via `make`:

```bash
make plot-loss LOG=mnist_training_log.csv OUT=mnist_training_loss.png
```
