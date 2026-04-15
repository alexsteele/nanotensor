# nanotensor

A minimal tensor/autodiff library in C for small machine-learning experiments.

This project was developed with ChatGPT Codex.

## Features

- 1-2D tensors
- Autograd, SGD
- Serialization

Our goal is to support some common model types: multi-layer neural nets, mnist convnets, bigram
language models.

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

## Build

```bash
make static      # builds libtensor.a
make             # builds demo linked against libtensor.a
make test        # builds and runs unit tests
make run         # runs the demo training program
make rebuild     # forces a clean rebuild of all binaries for this machine
make BLAS=accelerate bench # uses Apple Accelerate for matmul on macOS
make mnist-data  # downloads and unpacks raw MNIST IDX files into data/mnist
make gpt-char    # builds the GPT-like char attention demo
make bench       # builds the microbenchmark runner for core tensor ops
make ngram       # builds the neural word-level n-gram demo
make seq2seq     # builds the seq2seq reversal demo
make autoencoder # builds the MNIST autoencoder demo
make mnist-conv  # builds the MNIST patch-conv classifier demo
make resnet      # builds the MNIST residual patch-network demo
make bigdemo     # runs curated stronger demo configs and prints a summary table
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

For a heavier curated showcase run with stronger settings and distinct
`out/bigdemo_*` artifacts, see [docs/bigdemo.md](/Users/alex/Code/nanotensor/docs/bigdemo.md)
and run:

```bash
make BLAS=accelerate bigdemo
```

On macOS this is strongly recommended, since `bigdemo` spends most of its time
in matmul-heavy training loops and the BLAS backend is much faster.


## Demos

- `demo.c`: quickest synthetic smoke test for the core tensor/autodiff flow.
  Run `make run`.
- `llm.c`: minimal character-level language model with a Shakespeare helper
  script. See [docs/llm.md](/Users/alex/Code/nanotensor/docs/llm.md).
- `gpt_char.c`: GPT-like char LM with causal Q/K/V attention. See
  [docs/gpt_char.md](/Users/alex/Code/nanotensor/docs/gpt_char.md).
- `skipgram.c`: word2vec-style skip-gram embedding demo with nearest-neighbor
  reports. See [docs/skipgram.md](/Users/alex/Code/nanotensor/docs/skipgram.md).
- `ngram.c`: neural word-level n-gram language model demo. See
  [docs/ngram.md](/Users/alex/Code/nanotensor/docs/ngram.md).
- `seq2seq.c`: synthetic digit-sequence reversal demo with fixed-eval
  reporting. See [docs/seq2seq.md](/Users/alex/Code/nanotensor/docs/seq2seq.md).
- `tensor_bench.c`: microbenchmark table for core ops like matmul, relu,
  layernorm, and softmax. Run `make run-bench`.
- `convnet.c`: minimal MNIST conv-like classifier built on `im2col` and shared
  MNIST helpers. See [docs/convnet.md](/Users/alex/Code/nanotensor/docs/convnet.md).
- `resnet.c`: MNIST residual patch-network demo with a patch stem, two
  residual MLP blocks, and mean pooling over patches. Supports smaller
  `--train-limit` / `--test-limit` subsets for faster iteration. See
  [docs/resnet.md](/Users/alex/Code/nanotensor/docs/resnet.md).
- `autoencoder.c`: MNIST MLP autoencoder with reconstruction artifacts. See
  [docs/autoencoder.md](/Users/alex/Code/nanotensor/docs/autoencoder.md).

Patch extraction and patch-group mean pooling now live in the shared
[`patch.h`](/Users/alex/Code/nanotensor/patch.h) /
[`patch.c`](/Users/alex/Code/nanotensor/patch.c) helper module so patch-based
MNIST demos can share the same geometry and pooling utilities.

Dense matmul can optionally use a BLAS backend selected at build time:

- `BLAS=none`: current naive C implementation
- `BLAS=accelerate`: Apple Accelerate on macOS
- `BLAS=openblas`: CBLAS/OpenBLAS if installed

Example:

```bash
make BLAS=accelerate bench
./tensor_bench
```
