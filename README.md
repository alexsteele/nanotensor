# nanotensor

A minimal tensor/autodiff library in C for small machine-learning experiments.

This project was developed with ChatGPT Codex.

## Features

- Core tensor ops
- Autograd
- Serialization

## Build

```bash
make static      # builds libtensor.a
make             # builds demo linked against libtensor.a
make test        # builds and runs unit tests
make run         # runs the demo training program
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
