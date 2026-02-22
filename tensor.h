#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct Tensor Tensor;

struct Tensor {
    int rows;
    int cols;
    int requires_grad;
    float *data;
    float *grad;

    Tensor **parents;
    int n_parents;
    void (*backward_fn)(Tensor *);

    int op;
    float scalar;
};

Tensor *tensor_create(int rows, int cols, int requires_grad);
Tensor *tensor_from_array(int rows, int cols, const float *values, int requires_grad);
void tensor_free(Tensor *t);

void tensor_fill(Tensor *t, float value);
void tensor_fill_randn(Tensor *t, float mean, float stddev, unsigned int *seed);
void tensor_zero_grad(Tensor *t);
void tensor_print_shape(const Tensor *t);
void tensor_print(const Tensor *t, const char *name, int print_grad);

Tensor *tensor_add(Tensor *a, Tensor *b);
Tensor *tensor_sub(Tensor *a, Tensor *b);
Tensor *tensor_mul_elem(Tensor *a, Tensor *b);
Tensor *tensor_scalar_mul(Tensor *a, float scalar);
Tensor *tensor_matmul(Tensor *a, Tensor *b);
Tensor *tensor_add_bias(Tensor *x, Tensor *bias_row);

Tensor *tensor_relu(Tensor *x);
Tensor *tensor_sigmoid(Tensor *x);
Tensor *tensor_tanh(Tensor *x);

Tensor *tensor_mse_loss(Tensor *pred, Tensor *target);

Tensor *tensor_forward(Tensor *output);
void tensor_backward(Tensor *loss);

void tensor_sgd_step(Tensor **params, size_t n_params, float lr);

#endif
