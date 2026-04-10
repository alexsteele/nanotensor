#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdio.h>

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
Tensor *tensor_create_default(int rows, int cols);
Tensor *tensor_from_array_default(int rows, int cols, const float *values);
Tensor *tensor_cpy(Tensor *a);
void tensor_free(Tensor *t);

void tensor_set_grad_mode(int enabled);
int tensor_get_grad_mode(void);

void tensor_fill(Tensor *t, float value);
void tensor_fill_randn(Tensor *t, float mean, float stddev, unsigned int *seed);
void tensor_zero_grad(Tensor *t);
void tensor_print_shape(const Tensor *t);
void tensor_print(const Tensor *t, const char *name, int print_grad);
int tensor_save_file(const Tensor *t, FILE *f);
Tensor *tensor_load_file(FILE *f);
int tensor_save(const Tensor *t, const char *path);
Tensor *tensor_load(const char *path);
int tensor_snapshot_save(Tensor **tensors, size_t count, const char *path);
int tensor_snapshot_load(Tensor **tensors, size_t count, const char *path);

Tensor *tensor_add(Tensor *a, Tensor *b);
Tensor *tensor_sub(Tensor *a, Tensor *b);
Tensor *tensor_mul_elem(Tensor *a, Tensor *b);
Tensor *tensor_add_broadcast(Tensor *a, Tensor *b);
Tensor *tensor_mul_broadcast(Tensor *a, Tensor *b);
Tensor *tensor_scalar_mul(Tensor *a, float scalar);
Tensor *tensor_matmul(Tensor *a, Tensor *b);
Tensor *tensor_add_bias(Tensor *x, Tensor *bias_row);
Tensor *tensor_reshape(Tensor *a, int rows, int cols);
Tensor *tensor_transpose(Tensor *a);
Tensor *tensor_slice(Tensor *a, int row_start, int row_end, int col_start, int col_end);
Tensor *tensor_sum_axis(Tensor *a, int axis);
Tensor *tensor_mean_axis(Tensor *a, int axis);
Tensor *tensor_cmp(Tensor *a, Tensor *b);
int tensor_equal(Tensor *a, Tensor *b);
int tensor_allclose(Tensor *a, Tensor *b, float atol, float rtol);

Tensor *tensor_relu(Tensor *x);
Tensor *tensor_sigmoid(Tensor *x);
Tensor *tensor_tanh(Tensor *x);
Tensor *tensor_pow(Tensor *x, float exponent);
Tensor *tensor_sqrt(Tensor *x);
Tensor *tensor_softmax(Tensor *x);
Tensor *tensor_layernorm(Tensor *x, Tensor *gamma, Tensor *beta, float eps);

Tensor *tensor_mse_loss(Tensor *pred, Tensor *target);
Tensor *tensor_cross_entropy(Tensor *pred_probs, Tensor *target_probs);

Tensor *tensor_forward(Tensor *output);
void tensor_backward(Tensor *loss);

void tensor_sgd_step(Tensor **params, size_t n_params, float lr);
void tensor_sgd_momentum_step(Tensor **params, Tensor **velocity, size_t n_params, float lr, float momentum);

#endif
