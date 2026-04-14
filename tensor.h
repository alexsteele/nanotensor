#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdio.h>

typedef struct Tensor Tensor;
typedef struct TensorAdamOptions TensorAdamOptions;
typedef struct TensorList TensorList;

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
    /* Small per-op metadata slots used by some backward passes, e.g. slice bounds. */
    int aux0;
    int aux1;
    int aux2;
    int aux3;
};

struct TensorAdamOptions {
    float lr;
    float beta1;
    float beta2;
    float eps;
    int timestep;
};

/* Ownership helper for groups of tensors with the same lifetime.
 *
 * Initialization:
 * - stack locals may be zero-initialized directly with `TensorList list = {0};`
 * - or initialized explicitly with `tensor_list_init(&list)`
 *
 * Ownership:
 * - once a tensor is added with `tensor_list_add`, that list becomes
 *   responsible for freeing it via `tensor_list_clear` or `tensor_list_free`
 * - tensors tracked by a list should not also be freed manually
 */
struct TensorList {
    Tensor **items;
    int len;
    int cap;
};

Tensor *tensor_create(int rows, int cols, int requires_grad);
Tensor *tensor_from_array(int rows, int cols, const float *values, int requires_grad);
Tensor *tensor_create_default(int rows, int cols);
Tensor *tensor_from_array_default(int rows, int cols, const float *values);
Tensor *tensor_cpy(Tensor *a);
void tensor_free(Tensor *t);

void tensor_list_init(TensorList *list);
/* Adds `t` to `list` and returns it for convenient inline use.
 * The list takes ownership of the tensor pointer.
 */
Tensor *tensor_list_add(TensorList *list, Tensor *t);
void tensor_list_clear(TensorList *list);
void tensor_list_free(TensorList *list);

void tensor_set_grad_mode(int enabled);
int tensor_get_grad_mode(void);

void tensor_fill(Tensor *t, float value);
void tensor_fill_randn(Tensor *t, float mean, float stddev, unsigned int *seed);
void tensor_zero_grad(Tensor *t);
Tensor *tensor_one_hot(const int *idx, int n, int classes);
int tensor_argmax_row(const Tensor *t, int row);
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
Tensor *tensor_concat_cols(Tensor *a, Tensor *b);
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
/* Binary cross-entropy over probability inputs in [0, 1].
 * References:
 * - https://docs.pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
 * - https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html
 */
Tensor *tensor_binary_cross_entropy(Tensor *pred_probs, Tensor *target_probs);
Tensor *tensor_cross_entropy(Tensor *pred_probs, Tensor *target_probs);

Tensor *tensor_forward(Tensor *output);
void tensor_backward(Tensor *loss);

void tensor_sgd_step(Tensor **params, size_t n_params, float lr);
void tensor_sgd_momentum_step(Tensor **params, Tensor **velocity, size_t n_params, float lr, float momentum);
/* Adam update using caller-owned moment buffers:
 * - m1 stores the exponential moving average of gradients (first moment)
 * - m2 stores the exponential moving average of squared gradients (second moment)
 * References:
 * - https://arxiv.org/abs/1412.6980
 * - https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
 */
void tensor_adam_step(Tensor **params,
                      Tensor **m1,
                      Tensor **m2,
                      size_t n_params,
                      const TensorAdamOptions *opt);

#endif
