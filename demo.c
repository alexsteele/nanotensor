#include "tensor.h"

/*
 * demo.c
 *
 * Minimal regression demo using nanotensor.
 * - Generates a small synthetic nonlinear dataset in 2D
 * - Trains a 2-layer MLP with tanh hidden activations
 * - Saves and reloads a parameter snapshot after training
 *
 * Usage:
 *   ./demo
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float rand_uniform(unsigned int *seed) {
    *seed = (1103515245U * (*seed) + 12345U);
    return (float)((*seed) & 0x7fffffffU) / 2147483648.0f;
}

static void generate_dataset(float *x, float *y, int n, unsigned int *seed) {
    for (int i = 0; i < n; i++) {
        float a = -1.0f + 2.0f * rand_uniform(seed);
        float b = -1.0f + 2.0f * rand_uniform(seed);
        x[i * 2 + 0] = a;
        x[i * 2 + 1] = b;

        /* Non-linear target so a 2-layer net is meaningful. */
        y[i] = 0.5f * sinf(3.0f * a) + 0.7f * b * b - 0.2f;
    }
}

int main(void) {
    const int n_samples = 256;
    const int in_dim = 2;
    const int hidden_dim = 16;
    const int out_dim = 1;

    unsigned int seed = 42;

    float *x_data = (float *)malloc(sizeof(float) * (size_t)n_samples * (size_t)in_dim);
    float *y_data = (float *)malloc(sizeof(float) * (size_t)n_samples * (size_t)out_dim);
    if (!x_data || !y_data) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    generate_dataset(x_data, y_data, n_samples, &seed);

    Tensor *X = tensor_from_array(n_samples, in_dim, x_data, 0);
    Tensor *Y = tensor_from_array(n_samples, out_dim, y_data, 0);

    Tensor *W1 = tensor_create(in_dim, hidden_dim, 1);
    Tensor *b1 = tensor_create(1, hidden_dim, 1);
    Tensor *W2 = tensor_create(hidden_dim, out_dim, 1);
    Tensor *b2 = tensor_create(1, out_dim, 1);

    tensor_fill_randn(W1, 0.0f, 0.35f, &seed);
    tensor_fill_randn(W2, 0.0f, 0.35f, &seed);
    tensor_fill(b1, 0.0f);
    tensor_fill(b2, 0.0f);

    Tensor *params[] = {W1, b1, W2, b2};
    const size_t n_params = sizeof(params) / sizeof(params[0]);

    const int epochs = 1200;
    const float lr = 0.03f;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        TensorList temps = {0};
        Tensor *h1_lin = tensor_list_add(&temps, tensor_matmul(X, W1));
        Tensor *h1_bias = tensor_list_add(&temps, tensor_add_bias(h1_lin, b1));
        Tensor *h1 = tensor_list_add(&temps, tensor_tanh(h1_bias));
        Tensor *out_lin = tensor_list_add(&temps, tensor_matmul(h1, W2));
        Tensor *pred = tensor_list_add(&temps, tensor_add_bias(out_lin, b2));
        Tensor *pred_forward = tensor_list_add(&temps, tensor_forward(pred));
        Tensor *loss = tensor_list_add(&temps, tensor_mse_loss(pred_forward, Y));
        tensor_backward(loss);
        tensor_sgd_step(params, n_params, lr);

        if (epoch % 100 == 0 || epoch == 1) {
            printf("epoch %4d  loss %.6f\n", epoch, loss->data[0]);
        }

        tensor_list_free(&temps);
    }

    TensorList temps = {0};
    Tensor *h1_lin = tensor_list_add(&temps, tensor_matmul(X, W1));
    Tensor *h1_bias = tensor_list_add(&temps, tensor_add_bias(h1_lin, b1));
    Tensor *h1 = tensor_list_add(&temps, tensor_tanh(h1_bias));
    Tensor *out_lin = tensor_list_add(&temps, tensor_matmul(h1, W2));
    Tensor *pred = tensor_list_add(&temps, tensor_add_bias(out_lin, b2));
    Tensor *final_loss = tensor_list_add(&temps, tensor_mse_loss(pred, Y));
    float final_loss_value = final_loss->data[0];

    printf("final loss %.6f\n", final_loss->data[0]);

    tensor_list_free(&temps);

    if (tensor_snapshot_save(params, n_params, "out/model_snapshot.bin") != 0) {
        fprintf(stderr, "failed to save snapshot\n");
        return 1;
    }
    W1->data[0] += 5.0f;
    if (tensor_snapshot_load(params, n_params, "out/model_snapshot.bin") != 0) {
        fprintf(stderr, "failed to load snapshot\n");
        return 1;
    }

    temps = (TensorList){0};
    h1_lin = tensor_list_add(&temps, tensor_matmul(X, W1));
    h1_bias = tensor_list_add(&temps, tensor_add_bias(h1_lin, b1));
    h1 = tensor_list_add(&temps, tensor_tanh(h1_bias));
    out_lin = tensor_list_add(&temps, tensor_matmul(h1, W2));
    pred = tensor_list_add(&temps, tensor_add_bias(out_lin, b2));
    final_loss = tensor_list_add(&temps, tensor_mse_loss(pred, Y));
    printf("reloaded loss %.6f (was %.6f)\n", final_loss->data[0], final_loss_value);
    tensor_list_free(&temps);

    tensor_free(W1);
    tensor_free(b1);
    tensor_free(W2);
    tensor_free(b2);
    tensor_free(X);
    tensor_free(Y);

    free(x_data);
    free(y_data);
    return 0;
}
