#include "tensor.h"

/*
 * tensor_bench.c
 *
 * Small microbenchmark runner for a few high-value tensor ops.
 * - Prints a compact table with representative input shapes
 * - Reports average ns/op across a timed loop
 * - Includes a rough GFLOP/s estimate for ops where the math count is clear
 *
 * Usage:
 *   ./tensor_bench [--iters=N] [--warmup=N]
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

typedef enum {
    BENCH_MATMUL = 0,
    BENCH_RELU,
    BENCH_LAYERNORM,
    BENCH_SOFTMAX
} BenchKind;

typedef struct {
    const char *name;
    const char *tag;
    BenchKind kind;
    int rows;
    int inner;
    int cols;
    int flops_per_call;
} BenchCase;

static double now_seconds(void) {
    struct timeval tv;
    if (gettimeofday(&tv, NULL) != 0) {
        fprintf(stderr, "gettimeofday failed\n");
        exit(1);
    }
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static void fill_pattern(Tensor *t, float scale) {
    int n = t->rows * t->cols;
    for (int i = 0; i < n; i++) {
        int bucket = (i % 23) - 11;
        t->data[i] = scale * (float)bucket / 11.0f;
    }
}

static const char *shape_string(const BenchCase *bc, char *buf, size_t buf_size) {
    if (bc->kind == BENCH_MATMUL) {
        if (bc->tag && bc->tag[0] != '\0') {
            snprintf(buf, buf_size, "%s [%d,%d] x [%d,%d]", bc->tag, bc->rows, bc->inner, bc->inner, bc->cols);
        } else {
            snprintf(buf, buf_size, "[%d,%d] x [%d,%d]", bc->rows, bc->inner, bc->inner, bc->cols);
        }
    } else {
        if (bc->tag && bc->tag[0] != '\0') {
            snprintf(buf, buf_size, "%s [%d,%d]", bc->tag, bc->rows, bc->cols);
        } else {
            snprintf(buf, buf_size, "[%d,%d]", bc->rows, bc->cols);
        }
    }
    return buf;
}

static double run_matmul_case(const BenchCase *bc, int warmup_iters, int iters) {
    TensorList tensors = {0};
    Tensor *a = tensor_list_add(&tensors, tensor_create(bc->rows, bc->inner, 0));
    Tensor *b = tensor_list_add(&tensors, tensor_create(bc->inner, bc->cols, 0));
    Tensor *out = NULL;
    double start;
    double elapsed;

    fill_pattern(a, 1.0f);
    fill_pattern(b, 0.5f);

    for (int i = 0; i < warmup_iters; i++) {
        out = tensor_matmul(a, b);
        tensor_free(out);
    }

    start = now_seconds();
    for (int i = 0; i < iters; i++) {
        out = tensor_matmul(a, b);
        tensor_free(out);
    }
    elapsed = now_seconds() - start;

    tensor_list_free(&tensors);
    return elapsed;
}

static double run_relu_case(const BenchCase *bc, int warmup_iters, int iters) {
    TensorList tensors = {0};
    Tensor *x = tensor_list_add(&tensors, tensor_create(bc->rows, bc->cols, 0));
    Tensor *out = NULL;
    double start;
    double elapsed;

    fill_pattern(x, 1.0f);

    for (int i = 0; i < warmup_iters; i++) {
        out = tensor_relu(x);
        tensor_free(out);
    }

    start = now_seconds();
    for (int i = 0; i < iters; i++) {
        out = tensor_relu(x);
        tensor_free(out);
    }
    elapsed = now_seconds() - start;

    tensor_list_free(&tensors);
    return elapsed;
}

static double run_layernorm_case(const BenchCase *bc, int warmup_iters, int iters) {
    TensorList tensors = {0};
    Tensor *x = tensor_list_add(&tensors, tensor_create(bc->rows, bc->cols, 0));
    Tensor *gamma = tensor_list_add(&tensors, tensor_create(1, bc->cols, 0));
    Tensor *beta = tensor_list_add(&tensors, tensor_create(1, bc->cols, 0));
    Tensor *out = NULL;
    double start;
    double elapsed;

    fill_pattern(x, 1.0f);
    tensor_fill(gamma, 1.0f);
    tensor_fill(beta, 0.0f);

    for (int i = 0; i < warmup_iters; i++) {
        out = tensor_layernorm(x, gamma, beta, 1e-5f);
        tensor_free(out);
    }

    start = now_seconds();
    for (int i = 0; i < iters; i++) {
        out = tensor_layernorm(x, gamma, beta, 1e-5f);
        tensor_free(out);
    }
    elapsed = now_seconds() - start;

    tensor_list_free(&tensors);
    return elapsed;
}

static double run_softmax_case(const BenchCase *bc, int warmup_iters, int iters) {
    TensorList tensors = {0};
    Tensor *x = tensor_list_add(&tensors, tensor_create(bc->rows, bc->cols, 0));
    Tensor *out = NULL;
    double start;
    double elapsed;

    fill_pattern(x, 1.0f);

    for (int i = 0; i < warmup_iters; i++) {
        out = tensor_softmax(x);
        tensor_free(out);
    }

    start = now_seconds();
    for (int i = 0; i < iters; i++) {
        out = tensor_softmax(x);
        tensor_free(out);
    }
    elapsed = now_seconds() - start;

    tensor_list_free(&tensors);
    return elapsed;
}

static double run_case(const BenchCase *bc, int warmup_iters, int iters) {
    switch (bc->kind) {
        case BENCH_MATMUL:
            return run_matmul_case(bc, warmup_iters, iters);
        case BENCH_RELU:
            return run_relu_case(bc, warmup_iters, iters);
        case BENCH_LAYERNORM:
            return run_layernorm_case(bc, warmup_iters, iters);
        case BENCH_SOFTMAX:
            return run_softmax_case(bc, warmup_iters, iters);
    }
    fprintf(stderr, "unknown bench kind\n");
    exit(1);
}

static int default_iters(const BenchCase *bc) {
    if (bc->kind == BENCH_MATMUL) {
        int work = bc->rows * bc->inner * bc->cols;
        if (work >= 256 * 576 * 16) {
            return 50;
        }
        if (work >= 64 * 256 * 256) {
            return 20;
        }
        return 200;
    }
    if (bc->rows * bc->cols >= 512 * 256) {
        return 200;
    }
    return 1000;
}

int main(int argc, char **argv) {
    const BenchCase cases[] = {
        {"matmul", "", BENCH_MATMUL, 64, 256, 256, 2 * 64 * 256 * 256},
        {"matmul", "", BENCH_MATMUL, 256, 576, 16, 2 * 256 * 576 * 16},
        {"matmul", "", BENCH_MATMUL, 512, 32, 10, 2 * 512 * 32 * 10},
        {"matmul", "gpt-qkv", BENCH_MATMUL, 32, 32, 32, 2 * 32 * 32 * 32},
        {"matmul", "gpt-ff1", BENCH_MATMUL, 32, 32, 64, 2 * 32 * 32 * 64},
        {"matmul", "gpt-ff2", BENCH_MATMUL, 32, 64, 32, 2 * 32 * 64 * 32},
        {"matmul", "conv-stem", BENCH_MATMUL, 256 * 576, 25, 32, 2 * (256 * 576) * 25 * 32},
        {"matmul", "res-ff1", BENCH_MATMUL, 256 * 576, 32, 64, 2 * (256 * 576) * 32 * 64},
        {"matmul", "res-ff2", BENCH_MATMUL, 256 * 576, 64, 32, 2 * (256 * 576) * 64 * 32},
        {"matmul", "ae-enc1", BENCH_MATMUL, 32, 784, 256, 2 * 32 * 784 * 256},
        {"matmul", "ae-latent", BENCH_MATMUL, 32, 256, 64, 2 * 32 * 256 * 64},
        {"matmul", "ae-dec1", BENCH_MATMUL, 32, 64, 256, 2 * 32 * 64 * 256},
        {"matmul", "ae-out", BENCH_MATMUL, 32, 256, 784, 2 * 32 * 256 * 784},
        {"relu", "", BENCH_RELU, 256, 0, 256, 256 * 256},
        {"relu", "", BENCH_RELU, 512, 0, 64, 512 * 64},
        {"layernorm", "", BENCH_LAYERNORM, 256, 0, 32, 5 * 256 * 32},
        {"layernorm", "", BENCH_LAYERNORM, 512, 0, 64, 5 * 512 * 64},
        {"layernorm", "res-ln", BENCH_LAYERNORM, 256 * 576, 0, 32, 5 * (256 * 576) * 32},
        {"softmax", "", BENCH_SOFTMAX, 256, 0, 10, 4 * 256 * 10},
        {"softmax", "", BENCH_SOFTMAX, 32, 0, 256, 4 * 32 * 256},
    };
    int forced_iters = 0;
    int warmup_iters = 5;
    char shape_buf[64];

    for (int i = 1; i < argc; i++) {
        if (sscanf(argv[i], "--iters=%d", &forced_iters) == 1) {
            continue;
        }
        if (sscanf(argv[i], "--warmup=%d", &warmup_iters) == 1) {
            continue;
        }
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("usage: %s [--iters=N] [--warmup=N]\n", argv[0]);
            return 0;
        }
        fprintf(stderr, "unknown option: %s\n", argv[i]);
        return 1;
    }
    if (warmup_iters < 0) {
        fprintf(stderr, "warmup must be >= 0\n");
        return 1;
    }

    printf("nanotensor microbench\n");
    printf("matmul backend: %s\n", tensor_matmul_backend_name());
    printf("warmup iters: %d\n", warmup_iters);
    printf("%-10s %-30s %8s %14s %12s\n", "op", "shape", "iters", "ns/op", "GFLOP/s");
    printf("%-10s %-30s %8s %14s %12s\n", "----------", "------------------------------", "--------", "--------------", "------------");

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const BenchCase *bc = &cases[i];
        int iters = forced_iters > 0 ? forced_iters : default_iters(bc);
        double elapsed = run_case(bc, warmup_iters, iters);
        double ns_per_op = (elapsed * 1e9) / (double)iters;
        double gflops = bc->flops_per_call > 0 && elapsed > 0.0
                            ? ((double)bc->flops_per_call * (double)iters) / elapsed / 1e9
                            : 0.0;

        printf("%-10s %-30s %8d %14.0f %12.2f\n",
               bc->name,
               shape_string(bc, shape_buf, sizeof(shape_buf)),
               iters,
               ns_per_op,
               gflops);
    }

    return 0;
}
