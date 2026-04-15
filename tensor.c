#include "tensor.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(TENSOR_BLAS_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(TENSOR_USE_BLAS)
#include <cblas.h>
#endif

enum {
    OP_NONE = 0,
    OP_ADD,
    OP_SUB,
    OP_MUL_ELEM,
    OP_CONCAT_COLS,
    OP_ADD_BROADCAST,
    OP_MUL_BROADCAST,
    OP_SCALAR_MUL,
    OP_MATMUL,
    OP_ADD_BIAS,
    OP_RESHAPE,
    OP_TRANSPOSE,
    OP_SLICE,
    OP_SUM_AXIS0,
    OP_SUM_AXIS1,
    OP_MEAN_AXIS0,
    OP_MEAN_AXIS1,
    OP_RELU,
    OP_SIGMOID,
    OP_TANH,
    OP_POW,
    OP_SQRT,
    OP_SOFTMAX,
    OP_LAYERNORM,
    OP_MSE,
    OP_BINARY_CROSS_ENTROPY,
    OP_CROSS_ENTROPY
};

typedef struct {
    Tensor **items;
    int len;
    int cap;
} TensorPtrList;

static int g_default_requires_grad = 0;
static const uint32_t k_tensor_magic = 0x544e5352U;   /* "TNSR" */
static const uint32_t k_snapshot_magic = 0x4e545350U; /* "NTSP" */
static const uint32_t k_file_version = 1U;

#if defined(TENSOR_USE_BLAS)
static void matmul_forward_kernel(const Tensor *a, const Tensor *b, Tensor *out) {
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                a->rows,
                b->cols,
                a->cols,
                1.0f,
                a->data,
                a->cols,
                b->data,
                b->cols,
                0.0f,
                out->data,
                out->cols);
}

static void matmul_backward_a_kernel(Tensor *a, const Tensor *b, const Tensor *out) {
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                a->rows,
                a->cols,
                b->cols,
                1.0f,
                out->grad,
                out->cols,
                b->data,
                b->cols,
                1.0f,
                a->grad,
                a->cols);
}

static void matmul_backward_b_kernel(const Tensor *a, Tensor *b, const Tensor *out) {
    cblas_sgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                b->rows,
                b->cols,
                a->rows,
                1.0f,
                a->data,
                a->cols,
                out->grad,
                out->cols,
                1.0f,
                b->grad,
                b->cols);
}
#else
static void matmul_forward_kernel(const Tensor *a, const Tensor *b, Tensor *out) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float acc = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                acc += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            out->data[i * out->cols + j] = acc;
        }
    }
}

static void matmul_backward_a_kernel(Tensor *a, const Tensor *b, const Tensor *out) {
    for (int i = 0; i < a->rows; i++) {
        for (int k = 0; k < a->cols; k++) {
            float acc = 0.0f;
            for (int j = 0; j < b->cols; j++) {
                acc += out->grad[i * out->cols + j] * b->data[k * b->cols + j];
            }
            a->grad[i * a->cols + k] += acc;
        }
    }
}

static void matmul_backward_b_kernel(const Tensor *a, Tensor *b, const Tensor *out) {
    for (int k = 0; k < b->rows; k++) {
        for (int j = 0; j < b->cols; j++) {
            float acc = 0.0f;
            for (int i = 0; i < a->rows; i++) {
                acc += a->data[i * a->cols + k] * out->grad[i * out->cols + j];
            }
            b->grad[k * b->cols + j] += acc;
        }
    }
}
#endif

static int tensor_numel(const Tensor *t) {
    return t->rows * t->cols;
}

static int write_u32(FILE *f, uint32_t v) {
    return fwrite(&v, sizeof(v), 1, f) == 1 ? 0 : -1;
}

static int write_u64(FILE *f, uint64_t v) {
    return fwrite(&v, sizeof(v), 1, f) == 1 ? 0 : -1;
}

static int read_u32(FILE *f, uint32_t *out) {
    return fread(out, sizeof(*out), 1, f) == 1 ? 0 : -1;
}

static int read_u64(FILE *f, uint64_t *out) {
    return fread(out, sizeof(*out), 1, f) == 1 ? 0 : -1;
}

static int infer_requires_grad(Tensor **parents, int n_parents) {
    for (int i = 0; i < n_parents; i++) {
        if (parents[i] && parents[i]->requires_grad) {
            return 1;
        }
    }
    return 0;
}

static int tensor_write_payload(FILE *f, const Tensor *t);
static Tensor *tensor_read_payload_new(FILE *f);
static int tensor_read_payload_into(FILE *f, Tensor *t);

static void die(const char *msg) {
    fprintf(stderr, "tensor error: %s\n", msg);
    exit(1);
}

static void ptr_list_push(TensorPtrList *list, Tensor *t) {
    if (list->len == list->cap) {
        int next_cap = list->cap == 0 ? 16 : list->cap * 2;
        Tensor **next = (Tensor **)realloc(list->items, sizeof(Tensor *) * (size_t)next_cap);
        if (!next) {
            die("out of memory");
        }
        list->items = next;
        list->cap = next_cap;
    }
    list->items[list->len++] = t;
}

static int ptr_list_contains(const TensorPtrList *list, Tensor *t) {
    for (int i = 0; i < list->len; i++) {
        if (list->items[i] == t) {
            return 1;
        }
    }
    return 0;
}

Tensor *tensor_create(int rows, int cols, int requires_grad) {
    if (rows <= 0 || cols <= 0) {
        die("tensor_create: rows/cols must be > 0");
    }
    Tensor *t = (Tensor *)calloc(1, sizeof(Tensor));
    if (!t) {
        die("out of memory");
    }

    int n = rows * cols;
    t->rows = rows;
    t->cols = cols;
    t->requires_grad = requires_grad ? 1 : 0;
    t->data = (float *)calloc((size_t)n, sizeof(float));
    if (!t->data) {
        die("out of memory");
    }
    if (t->requires_grad) {
        t->grad = (float *)calloc((size_t)n, sizeof(float));
        if (!t->grad) {
            die("out of memory");
        }
    }
    t->op = OP_NONE;
    return t;
}

Tensor *tensor_from_array(int rows, int cols, const float *values, int requires_grad) {
    Tensor *t = tensor_create(rows, cols, requires_grad);
    int n = tensor_numel(t);
    memcpy(t->data, values, sizeof(float) * (size_t)n);
    return t;
}

Tensor *tensor_create_default(int rows, int cols) {
    return tensor_create(rows, cols, g_default_requires_grad);
}

Tensor *tensor_from_array_default(int rows, int cols, const float *values) {
    return tensor_from_array(rows, cols, values, g_default_requires_grad);
}

Tensor *tensor_cpy(Tensor *a) {
    Tensor *out;
    int n;
    if (!a) {
        return NULL;
    }
    out = tensor_create(a->rows, a->cols, a->requires_grad);
    n = tensor_numel(a);
    memcpy(out->data, a->data, sizeof(float) * (size_t)n);
    return out;
}

void tensor_set_grad_mode(int enabled) {
    g_default_requires_grad = enabled ? 1 : 0;
}

int tensor_get_grad_mode(void) {
    return g_default_requires_grad;
}

const char *tensor_matmul_backend_name(void) {
#if defined(TENSOR_BLAS_ACCELERATE)
    return "accelerate";
#elif defined(TENSOR_BLAS_OPENBLAS)
    return "openblas";
#else
    return "naive";
#endif
}

void tensor_free(Tensor *t) {
    if (!t) {
        return;
    }
    free(t->data);
    free(t->grad);
    free(t->parents);
    free(t);
}

void tensor_list_init(TensorList *list) {
    if (!list) {
        die("tensor_list_init: list must not be null");
    }
    list->items = NULL;
    list->len = 0;
    list->cap = 0;
}

Tensor *tensor_list_add(TensorList *list, Tensor *t) {
    if (!list || !t) {
        die("tensor_list_add: list and tensor must not be null");
    }
    if (list->len == list->cap) {
        int next_cap = list->cap == 0 ? 16 : list->cap * 2;
        Tensor **next = (Tensor **)realloc(list->items, sizeof(Tensor *) * (size_t)next_cap);
        if (!next) {
            die("out of memory");
        }
        list->items = next;
        list->cap = next_cap;
    }
    list->items[list->len++] = t;
    return t;
}

void tensor_list_clear(TensorList *list) {
    if (!list) {
        return;
    }
    for (int i = 0; i < list->len; i++) {
        tensor_free(list->items[i]);
        list->items[i] = NULL;
    }
    list->len = 0;
}

void tensor_list_free(TensorList *list) {
    if (!list) {
        return;
    }
    tensor_list_clear(list);
    free(list->items);
    list->items = NULL;
    list->cap = 0;
}

int tensor_list_snapshot_save(const TensorList *list, const char *path) {
    if (!list) {
        return -1;
    }
    return tensor_snapshot_save(list->items, (size_t)list->len, path);
}

int tensor_list_snapshot_load(TensorList *list, const char *path) {
    if (!list) {
        return -1;
    }
    return tensor_snapshot_load(list->items, (size_t)list->len, path);
}

void tensor_fill(Tensor *t, float value) {
    int n = tensor_numel(t);
    for (int i = 0; i < n; i++) {
        t->data[i] = value;
    }
}

static float rand_uniform(unsigned int *seed) {
    *seed = (1103515245U * (*seed) + 12345U);
    return (float)((*seed) & 0x7fffffffU) / 2147483648.0f;
}

void tensor_fill_randn(Tensor *t, float mean, float stddev, unsigned int *seed) {
    int n = tensor_numel(t);
    for (int i = 0; i < n; i += 2) {
        float u1 = rand_uniform(seed);
        float u2 = rand_uniform(seed);
        if (u1 < 1e-7f) {
            u1 = 1e-7f;
        }
        float mag = sqrtf(-2.0f * logf(u1));
        float z0 = mag * cosf(2.0f * 3.1415926535f * u2);
        float z1 = mag * sinf(2.0f * 3.1415926535f * u2);
        t->data[i] = mean + stddev * z0;
        if (i + 1 < n) {
            t->data[i + 1] = mean + stddev * z1;
        }
    }
}

void tensor_zero_grad(Tensor *t) {
    if (!t->grad) {
        return;
    }
    memset(t->grad, 0, sizeof(float) * (size_t)tensor_numel(t));
}

Tensor *tensor_one_hot(const int *idx, int n, int classes) {
    Tensor *t;

    if (!idx || n <= 0 || classes <= 0) {
        die("tensor_one_hot: invalid shape or indices");
    }
    t = tensor_create(n, classes, 0);
    tensor_fill(t, 0.0f);
    for (int i = 0; i < n; i++) {
        if (idx[i] < 0 || idx[i] >= classes) {
            tensor_free(t);
            die("tensor_one_hot: index out of range");
        }
        t->data[i * classes + idx[i]] = 1.0f;
    }
    return t;
}

int tensor_argmax_row(const Tensor *t, int row) {
    int best;
    float best_value;

    if (!t || row < 0 || row >= t->rows) {
        die("tensor_argmax_row: row out of range");
    }
    best = 0;
    best_value = t->data[row * t->cols];
    for (int i = 1; i < t->cols; i++) {
        float value = t->data[row * t->cols + i];
        if (value > best_value) {
            best_value = value;
            best = i;
        }
    }
    return best;
}

void tensor_print_shape(const Tensor *t) {
    if (!t) {
        printf("(null)\n");
        return;
    }
    printf("shape: [%d, %d]\n", t->rows, t->cols);
}

void tensor_print(const Tensor *t, const char *name, int print_grad) {
    if (!t) {
        if (name) {
            printf("%s: (null)\n", name);
        } else {
            printf("(null)\n");
        }
        return;
    }

    if (name) {
        printf("%s ", name);
    }
    printf("shape=[%d, %d]\n", t->rows, t->cols);

    for (int i = 0; i < t->rows; i++) {
        printf("[");
        for (int j = 0; j < t->cols; j++) {
            printf("%8.5f", t->data[i * t->cols + j]);
            if (j + 1 < t->cols) {
                printf(", ");
            }
        }
        printf("]\n");
    }

    if (print_grad) {
        if (!t->grad) {
            printf("grad: (none)\n");
            return;
        }
        printf("grad\n");
        for (int i = 0; i < t->rows; i++) {
            printf("[");
            for (int j = 0; j < t->cols; j++) {
                printf("%8.5f", t->grad[i * t->cols + j]);
                if (j + 1 < t->cols) {
                    printf(", ");
                }
            }
            printf("]\n");
        }
    }
}

int tensor_save_file(const Tensor *t, FILE *f) {
    if (!f || !t) {
        return -1;
    }
    if (write_u32(f, k_tensor_magic) != 0 || write_u32(f, k_file_version) != 0) {
        return -1;
    }
    return tensor_write_payload(f, t);
}

Tensor *tensor_load_file(FILE *f) {
    uint32_t magic = 0;
    uint32_t version = 0;

    if (!f) {
        return NULL;
    }
    if (read_u32(f, &magic) != 0 || read_u32(f, &version) != 0) {
        return NULL;
    }
    if (magic != k_tensor_magic || version != k_file_version) {
        return NULL;
    }
    return tensor_read_payload_new(f);
}

int tensor_save(const Tensor *t, const char *path) {
    FILE *f;
    int ok;
    if (!path || !t) {
        return -1;
    }
    f = fopen(path, "wb");
    if (!f) {
        return -1;
    }
    ok = tensor_save_file(t, f);
    if (fclose(f) != 0) {
        return -1;
    }
    return ok;
}

Tensor *tensor_load(const char *path) {
    FILE *f;
    Tensor *t;
    if (!path) {
        return NULL;
    }
    f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    t = tensor_load_file(f);
    fclose(f);
    return t;
}

int tensor_snapshot_save(Tensor **tensors, size_t count, const char *path) {
    FILE *f;
    if (!path || (!tensors && count > 0)) {
        return -1;
    }
    f = fopen(path, "wb");
    if (!f) {
        return -1;
    }
    if (write_u32(f, k_snapshot_magic) != 0 ||
        write_u32(f, k_file_version) != 0 ||
        write_u64(f, (uint64_t)count) != 0) {
        fclose(f);
        return -1;
    }
    for (size_t i = 0; i < count; i++) {
        if (!tensors[i] || tensor_write_payload(f, tensors[i]) != 0) {
            fclose(f);
            return -1;
        }
    }
    if (fclose(f) != 0) {
        return -1;
    }
    return 0;
}

int tensor_snapshot_load(Tensor **tensors, size_t count, const char *path) {
    FILE *f;
    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t file_count = 0;
    if (!path || (!tensors && count > 0)) {
        return -1;
    }
    f = fopen(path, "rb");
    if (!f) {
        return -1;
    }
    if (read_u32(f, &magic) != 0 ||
        read_u32(f, &version) != 0 ||
        read_u64(f, &file_count) != 0) {
        fclose(f);
        return -1;
    }
    if (magic != k_snapshot_magic || version != k_file_version || file_count != (uint64_t)count) {
        fclose(f);
        return -1;
    }
    for (size_t i = 0; i < count; i++) {
        if (!tensors[i] || tensor_read_payload_into(f, tensors[i]) != 0) {
            fclose(f);
            return -1;
        }
    }
    if (fclose(f) != 0) {
        return -1;
    }
    return 0;
}

static Tensor *tensor_new_op(int rows, int cols, int requires_grad, int op, Tensor **parents, int n_parents,
                             void (*backward_fn)(Tensor *)) {
    Tensor *out = tensor_create(rows, cols, requires_grad);
    out->op = op;
    out->n_parents = n_parents;
    if (n_parents > 0) {
        out->parents = (Tensor **)calloc((size_t)n_parents, sizeof(Tensor *));
        if (!out->parents) {
            die("out of memory");
        }
        for (int i = 0; i < n_parents; i++) {
            out->parents[i] = parents[i];
        }
    }
    out->backward_fn = backward_fn;
    return out;
}

static void ensure_same_shape(Tensor *a, Tensor *b, const char *who) {
    if (a->rows != b->rows || a->cols != b->cols) {
        die(who);
    }
}

static void infer_broadcast_shape(Tensor *a, Tensor *b, int *out_rows, int *out_cols, const char *who) {
    int rows;
    int cols;
    if ((a->rows != b->rows) && (a->rows != 1) && (b->rows != 1)) {
        die(who);
    }
    if ((a->cols != b->cols) && (a->cols != 1) && (b->cols != 1)) {
        die(who);
    }
    rows = a->rows > b->rows ? a->rows : b->rows;
    cols = a->cols > b->cols ? a->cols : b->cols;
    *out_rows = rows;
    *out_cols = cols;
}

static void ensure_slice_bounds(Tensor *a, int row_start, int row_end, int col_start, int col_end, const char *who) {
    if (row_start < 0 || row_end < row_start || row_end > a->rows ||
        col_start < 0 || col_end < col_start || col_end > a->cols) {
        die(who);
    }
}

static void ensure_grad(Tensor *t) {
    if (!t->grad) {
        int n = tensor_numel(t);
        t->grad = (float *)calloc((size_t)n, sizeof(float));
        if (!t->grad) {
            die("out of memory");
        }
    }
}

static int tensor_set_requires_grad(Tensor *t, int requires_grad) {
    t->requires_grad = requires_grad ? 1 : 0;
    if (t->requires_grad) {
        ensure_grad(t);
        tensor_zero_grad(t);
    } else if (t->grad) {
        free(t->grad);
        t->grad = NULL;
    }
    return 0;
}

static int tensor_write_payload(FILE *f, const Tensor *t) {
    uint32_t rows = (uint32_t)t->rows;
    uint32_t cols = (uint32_t)t->cols;
    uint32_t req = (uint32_t)(t->requires_grad ? 1 : 0);
    int n = tensor_numel(t);

    if (write_u32(f, rows) != 0 || write_u32(f, cols) != 0 || write_u32(f, req) != 0) {
        return -1;
    }
    if (fwrite(t->data, sizeof(float), (size_t)n, f) != (size_t)n) {
        return -1;
    }
    return 0;
}

static Tensor *tensor_read_payload_new(FILE *f) {
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t req = 0;

    if (read_u32(f, &rows) != 0 || read_u32(f, &cols) != 0 || read_u32(f, &req) != 0) {
        return NULL;
    }
    if (rows == 0 || cols == 0 || rows > 2147483647U || cols > 2147483647U) {
        return NULL;
    }

    Tensor *t = tensor_create((int)rows, (int)cols, req ? 1 : 0);
    int n = tensor_numel(t);
    if (fread(t->data, sizeof(float), (size_t)n, f) != (size_t)n) {
        tensor_free(t);
        return NULL;
    }
    return t;
}

static int tensor_read_payload_into(FILE *f, Tensor *t) {
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t req = 0;
    int n = tensor_numel(t);

    if (read_u32(f, &rows) != 0 || read_u32(f, &cols) != 0 || read_u32(f, &req) != 0) {
        return -1;
    }
    if ((int)rows != t->rows || (int)cols != t->cols) {
        return -1;
    }
    if (tensor_set_requires_grad(t, req ? 1 : 0) != 0) {
        return -1;
    }
    if (fread(t->data, sizeof(float), (size_t)n, f) != (size_t)n) {
        return -1;
    }
    return 0;
}

static void backward_add(Tensor *out) {
    Tensor *a = out->parents[0];
    Tensor *b = out->parents[1];
    int n = tensor_numel(out);
    if (a->requires_grad) {
        ensure_grad(a);
        for (int i = 0; i < n; i++) {
            a->grad[i] += out->grad[i];
        }
    }
    if (b->requires_grad) {
        ensure_grad(b);
        for (int i = 0; i < n; i++) {
            b->grad[i] += out->grad[i];
        }
    }
}

Tensor *tensor_add(Tensor *a, Tensor *b) {
    ensure_same_shape(a, b, "tensor_add: shape mismatch");
    Tensor *parents[2] = {a, b};
    int req = infer_requires_grad(parents, 2);
    Tensor *out = tensor_new_op(a->rows, a->cols, req, OP_ADD, parents, 2, backward_add);
    int n = tensor_numel(out);
    for (int i = 0; i < n; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return out;
}

static void backward_sub(Tensor *out) {
    Tensor *a = out->parents[0];
    Tensor *b = out->parents[1];
    int n = tensor_numel(out);
    if (a->requires_grad) {
        ensure_grad(a);
        for (int i = 0; i < n; i++) {
            a->grad[i] += out->grad[i];
        }
    }
    if (b->requires_grad) {
        ensure_grad(b);
        for (int i = 0; i < n; i++) {
            b->grad[i] -= out->grad[i];
        }
    }
}

Tensor *tensor_sub(Tensor *a, Tensor *b) {
    ensure_same_shape(a, b, "tensor_sub: shape mismatch");
    Tensor *parents[2] = {a, b};
    int req = infer_requires_grad(parents, 2);
    Tensor *out = tensor_new_op(a->rows, a->cols, req, OP_SUB, parents, 2, backward_sub);
    int n = tensor_numel(out);
    for (int i = 0; i < n; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
    return out;
}

static void backward_mul_elem(Tensor *out) {
    Tensor *a = out->parents[0];
    Tensor *b = out->parents[1];
    int n = tensor_numel(out);
    if (a->requires_grad) {
        ensure_grad(a);
        for (int i = 0; i < n; i++) {
            a->grad[i] += b->data[i] * out->grad[i];
        }
    }
    if (b->requires_grad) {
        ensure_grad(b);
        for (int i = 0; i < n; i++) {
            b->grad[i] += a->data[i] * out->grad[i];
        }
    }
}

Tensor *tensor_mul_elem(Tensor *a, Tensor *b) {
    ensure_same_shape(a, b, "tensor_mul_elem: shape mismatch");
    Tensor *parents[2] = {a, b};
    int req = infer_requires_grad(parents, 2);
    Tensor *out = tensor_new_op(a->rows, a->cols, req, OP_MUL_ELEM, parents, 2, backward_mul_elem);
    int n = tensor_numel(out);
    for (int i = 0; i < n; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
    return out;
}

static void backward_concat_cols(Tensor *out) {
    Tensor *a = out->parents[0];
    Tensor *b = out->parents[1];
    if (a->requires_grad) {
        ensure_grad(a);
        for (int i = 0; i < a->rows; i++) {
            for (int j = 0; j < a->cols; j++) {
                a->grad[i * a->cols + j] += out->grad[i * out->cols + j];
            }
        }
    }
    if (b->requires_grad) {
        ensure_grad(b);
        for (int i = 0; i < b->rows; i++) {
            for (int j = 0; j < b->cols; j++) {
                b->grad[i * b->cols + j] += out->grad[i * out->cols + (a->cols + j)];
            }
        }
    }
}

Tensor *tensor_concat_cols(Tensor *a, Tensor *b) {
    Tensor *parents[2] = {a, b};
    int req;
    Tensor *out;

    if (a->rows != b->rows) {
        die("tensor_concat_cols: row mismatch");
    }
    req = infer_requires_grad(parents, 2);
    out = tensor_new_op(a->rows, a->cols + b->cols, req, OP_CONCAT_COLS, parents, 2, backward_concat_cols);
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            out->data[i * out->cols + j] = a->data[i * a->cols + j];
        }
        for (int j = 0; j < b->cols; j++) {
            out->data[i * out->cols + (a->cols + j)] = b->data[i * b->cols + j];
        }
    }
    return out;
}

static void backward_add_broadcast(Tensor *out) {
    Tensor *a = out->parents[0];
    Tensor *b = out->parents[1];
    if (a->requires_grad) {
        ensure_grad(a);
    }
    if (b->requires_grad) {
        ensure_grad(b);
    }
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            int oi = i * out->cols + j;
            int ai = (a->rows == 1 ? 0 : i) * a->cols + (a->cols == 1 ? 0 : j);
            int bi = (b->rows == 1 ? 0 : i) * b->cols + (b->cols == 1 ? 0 : j);
            if (a->requires_grad) {
                a->grad[ai] += out->grad[oi];
            }
            if (b->requires_grad) {
                b->grad[bi] += out->grad[oi];
            }
        }
    }
}

Tensor *tensor_add_broadcast(Tensor *a, Tensor *b) {
    int out_rows;
    int out_cols;
    Tensor *parents[2] = {a, b};
    int req;
    Tensor *out;
    infer_broadcast_shape(a, b, &out_rows, &out_cols, "tensor_add_broadcast: shape mismatch");
    req = infer_requires_grad(parents, 2);
    out = tensor_new_op(out_rows, out_cols, req, OP_ADD_BROADCAST, parents, 2, backward_add_broadcast);
    for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
            int oi = i * out_cols + j;
            int ai = (a->rows == 1 ? 0 : i) * a->cols + (a->cols == 1 ? 0 : j);
            int bi = (b->rows == 1 ? 0 : i) * b->cols + (b->cols == 1 ? 0 : j);
            out->data[oi] = a->data[ai] + b->data[bi];
        }
    }
    return out;
}

static void backward_mul_broadcast(Tensor *out) {
    Tensor *a = out->parents[0];
    Tensor *b = out->parents[1];
    if (a->requires_grad) {
        ensure_grad(a);
    }
    if (b->requires_grad) {
        ensure_grad(b);
    }
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            int oi = i * out->cols + j;
            int ai = (a->rows == 1 ? 0 : i) * a->cols + (a->cols == 1 ? 0 : j);
            int bi = (b->rows == 1 ? 0 : i) * b->cols + (b->cols == 1 ? 0 : j);
            if (a->requires_grad) {
                a->grad[ai] += b->data[bi] * out->grad[oi];
            }
            if (b->requires_grad) {
                b->grad[bi] += a->data[ai] * out->grad[oi];
            }
        }
    }
}

Tensor *tensor_mul_broadcast(Tensor *a, Tensor *b) {
    int out_rows;
    int out_cols;
    Tensor *parents[2] = {a, b};
    int req;
    Tensor *out;
    infer_broadcast_shape(a, b, &out_rows, &out_cols, "tensor_mul_broadcast: shape mismatch");
    req = infer_requires_grad(parents, 2);
    out = tensor_new_op(out_rows, out_cols, req, OP_MUL_BROADCAST, parents, 2, backward_mul_broadcast);
    for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
            int oi = i * out_cols + j;
            int ai = (a->rows == 1 ? 0 : i) * a->cols + (a->cols == 1 ? 0 : j);
            int bi = (b->rows == 1 ? 0 : i) * b->cols + (b->cols == 1 ? 0 : j);
            out->data[oi] = a->data[ai] * b->data[bi];
        }
    }
    return out;
}

static void backward_sum_axis0(Tensor *out) {
    Tensor *a = out->parents[0];
    if (!a->requires_grad) {
        return;
    }
    ensure_grad(a);
    for (int j = 0; j < a->cols; j++) {
        float g = out->grad[j];
        for (int i = 0; i < a->rows; i++) {
            a->grad[i * a->cols + j] += g;
        }
    }
}

static void backward_sum_axis1(Tensor *out) {
    Tensor *a = out->parents[0];
    if (!a->requires_grad) {
        return;
    }
    ensure_grad(a);
    for (int i = 0; i < a->rows; i++) {
        float g = out->grad[i];
        for (int j = 0; j < a->cols; j++) {
            a->grad[i * a->cols + j] += g;
        }
    }
}

Tensor *tensor_sum_axis(Tensor *a, int axis) {
    Tensor *parents[1] = {a};
    int req = infer_requires_grad(parents, 1);
    Tensor *out;
    if (axis == 0) {
        out = tensor_new_op(1, a->cols, req, OP_SUM_AXIS0, parents, 1, backward_sum_axis0);
        for (int j = 0; j < a->cols; j++) {
            float acc = 0.0f;
            for (int i = 0; i < a->rows; i++) {
                acc += a->data[i * a->cols + j];
            }
            out->data[j] = acc;
        }
        return out;
    }
    if (axis == 1) {
        out = tensor_new_op(a->rows, 1, req, OP_SUM_AXIS1, parents, 1, backward_sum_axis1);
        for (int i = 0; i < a->rows; i++) {
            float acc = 0.0f;
            for (int j = 0; j < a->cols; j++) {
                acc += a->data[i * a->cols + j];
            }
            out->data[i] = acc;
        }
        return out;
    }
    die("tensor_sum_axis: axis must be 0 or 1");
    return NULL;
}

static void backward_mean_axis0(Tensor *out) {
    Tensor *a = out->parents[0];
    float scale = 1.0f / (float)a->rows;
    if (!a->requires_grad) {
        return;
    }
    ensure_grad(a);
    for (int j = 0; j < a->cols; j++) {
        float g = out->grad[j] * scale;
        for (int i = 0; i < a->rows; i++) {
            a->grad[i * a->cols + j] += g;
        }
    }
}

static void backward_mean_axis1(Tensor *out) {
    Tensor *a = out->parents[0];
    float scale = 1.0f / (float)a->cols;
    if (!a->requires_grad) {
        return;
    }
    ensure_grad(a);
    for (int i = 0; i < a->rows; i++) {
        float g = out->grad[i] * scale;
        for (int j = 0; j < a->cols; j++) {
            a->grad[i * a->cols + j] += g;
        }
    }
}

Tensor *tensor_mean_axis(Tensor *a, int axis) {
    Tensor *out = tensor_sum_axis(a, axis);
    if (axis == 0) {
        out->op = OP_MEAN_AXIS0;
        out->backward_fn = backward_mean_axis0;
        for (int j = 0; j < out->cols; j++) {
            out->data[j] /= (float)a->rows;
        }
        return out;
    }
    if (axis == 1) {
        out->op = OP_MEAN_AXIS1;
        out->backward_fn = backward_mean_axis1;
        for (int i = 0; i < out->rows; i++) {
            out->data[i] /= (float)a->cols;
        }
        return out;
    }
    die("tensor_mean_axis: axis must be 0 or 1");
    return NULL;
}

static void backward_scalar_mul(Tensor *out) {
    Tensor *a = out->parents[0];
    const float scalar = out->scalar;
    int n = tensor_numel(out);
    if (a->requires_grad) {
        ensure_grad(a);
        for (int i = 0; i < n; i++) {
            a->grad[i] += scalar * out->grad[i];
        }
    }
}

Tensor *tensor_scalar_mul(Tensor *a, float scalar) {
    Tensor *parents[1] = {a};
    int req = infer_requires_grad(parents, 1);
    Tensor *out = tensor_new_op(a->rows, a->cols, req, OP_SCALAR_MUL, parents, 1, backward_scalar_mul);
    out->scalar = scalar;
    int n = tensor_numel(out);
    for (int i = 0; i < n; i++) {
        out->data[i] = a->data[i] * scalar;
    }
    return out;
}

static void backward_matmul(Tensor *out) {
    Tensor *a = out->parents[0];
    Tensor *b = out->parents[1];

    if (a->requires_grad) {
        ensure_grad(a);
        matmul_backward_a_kernel(a, b, out);
    }

    if (b->requires_grad) {
        ensure_grad(b);
        matmul_backward_b_kernel(a, b, out);
    }
}

Tensor *tensor_matmul(Tensor *a, Tensor *b) {
    if (a->cols != b->rows) {
        die("tensor_matmul: shape mismatch");
    }
    Tensor *parents[2] = {a, b};
    int req = infer_requires_grad(parents, 2);
    Tensor *out = tensor_new_op(a->rows, b->cols, req, OP_MATMUL, parents, 2, backward_matmul);
    matmul_forward_kernel(a, b, out);
    return out;
}

static void backward_add_bias(Tensor *out) {
    Tensor *x = out->parents[0];
    Tensor *b = out->parents[1];
    if (x->requires_grad) {
        ensure_grad(x);
        int n = tensor_numel(x);
        for (int i = 0; i < n; i++) {
            x->grad[i] += out->grad[i];
        }
    }
    if (b->requires_grad) {
        ensure_grad(b);
        for (int j = 0; j < b->cols; j++) {
            float acc = 0.0f;
            for (int i = 0; i < x->rows; i++) {
                acc += out->grad[i * out->cols + j];
            }
            b->grad[j] += acc;
        }
    }
}

Tensor *tensor_add_bias(Tensor *x, Tensor *bias_row) {
    if (bias_row->rows != 1 || bias_row->cols != x->cols) {
        die("tensor_add_bias: bias must be shape [1, cols]");
    }
    Tensor *parents[2] = {x, bias_row};
    int req = infer_requires_grad(parents, 2);
    Tensor *out = tensor_new_op(x->rows, x->cols, req, OP_ADD_BIAS, parents, 2, backward_add_bias);

    for (int i = 0; i < x->rows; i++) {
        for (int j = 0; j < x->cols; j++) {
            out->data[i * out->cols + j] = x->data[i * x->cols + j] + bias_row->data[j];
        }
    }
    return out;
}

static void backward_reshape(Tensor *out) {
    Tensor *a = out->parents[0];
    if (!a->requires_grad) {
        return;
    }
    ensure_grad(a);
    int n = tensor_numel(a);
    for (int i = 0; i < n; i++) {
        a->grad[i] += out->grad[i];
    }
}

Tensor *tensor_reshape(Tensor *a, int rows, int cols) {
    if (rows <= 0 || cols <= 0 || rows * cols != tensor_numel(a)) {
        die("tensor_reshape: invalid shape");
    }
    Tensor *parents[1] = {a};
    int req = infer_requires_grad(parents, 1);
    Tensor *out = tensor_new_op(rows, cols, req, OP_RESHAPE, parents, 1, backward_reshape);
    int n = tensor_numel(a);
    for (int i = 0; i < n; i++) {
        out->data[i] = a->data[i];
    }
    return out;
}

static void backward_transpose(Tensor *out) {
    Tensor *a = out->parents[0];
    if (!a->requires_grad) {
        return;
    }
    ensure_grad(a);
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            a->grad[j * a->cols + i] += out->grad[i * out->cols + j];
        }
    }
}

Tensor *tensor_transpose(Tensor *a) {
    Tensor *parents[1] = {a};
    int req = infer_requires_grad(parents, 1);
    Tensor *out = tensor_new_op(a->cols, a->rows, req, OP_TRANSPOSE, parents, 1, backward_transpose);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            out->data[j * out->cols + i] = a->data[i * a->cols + j];
        }
    }
    return out;
}

static void backward_slice(Tensor *out) {
    Tensor *a = out->parents[0];
    int row_start = out->aux0;
    int col_start = out->aux2;

    if (!a->requires_grad) {
        return;
    }
    ensure_grad(a);
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            a->grad[(row_start + i) * a->cols + (col_start + j)] += out->grad[i * out->cols + j];
        }
    }
}

Tensor *tensor_slice(Tensor *a, int row_start, int row_end, int col_start, int col_end) {
    Tensor *parents[1] = {a};
    int req = infer_requires_grad(parents, 1);
    Tensor *out;
    ensure_slice_bounds(a, row_start, row_end, col_start, col_end, "tensor_slice: invalid range");
    int out_rows = row_end - row_start;
    int out_cols = col_end - col_start;
    if (out_rows <= 0 || out_cols <= 0) {
        die("tensor_slice: empty slice");
    }

    out = tensor_new_op(out_rows, out_cols, req, OP_SLICE, parents, 1, backward_slice);
    out->aux0 = row_start;
    out->aux1 = row_end;
    out->aux2 = col_start;
    out->aux3 = col_end;

    for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
            out->data[i * out_cols + j] = a->data[(row_start + i) * a->cols + (col_start + j)];
        }
    }
    return out;
}

Tensor *tensor_cmp(Tensor *a, Tensor *b) {
    ensure_same_shape(a, b, "tensor_cmp: shape mismatch");
    Tensor *out = tensor_create(a->rows, a->cols, 0);
    int n = tensor_numel(out);
    for (int i = 0; i < n; i++) {
        if (a->data[i] < b->data[i]) {
            out->data[i] = -1.0f;
        } else if (a->data[i] > b->data[i]) {
            out->data[i] = 1.0f;
        } else {
            out->data[i] = 0.0f;
        }
    }
    return out;
}

int tensor_equal(Tensor *a, Tensor *b) {
    ensure_same_shape(a, b, "tensor_equal: shape mismatch");
    int n = tensor_numel(a);
    for (int i = 0; i < n; i++) {
        if (a->data[i] != b->data[i]) {
            return 0;
        }
    }
    return 1;
}

int tensor_allclose(Tensor *a, Tensor *b, float atol, float rtol) {
    ensure_same_shape(a, b, "tensor_allclose: shape mismatch");
    int n = tensor_numel(a);
    for (int i = 0; i < n; i++) {
        float ai = a->data[i];
        float bi = b->data[i];
        float diff = fabsf(ai - bi);
        float tol = atol + rtol * fabsf(bi);
        if (diff > tol) {
            return 0;
        }
    }
    return 1;
}

static void backward_relu(Tensor *out) {
    Tensor *x = out->parents[0];
    if (!x->requires_grad) {
        return;
    }
    ensure_grad(x);
    int n = tensor_numel(x);
    for (int i = 0; i < n; i++) {
        x->grad[i] += (x->data[i] > 0.0f ? 1.0f : 0.0f) * out->grad[i];
    }
}

Tensor *tensor_relu(Tensor *x) {
    Tensor *parents[1] = {x};
    int req = infer_requires_grad(parents, 1);
    Tensor *out = tensor_new_op(x->rows, x->cols, req, OP_RELU, parents, 1, backward_relu);
    int n = tensor_numel(out);
    for (int i = 0; i < n; i++) {
        out->data[i] = x->data[i] > 0.0f ? x->data[i] : 0.0f;
    }
    return out;
}

static void backward_sigmoid(Tensor *out) {
    Tensor *x = out->parents[0];
    if (!x->requires_grad) {
        return;
    }
    ensure_grad(x);
    int n = tensor_numel(x);
    for (int i = 0; i < n; i++) {
        float s = out->data[i];
        x->grad[i] += s * (1.0f - s) * out->grad[i];
    }
}

Tensor *tensor_sigmoid(Tensor *x) {
    Tensor *parents[1] = {x};
    int req = infer_requires_grad(parents, 1);
    Tensor *out = tensor_new_op(x->rows, x->cols, req, OP_SIGMOID, parents, 1, backward_sigmoid);
    int n = tensor_numel(out);
    for (int i = 0; i < n; i++) {
        out->data[i] = 1.0f / (1.0f + expf(-x->data[i]));
    }
    return out;
}

static void backward_tanh(Tensor *out) {
    Tensor *x = out->parents[0];
    if (!x->requires_grad) {
        return;
    }
    ensure_grad(x);
    int n = tensor_numel(x);
    for (int i = 0; i < n; i++) {
        float t = out->data[i];
        x->grad[i] += (1.0f - t * t) * out->grad[i];
    }
}

Tensor *tensor_tanh(Tensor *x) {
    Tensor *parents[1] = {x};
    int req = infer_requires_grad(parents, 1);
    Tensor *out = tensor_new_op(x->rows, x->cols, req, OP_TANH, parents, 1, backward_tanh);
    int n = tensor_numel(out);
    for (int i = 0; i < n; i++) {
        out->data[i] = tanhf(x->data[i]);
    }
    return out;
}

static void backward_pow(Tensor *out) {
    Tensor *x = out->parents[0];
    float exponent = out->scalar;
    if (!x->requires_grad) {
        return;
    }
    ensure_grad(x);
    for (int i = 0; i < tensor_numel(x); i++) {
        float xi = x->data[i];
        float local_grad = exponent * powf(xi, exponent - 1.0f);
        x->grad[i] += local_grad * out->grad[i];
    }
}

Tensor *tensor_pow(Tensor *x, float exponent) {
    Tensor *parents[1] = {x};
    int req = infer_requires_grad(parents, 1);
    Tensor *out = tensor_new_op(x->rows, x->cols, req, OP_POW, parents, 1, backward_pow);
    out->scalar = exponent;
    for (int i = 0; i < tensor_numel(x); i++) {
        out->data[i] = powf(x->data[i], exponent);
    }
    return out;
}

static void backward_sqrt(Tensor *out) {
    Tensor *x = out->parents[0];
    if (!x->requires_grad) {
        return;
    }
    ensure_grad(x);
    for (int i = 0; i < tensor_numel(x); i++) {
        float y = out->data[i];
        float denom = 2.0f * (y > 1e-12f ? y : 1e-12f);
        x->grad[i] += out->grad[i] / denom;
    }
}

Tensor *tensor_sqrt(Tensor *x) {
    Tensor *parents[1] = {x};
    int req = infer_requires_grad(parents, 1);
    Tensor *out = tensor_new_op(x->rows, x->cols, req, OP_SQRT, parents, 1, backward_sqrt);
    for (int i = 0; i < tensor_numel(x); i++) {
        out->data[i] = sqrtf(x->data[i]);
    }
    return out;
}

static void backward_layernorm(Tensor *out) {
    Tensor *x = out->parents[0];
    Tensor *gamma = out->parents[1];
    Tensor *beta = out->parents[2];
    float eps = out->scalar;
    int d = x->cols;

    if (gamma->requires_grad) {
        ensure_grad(gamma);
    }
    if (beta->requires_grad) {
        ensure_grad(beta);
    }
    if (x->requires_grad) {
        ensure_grad(x);
    }

    for (int r = 0; r < x->rows; r++) {
        float mean = 0.0f;
        float var = 0.0f;
        float inv_std;
        float sum_dy_hat = 0.0f;
        float sum_dy_hat_xhat = 0.0f;

        for (int c = 0; c < d; c++) {
            mean += x->data[r * d + c];
        }
        mean /= (float)d;
        for (int c = 0; c < d; c++) {
            float v = x->data[r * d + c] - mean;
            var += v * v;
        }
        var /= (float)d;
        inv_std = 1.0f / sqrtf(var + eps);

        for (int c = 0; c < d; c++) {
            int idx = r * d + c;
            float xhat = (x->data[idx] - mean) * inv_std;
            float dy = out->grad[idx];
            float dy_hat = dy * gamma->data[c];
            sum_dy_hat += dy_hat;
            sum_dy_hat_xhat += dy_hat * xhat;

            if (gamma->requires_grad) {
                gamma->grad[c] += dy * xhat;
            }
            if (beta->requires_grad) {
                beta->grad[c] += dy;
            }
        }

        if (x->requires_grad) {
            for (int c = 0; c < d; c++) {
                int idx = r * d + c;
                float xhat = (x->data[idx] - mean) * inv_std;
                float dy = out->grad[idx];
                float dy_hat = dy * gamma->data[c];
                x->grad[idx] += (inv_std / (float)d) *
                               ((float)d * dy_hat - sum_dy_hat - xhat * sum_dy_hat_xhat);
            }
        }
    }
}

Tensor *tensor_layernorm(Tensor *x, Tensor *gamma, Tensor *beta, float eps) {
    Tensor *parents[3] = {x, gamma, beta};
    int req;
    Tensor *out;
    if (gamma->rows != 1 || beta->rows != 1 || gamma->cols != x->cols || beta->cols != x->cols) {
        die("tensor_layernorm: gamma/beta must be shape [1, x->cols]");
    }
    req = infer_requires_grad(parents, 3);
    out = tensor_new_op(x->rows, x->cols, req, OP_LAYERNORM, parents, 3, backward_layernorm);
    out->scalar = eps;

    for (int r = 0; r < x->rows; r++) {
        float mean = 0.0f;
        float var = 0.0f;
        float inv_std;
        for (int c = 0; c < x->cols; c++) {
            mean += x->data[r * x->cols + c];
        }
        mean /= (float)x->cols;
        for (int c = 0; c < x->cols; c++) {
            float v = x->data[r * x->cols + c] - mean;
            var += v * v;
        }
        var /= (float)x->cols;
        inv_std = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < x->cols; c++) {
            int idx = r * x->cols + c;
            float xhat = (x->data[idx] - mean) * inv_std;
            out->data[idx] = xhat * gamma->data[c] + beta->data[c];
        }
    }
    return out;
}

static void backward_softmax(Tensor *out) {
    Tensor *x = out->parents[0];
    if (!x->requires_grad) {
        return;
    }
    ensure_grad(x);
    for (int r = 0; r < out->rows; r++) {
        float dot = 0.0f;
        for (int c = 0; c < out->cols; c++) {
            int idx = r * out->cols + c;
            dot += out->grad[idx] * out->data[idx];
        }
        for (int c = 0; c < out->cols; c++) {
            int idx = r * out->cols + c;
            x->grad[idx] += out->data[idx] * (out->grad[idx] - dot);
        }
    }
}

Tensor *tensor_softmax(Tensor *x) {
    Tensor *parents[1] = {x};
    int req = infer_requires_grad(parents, 1);
    Tensor *out = tensor_new_op(x->rows, x->cols, req, OP_SOFTMAX, parents, 1, backward_softmax);

    for (int r = 0; r < x->rows; r++) {
        float maxv = x->data[r * x->cols];
        float sum = 0.0f;
        for (int c = 1; c < x->cols; c++) {
            float v = x->data[r * x->cols + c];
            if (v > maxv) {
                maxv = v;
            }
        }
        for (int c = 0; c < x->cols; c++) {
            float e = expf(x->data[r * x->cols + c] - maxv);
            out->data[r * out->cols + c] = e;
            sum += e;
        }
        for (int c = 0; c < x->cols; c++) {
            out->data[r * out->cols + c] /= sum;
        }
    }
    return out;
}

static void backward_cross_entropy(Tensor *out) {
    const float eps = 1e-12f;
    Tensor *pred = out->parents[0];
    Tensor *target = out->parents[1];
    float g = out->grad[0] / (float)pred->rows;

    if (pred->requires_grad) {
        ensure_grad(pred);
        for (int i = 0; i < tensor_numel(pred); i++) {
            float p = pred->data[i] > eps ? pred->data[i] : eps;
            pred->grad[i] += g * (-target->data[i] / p);
        }
    }
    if (target->requires_grad) {
        ensure_grad(target);
        for (int i = 0; i < tensor_numel(target); i++) {
            float p = pred->data[i] > eps ? pred->data[i] : eps;
            target->grad[i] += g * (-logf(p));
        }
    }
}

static void backward_binary_cross_entropy(Tensor *out) {
    const float eps = 1e-6f;
    Tensor *pred = out->parents[0];
    Tensor *target = out->parents[1];
    int n = tensor_numel(pred);
    float g = out->grad[0] / (float)n;

    if (pred->requires_grad) {
        ensure_grad(pred);
        for (int i = 0; i < n; i++) {
            float p = pred->data[i];
            if (p < eps) p = eps;
            if (p > 1.0f - eps) p = 1.0f - eps;
            pred->grad[i] += g * ((1.0f - target->data[i]) / (1.0f - p) - target->data[i] / p);
        }
    }
    if (target->requires_grad) {
        ensure_grad(target);
        for (int i = 0; i < n; i++) {
            float p = pred->data[i];
            if (p < eps) p = eps;
            if (p > 1.0f - eps) p = 1.0f - eps;
            target->grad[i] += g * (logf(1.0f - p) - logf(p));
        }
    }
}

Tensor *tensor_binary_cross_entropy(Tensor *pred_probs, Tensor *target_probs) {
    const float eps = 1e-6f;
    int n;
    float acc = 0.0f;
    Tensor *parents[2] = {pred_probs, target_probs};
    int req;
    Tensor *out;

    ensure_same_shape(pred_probs, target_probs, "tensor_binary_cross_entropy: shape mismatch");
    req = infer_requires_grad(parents, 2);
    out = tensor_new_op(1, 1, req, OP_BINARY_CROSS_ENTROPY, parents, 2, backward_binary_cross_entropy);

    n = tensor_numel(pred_probs);
    for (int i = 0; i < n; i++) {
        float p = pred_probs->data[i];
        float y = target_probs->data[i];
        if (p < eps) p = eps;
        if (p > 1.0f - eps) p = 1.0f - eps;
        acc += -(y * logf(p) + (1.0f - y) * logf(1.0f - p));
    }
    out->data[0] = acc / (float)n;
    return out;
}

Tensor *tensor_cross_entropy(Tensor *pred_probs, Tensor *target_probs) {
    const float eps = 1e-12f;
    ensure_same_shape(pred_probs, target_probs, "tensor_cross_entropy: shape mismatch");
    Tensor *parents[2] = {pred_probs, target_probs};
    int req = infer_requires_grad(parents, 2);
    Tensor *out = tensor_new_op(1, 1, req, OP_CROSS_ENTROPY, parents, 2, backward_cross_entropy);
    float acc = 0.0f;

    for (int r = 0; r < pred_probs->rows; r++) {
        float row_loss = 0.0f;
        for (int c = 0; c < pred_probs->cols; c++) {
            int idx = r * pred_probs->cols + c;
            float p = pred_probs->data[idx] > eps ? pred_probs->data[idx] : eps;
            row_loss += -target_probs->data[idx] * logf(p);
        }
        acc += row_loss;
    }
    out->data[0] = acc / (float)pred_probs->rows;
    return out;
}

static void backward_mse(Tensor *out) {
    Tensor *pred = out->parents[0];
    Tensor *target = out->parents[1];
    int n = tensor_numel(pred);
    float g = out->grad[0];

    if (pred->requires_grad) {
        ensure_grad(pred);
        for (int i = 0; i < n; i++) {
            pred->grad[i] += g * (2.0f / (float)n) * (pred->data[i] - target->data[i]);
        }
    }

    if (target->requires_grad) {
        ensure_grad(target);
        for (int i = 0; i < n; i++) {
            target->grad[i] -= g * (2.0f / (float)n) * (pred->data[i] - target->data[i]);
        }
    }
}

Tensor *tensor_mse_loss(Tensor *pred, Tensor *target) {
    ensure_same_shape(pred, target, "tensor_mse_loss: shape mismatch");
    Tensor *parents[2] = {pred, target};
    int req = infer_requires_grad(parents, 2);
    Tensor *out = tensor_new_op(1, 1, req, OP_MSE, parents, 2, backward_mse);

    int n = tensor_numel(pred);
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = pred->data[i] - target->data[i];
        acc += d * d;
    }
    out->data[0] = acc / (float)n;
    return out;
}

Tensor *tensor_forward(Tensor *output) {
    return output;
}

static void build_topo(Tensor *t, TensorPtrList *visited, TensorPtrList *topo) {
    if (ptr_list_contains(visited, t)) {
        return;
    }
    ptr_list_push(visited, t);
    for (int i = 0; i < t->n_parents; i++) {
        build_topo(t->parents[i], visited, topo);
    }
    ptr_list_push(topo, t);
}

void tensor_backward(Tensor *loss) {
    TensorPtrList visited = {0};
    TensorPtrList topo = {0};
    build_topo(loss, &visited, &topo);

    for (int i = 0; i < topo.len; i++) {
        Tensor *t = topo.items[i];
        if (t->requires_grad) {
            ensure_grad(t);
            tensor_zero_grad(t);
        }
    }

    if (!loss->requires_grad) {
        free(visited.items);
        free(topo.items);
        return;
    }
    ensure_grad(loss);
    if (tensor_numel(loss) != 1) {
        die("tensor_backward: loss must be scalar");
    }
    loss->grad[0] = 1.0f;

    for (int i = topo.len - 1; i >= 0; i--) {
        Tensor *t = topo.items[i];
        if (t->backward_fn) {
            t->backward_fn(t);
        }
    }

    free(visited.items);
    free(topo.items);
}

void tensor_sgd_step(Tensor **params, size_t n_params, float lr) {
    for (size_t p = 0; p < n_params; p++) {
        Tensor *t = params[p];
        if (!t->requires_grad || !t->grad) {
            continue;
        }
        int n = tensor_numel(t);
        for (int i = 0; i < n; i++) {
            t->data[i] -= lr * t->grad[i];
        }
    }
}

void tensor_sgd_momentum_step(Tensor **params, Tensor **velocity, size_t n_params, float lr, float momentum) {
    for (size_t p = 0; p < n_params; p++) {
        Tensor *t = params[p];
        Tensor *v;
        int n;
        if (!t->requires_grad || !t->grad) {
            continue;
        }
        if (!velocity || !velocity[p]) {
            die("tensor_sgd_momentum_step: missing velocity tensor");
        }
        v = velocity[p];
        if (v->rows != t->rows || v->cols != t->cols) {
            die("tensor_sgd_momentum_step: velocity shape mismatch");
        }
        n = tensor_numel(t);
        for (int i = 0; i < n; i++) {
            v->data[i] = momentum * v->data[i] - lr * t->grad[i];
            t->data[i] += v->data[i];
        }
    }
}

void tensor_adam_step(Tensor **params,
                      Tensor **m1,
                      Tensor **m2,
                      size_t n_params,
                      const TensorAdamOptions *opt) {
    if (!opt) {
        die("tensor_adam_step: options must not be null");
    }
    if (opt->timestep <= 0) {
        die("tensor_adam_step: timestep must be >= 1");
    }
    for (size_t p = 0; p < n_params; p++) {
        Tensor *t = params[p];
        Tensor *m;
        Tensor *v;
        int n;
        float bias1;
        float bias2;

        if (!t->requires_grad || !t->grad) {
            continue;
        }
        if (!m1 || !m1[p] || !m2 || !m2[p]) {
            die("tensor_adam_step: missing moment tensor");
        }
        m = m1[p];
        v = m2[p];
        if (m->rows != t->rows || m->cols != t->cols || v->rows != t->rows || v->cols != t->cols) {
            die("tensor_adam_step: moment shape mismatch");
        }

        n = tensor_numel(t);
        /* Bias correction converts the moving averages m1/m2 into unbiased
         * estimates during the early Adam steps.
         */
        bias1 = 1.0f - powf(opt->beta1, (float)opt->timestep);
        bias2 = 1.0f - powf(opt->beta2, (float)opt->timestep);
        for (int i = 0; i < n; i++) {
            float g = t->grad[i];
            float m_hat;
            float v_hat;

            m->data[i] = opt->beta1 * m->data[i] + (1.0f - opt->beta1) * g;
            v->data[i] = opt->beta2 * v->data[i] + (1.0f - opt->beta2) * g * g;
            m_hat = m->data[i] / bias1;
            v_hat = v->data[i] / bias2;
            t->data[i] -= opt->lr * m_hat / (sqrtf(v_hat) + opt->eps);
        }
    }
}
