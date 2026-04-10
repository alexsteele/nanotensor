#include "tensor.h"

#include <sys/time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MNIST_ROWS 28
#define MNIST_COLS 28
#define MNIST_PIXELS (MNIST_ROWS * MNIST_COLS)
#define N_CLASSES 10

typedef struct {
    int n;
    int rows;
    int cols;
    float *images;      /* [n, rows*cols], normalized to [0, 1] */
    unsigned char *labels;
} MnistSet;

/* Architecture:
 * input [batch, 28x28]
 * -> im2col patches [batch * 24 * 24, 25]
 * -> conv projection [batch * 24 * 24, channels]
 * -> relu
 * -> classifier head [batch * 24 * 24, 10]
 * -> mean pool over patches [batch, 10]
 * -> logits; softmax is applied only for training loss/probabilities
 */
typedef struct {
    int kh;
    int kw;
    int h;
    int w;
    int oh;
    int ow;
    int patches;
    int patch_dim;
    int channels;
    int batch;
    Tensor *Wc;
    Tensor *bc;
    Tensor *Wcls;
    Tensor *bcls;
    Tensor *pool;
    Tensor *params[4];
    float *xcol_buf;
    Tensor *tmp_conv_lin;
    Tensor *tmp_conv_bias;
    Tensor *tmp_conv_act;
    Tensor *tmp_patch_logits;
    Tensor *tmp_patch_logits_bias;
} MnistConvModel;

static void mnist_model_init(MnistConvModel *model, int batch, int channels, unsigned int *seed);
static void mnist_model_free(MnistConvModel *model);
static float mnist_model_train_epoch(MnistConvModel *model, const MnistSet *train, float lr);
static Tensor *mnist_model_forward(MnistConvModel *model, Tensor *xcol);
static void mnist_model_clear_forward_cache(MnistConvModel *model);
static float evaluate_accuracy(const MnistSet *ds, MnistConvModel *model);

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static double now_seconds(void) {
    struct timeval tv;
    if (gettimeofday(&tv, NULL) != 0) {
        die("gettimeofday failed");
    }
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static uint32_t read_u32_be(FILE *f) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) {
        die("failed to read u32");
    }
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | (uint32_t)b[3];
}

static MnistSet load_mnist(const char *images_path, const char *labels_path, int max_samples) {
    FILE *fi = fopen(images_path, "rb");
    FILE *fl = fopen(labels_path, "rb");
    MnistSet ds = {0};
    uint32_t img_magic;
    uint32_t lbl_magic;
    uint32_t n_images;
    uint32_t n_labels;
    uint32_t rows;
    uint32_t cols;
    unsigned char *buf;

    if (!fi || !fl) {
        die("failed to open MNIST files");
    }

    img_magic = read_u32_be(fi);
    n_images = read_u32_be(fi);
    rows = read_u32_be(fi);
    cols = read_u32_be(fi);

    lbl_magic = read_u32_be(fl);
    n_labels = read_u32_be(fl);

    if (img_magic != 2051U || lbl_magic != 2049U) {
        die("invalid MNIST magic (expecting raw IDX files, not .gz)");
    }
    if (rows != MNIST_ROWS || cols != MNIST_COLS) {
        die("unexpected MNIST image shape");
    }
    if (n_images != n_labels) {
        die("MNIST images/labels length mismatch");
    }

    ds.n = (int)n_images;
    if (max_samples > 0 && max_samples < ds.n) {
        ds.n = max_samples;
    }
    ds.rows = (int)rows;
    ds.cols = (int)cols;
    ds.images = (float *)malloc(sizeof(float) * (size_t)ds.n * MNIST_PIXELS);
    ds.labels = (unsigned char *)malloc((size_t)ds.n);
    buf = (unsigned char *)malloc((size_t)ds.n * MNIST_PIXELS);

    if (!ds.images || !ds.labels || !buf) {
        die("allocation failed loading MNIST");
    }

    if (fread(buf, 1, (size_t)ds.n * MNIST_PIXELS, fi) != (size_t)ds.n * MNIST_PIXELS) {
        die("failed to read MNIST images");
    }
    if (fread(ds.labels, 1, (size_t)ds.n, fl) != (size_t)ds.n) {
        die("failed to read MNIST labels");
    }

    for (int i = 0; i < ds.n * MNIST_PIXELS; i++) {
        ds.images[i] = (float)buf[i] / 255.0f;
    }

    free(buf);
    fclose(fi);
    fclose(fl);
    return ds;
}

static void free_mnist(MnistSet *ds) {
    if (!ds) return;
    free(ds->images);
    free(ds->labels);
    ds->images = NULL;
    ds->labels = NULL;
    ds->n = 0;
}

static int argmax_row(const float *row, int n) {
    int best = 0;
    float bestv = row[0];
    for (int i = 1; i < n; i++) {
        if (row[i] > bestv) {
            bestv = row[i];
            best = i;
        }
    }
    return best;
}

static Tensor *make_one_hot_batch(const unsigned char *labels, int batch_size) {
    Tensor *t = tensor_create(batch_size, N_CLASSES, 0);
    tensor_fill(t, 0.0f);
    for (int i = 0; i < batch_size; i++) {
        int y = (int)labels[i];
        if (y < 0 || y >= N_CLASSES) {
            die("invalid label in MNIST");
        }
        t->data[i * N_CLASSES + y] = 1.0f;
    }
    return t;
}

static Tensor *build_patch_pool(int batch, int patches_per_image) {
    Tensor *pool = tensor_create(batch, batch * patches_per_image, 0);
    float scale = 1.0f / (float)patches_per_image;
    tensor_fill(pool, 0.0f);
    for (int b = 0; b < batch; b++) {
        int base = b * patches_per_image;
        for (int p = 0; p < patches_per_image; p++) {
            pool->data[b * pool->cols + base + p] = scale;
        }
    }
    return pool;
}

/* Extract every sliding kh x kw patch into one row so the conv-like
 * stage can be implemented as a matrix multiply over all patches. */
static void im2col_batch(const float *images, int batch, int h, int w, int kh, int kw, float *out) {
    int oh = h - kh + 1;
    int ow = w - kw + 1;
    int patch = kh * kw;

    for (int b = 0; b < batch; b++) {
        const float *img = images + (size_t)b * h * w;
        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                int row = (b * oh + y) * ow + x;
                float *dst = out + (size_t)row * patch;
                int k = 0;
                for (int ky = 0; ky < kh; ky++) {
                    for (int kx = 0; kx < kw; kx++) {
                        dst[k++] = img[(y + ky) * w + (x + kx)];
                    }
                }
            }
        }
    }
}

static void mnist_model_init(MnistConvModel *model, int batch, int channels, unsigned int *seed) {
    if (!model) {
        die("mnist_model_init: model is null");
    }

    memset(model, 0, sizeof(*model));
    model->kh = 5;
    model->kw = 5;
    model->h = MNIST_ROWS;
    model->w = MNIST_COLS;
    model->oh = model->h - model->kh + 1;
    model->ow = model->w - model->kw + 1;
    model->patches = model->oh * model->ow;
    model->patch_dim = model->kh * model->kw;
    model->channels = channels;
    model->batch = batch;

    model->Wc = tensor_create(model->patch_dim, channels, 1);
    model->bc = tensor_create(1, channels, 1);
    model->Wcls = tensor_create(channels, N_CLASSES, 1);
    model->bcls = tensor_create(1, N_CLASSES, 1);
    model->pool = build_patch_pool(batch, model->patches);

    tensor_fill_randn(model->Wc, 0.0f, 0.08f, seed);
    tensor_fill_randn(model->Wcls, 0.0f, 0.08f, seed);
    tensor_fill(model->bc, 0.0f);
    tensor_fill(model->bcls, 0.0f);

    model->params[0] = model->Wc;
    model->params[1] = model->bc;
    model->params[2] = model->Wcls;
    model->params[3] = model->bcls;

    model->xcol_buf = (float *)malloc(sizeof(float) * (size_t)batch * model->patches * model->patch_dim);
    if (!model->xcol_buf) {
        die("allocation failed for im2col buffer");
    }
}

static void mnist_model_clear_forward_cache(MnistConvModel *model) {
    if (!model) {
        return;
    }
    tensor_free(model->tmp_conv_lin);
    tensor_free(model->tmp_conv_bias);
    tensor_free(model->tmp_conv_act);
    tensor_free(model->tmp_patch_logits);
    tensor_free(model->tmp_patch_logits_bias);
    model->tmp_conv_lin = NULL;
    model->tmp_conv_bias = NULL;
    model->tmp_conv_act = NULL;
    model->tmp_patch_logits = NULL;
    model->tmp_patch_logits_bias = NULL;
}

static void mnist_model_free(MnistConvModel *model) {
    if (!model) {
        return;
    }
    mnist_model_clear_forward_cache(model);
    free(model->xcol_buf);
    tensor_free(model->Wc);
    tensor_free(model->bc);
    tensor_free(model->Wcls);
    tensor_free(model->bcls);
    tensor_free(model->pool);
    memset(model, 0, sizeof(*model));
}

static Tensor *forward_logits(MnistConvModel *model, Tensor *x_col) {
    Tensor *conv_lin = tensor_matmul(x_col, model->Wc);
    Tensor *conv_bias = tensor_add_bias(conv_lin, model->bc);
    Tensor *conv_act = tensor_relu(conv_bias);
    Tensor *patch_logits = tensor_matmul(conv_act, model->Wcls);
    Tensor *patch_logits_bias = tensor_add_bias(patch_logits, model->bcls);
    Tensor *logits = tensor_matmul(model->pool, patch_logits_bias);

    model->tmp_conv_lin = conv_lin;
    model->tmp_conv_bias = conv_bias;
    model->tmp_conv_act = conv_act;
    model->tmp_patch_logits = patch_logits;
    model->tmp_patch_logits_bias = patch_logits_bias;
    return logits;
}

static Tensor *mnist_model_forward(MnistConvModel *model, Tensor *xcol) {
    return forward_logits(model, xcol);
}

static void print_architecture_summary(FILE *out,
                                       const char *prefix,
                                       int kh,
                                       int kw,
                                       int channels,
                                       int patches_per_image) {
    const char *p = prefix ? prefix : "";
    fprintf(out, "%sarch: input=28x28x1 im2col=%dx%d\n", p, kh, kw);
    fprintf(out, "%sarch: patch_dim=%d patches=%d\n", p, kh * kw, patches_per_image);
    fprintf(out, "%sarch: conv=matmul(%d->%d)+bias+relu\n", p, kh * kw, channels);
    fprintf(out, "%sarch: head=matmul(%d->10)+bias\n", p, channels);
    fprintf(out, "%sarch: pool=mean_over_patches loss=softmax_ce\n", p);
}

static float evaluate_accuracy(const MnistSet *ds,
                               MnistConvModel *model) {
    int n_eval = (ds->n / model->batch) * model->batch;
    int correct = 0;

    for (int i = 0; i < n_eval; i += model->batch) {
        Tensor *xcol;
        Tensor *logits;

        im2col_batch(ds->images + (size_t)i * MNIST_PIXELS,
                     model->batch,
                     ds->rows,
                     ds->cols,
                     model->kh,
                     model->kw,
                     model->xcol_buf);
        xcol = tensor_from_array(model->batch * model->patches, model->patch_dim, model->xcol_buf, 0);
        logits = mnist_model_forward(model, xcol);

        for (int b = 0; b < model->batch; b++) {
            int pred = argmax_row(logits->data + (size_t)b * N_CLASSES, N_CLASSES);
            if (pred == (int)ds->labels[i + b]) {
                correct++;
            }
        }

        tensor_free(xcol);
        tensor_free(logits);
        mnist_model_clear_forward_cache(model);
    }

    if (n_eval == 0) {
        return 0.0f;
    }
    return (float)correct / (float)n_eval;
}

static float mnist_model_train_epoch(MnistConvModel *model, const MnistSet *train, float lr) {
    int n_train = (train->n / model->batch) * model->batch;
    float loss_sum = 0.0f;
    int loss_count = 0;

    for (int i = 0; i < n_train; i += model->batch) {
        Tensor *xcol;
        Tensor *y;
        Tensor *logits;
        Tensor *probs;
        Tensor *loss;

        im2col_batch(train->images + (size_t)i * MNIST_PIXELS,
                     model->batch,
                     train->rows,
                     train->cols,
                     model->kh,
                     model->kw,
                     model->xcol_buf);
        xcol = tensor_from_array(model->batch * model->patches, model->patch_dim, model->xcol_buf, 0);
        y = make_one_hot_batch(train->labels + i, model->batch);

        logits = mnist_model_forward(model, xcol);
        probs = tensor_softmax(logits);
        loss = tensor_cross_entropy(probs, y);

        tensor_backward(loss);
        tensor_sgd_step(model->params, 4, lr);

        loss_sum += loss->data[0];
        loss_count++;

        tensor_free(xcol);
        tensor_free(y);
        tensor_free(logits);
        tensor_free(probs);
        tensor_free(loss);
        mnist_model_clear_forward_cache(model);
    }

    return loss_count > 0 ? loss_sum / (float)loss_count : 0.0f;
}

int main(int argc, char **argv) {
    const char *train_images = argc > 1 ? argv[1] : "data/mnist/train-images-idx3-ubyte";
    const char *train_labels = argc > 2 ? argv[2] : "data/mnist/train-labels-idx1-ubyte";
    const char *test_images = argc > 3 ? argv[3] : "data/mnist/t10k-images-idx3-ubyte";
    const char *test_labels = argc > 4 ? argv[4] : "data/mnist/t10k-labels-idx1-ubyte";
    int epochs = argc > 5 ? atoi(argv[5]) : 5;
    int batch = argc > 6 ? atoi(argv[6]) : 32;
    float lr = argc > 7 ? (float)atof(argv[7]) : 0.03f;
    int channels = argc > 8 ? atoi(argv[8]) : 8;
    int max_train = argc > 9 ? atoi(argv[9]) : 10000;
    int max_test = argc > 10 ? atoi(argv[10]) : 2000;
    const char *log_path = argc > 11 ? argv[11] : "mnist_training_log.csv";

    unsigned int seed = 1337U;

    MnistSet train;
    MnistSet test;
    MnistConvModel model;
    FILE *log_file;

    if (batch <= 0 || epochs <= 0 || channels <= 0) {
        die("epochs, batch, channels must be > 0");
    }

    train = load_mnist(train_images, train_labels, max_train);
    test = load_mnist(test_images, test_labels, max_test);

    if (train.n < batch || test.n < batch) {
        die("dataset too small for chosen batch size");
    }

    mnist_model_init(&model, batch, channels, &seed);

    log_file = fopen(log_path, "w");
    if (!log_file) {
        die("failed to open log file");
    }
    print_architecture_summary(log_file, "# ", model.kh, model.kw, model.channels, model.patches);
    fprintf(log_file, "# train=%d test=%d batch=%d epochs=%d lr=%.4f channels=%d\n",
            train.n, test.n, batch, epochs, lr, channels);
    fprintf(log_file, "epoch,train_loss,train_acc,train_error,test_acc,test_error\n");

    printf("MNIST conv-matmul demo\n");
    printf("train=%d test=%d batch=%d epochs=%d lr=%.4f channels=%d\n",
           train.n, test.n, batch, epochs, lr, channels);
    print_architecture_summary(stdout, NULL, model.kh, model.kw, model.channels, model.patches);
    printf("logging metrics to %s\n", log_path);

    {
        double start_time = now_seconds();

    for (int epoch = 1; epoch <= epochs; epoch++) {
        {
            float avg_loss = mnist_model_train_epoch(&model, &train, lr);
            float train_acc = evaluate_accuracy(&train, &model);
            float test_acc = evaluate_accuracy(&test, &model);
            float train_error = 1.0f - train_acc;
            float test_error = 1.0f - test_acc;
            double elapsed = now_seconds() - start_time;

            printf("epoch %d/%d loss %.4f train_acc %.3f test_acc %.3f elapsed %.2fs\n",
                   epoch, epochs, avg_loss, train_acc, test_acc, elapsed);
            fprintf(log_file, "%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    epoch, avg_loss, train_acc, train_error, test_acc, test_error);
            fflush(log_file);
        }
    }
    }

    fclose(log_file);
    mnist_model_free(&model);
    free_mnist(&train);
    free_mnist(&test);

    return 0;
}
