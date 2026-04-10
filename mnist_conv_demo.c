#include "tensor.h"

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

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
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

static Tensor *forward_logits(Tensor *x_col,
                              Tensor *Wc,
                              Tensor *bc,
                              Tensor *Wcls,
                              Tensor *bcls,
                              Tensor *pool,
                              Tensor **tmp_conv_lin,
                              Tensor **tmp_conv_bias,
                              Tensor **tmp_conv_act,
                              Tensor **tmp_patch_logits,
                              Tensor **tmp_patch_logits_bias) {
    Tensor *conv_lin = tensor_matmul(x_col, Wc);
    Tensor *conv_bias = tensor_add_bias(conv_lin, bc);
    Tensor *conv_act = tensor_relu(conv_bias);
    Tensor *patch_logits = tensor_matmul(conv_act, Wcls);
    Tensor *patch_logits_bias = tensor_add_bias(patch_logits, bcls);
    Tensor *logits = tensor_matmul(pool, patch_logits_bias);

    *tmp_conv_lin = conv_lin;
    *tmp_conv_bias = conv_bias;
    *tmp_conv_act = conv_act;
    *tmp_patch_logits = patch_logits;
    *tmp_patch_logits_bias = patch_logits_bias;
    return logits;
}

static void print_architecture_summary(FILE *out, int kh, int kw, int channels, int patches_per_image) {
    fprintf(out, "arch: input=28x28x1 im2col=%dx%d\n", kh, kw);
    fprintf(out, "arch: patch_dim=%d patches=%d\n", kh * kw, patches_per_image);
    fprintf(out, "arch: conv=matmul(%d->%d)+bias+relu\n", kh * kw, channels);
    fprintf(out, "arch: head=matmul(%d->10)+bias\n", channels);
    fprintf(out, "arch: pool=mean_over_patches loss=softmax_ce\n");
}

static float evaluate_accuracy(const MnistSet *ds,
                               int batch,
                               int kh,
                               int kw,
                               Tensor *Wc,
                               Tensor *bc,
                               Tensor *Wcls,
                               Tensor *bcls,
                               Tensor *pool,
                               float *xcol_buf) {
    int h = ds->rows;
    int w = ds->cols;
    int oh = h - kh + 1;
    int ow = w - kw + 1;
    int patches = oh * ow;
    int n_eval = (ds->n / batch) * batch;
    int correct = 0;

    for (int i = 0; i < n_eval; i += batch) {
        Tensor *xcol;
        Tensor *conv_lin;
        Tensor *conv_bias;
        Tensor *conv_act;
        Tensor *patch_logits;
        Tensor *patch_logits_bias;
        Tensor *logits;

        im2col_batch(ds->images + (size_t)i * MNIST_PIXELS, batch, h, w, kh, kw, xcol_buf);
        xcol = tensor_from_array(batch * patches, kh * kw, xcol_buf, 0);
        logits = forward_logits(
            xcol, Wc, bc, Wcls, bcls, pool, &conv_lin, &conv_bias, &conv_act, &patch_logits, &patch_logits_bias);

        for (int b = 0; b < batch; b++) {
            int pred = argmax_row(logits->data + (size_t)b * N_CLASSES, N_CLASSES);
            if (pred == (int)ds->labels[i + b]) {
                correct++;
            }
        }

        tensor_free(xcol);
        tensor_free(conv_lin);
        tensor_free(conv_bias);
        tensor_free(conv_act);
        tensor_free(patch_logits);
        tensor_free(patch_logits_bias);
        tensor_free(logits);
    }

    if (n_eval == 0) {
        return 0.0f;
    }
    return (float)correct / (float)n_eval;
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

    const int kh = 5;
    const int kw = 5;
    const int h = MNIST_ROWS;
    const int w = MNIST_COLS;
    const int oh = h - kh + 1;
    const int ow = w - kw + 1;
    const int patches = oh * ow;
    const int patch_dim = kh * kw;
    unsigned int seed = 1337U;

    MnistSet train;
    MnistSet test;
    Tensor *Wc;
    Tensor *bc;
    Tensor *Wcls;
    Tensor *bcls;
    Tensor *pool;
    Tensor *params[4];
    float *xcol_buf;
    FILE *log_file;

    if (batch <= 0 || epochs <= 0 || channels <= 0) {
        die("epochs, batch, channels must be > 0");
    }

    train = load_mnist(train_images, train_labels, max_train);
    test = load_mnist(test_images, test_labels, max_test);

    if (train.n < batch || test.n < batch) {
        die("dataset too small for chosen batch size");
    }

    Wc = tensor_create(patch_dim, channels, 1);
    bc = tensor_create(1, channels, 1);
    Wcls = tensor_create(channels, N_CLASSES, 1);
    bcls = tensor_create(1, N_CLASSES, 1);
    pool = build_patch_pool(batch, patches);

    tensor_fill_randn(Wc, 0.0f, 0.08f, &seed);
    tensor_fill_randn(Wcls, 0.0f, 0.08f, &seed);
    tensor_fill(bc, 0.0f);
    tensor_fill(bcls, 0.0f);

    params[0] = Wc;
    params[1] = bc;
    params[2] = Wcls;
    params[3] = bcls;

    xcol_buf = (float *)malloc(sizeof(float) * (size_t)batch * patches * patch_dim);
    if (!xcol_buf) {
        die("allocation failed for im2col buffer");
    }

    log_file = fopen(log_path, "w");
    if (!log_file) {
        die("failed to open log file");
    }
    fprintf(log_file, "# ");
    print_architecture_summary(log_file, kh, kw, channels, patches);
    fprintf(log_file, "# train=%d test=%d batch=%d epochs=%d lr=%.4f channels=%d\n",
            train.n, test.n, batch, epochs, lr, channels);
    fprintf(log_file, "epoch,train_loss,train_acc,train_error,test_acc,test_error\n");

    printf("MNIST conv-matmul demo\n");
    printf("train=%d test=%d batch=%d epochs=%d lr=%.4f channels=%d\n",
           train.n, test.n, batch, epochs, lr, channels);
    print_architecture_summary(stdout, kh, kw, channels, patches);
    printf("logging metrics to %s\n", log_path);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        int n_train = (train.n / batch) * batch;
        float loss_sum = 0.0f;
        int loss_count = 0;

        for (int i = 0; i < n_train; i += batch) {
            Tensor *xcol;
            Tensor *y;
            Tensor *conv_lin;
            Tensor *conv_bias;
            Tensor *conv_act;
            Tensor *patch_logits;
            Tensor *patch_logits_bias;
            Tensor *logits;
            Tensor *probs;
            Tensor *loss;

            im2col_batch(train.images + (size_t)i * MNIST_PIXELS, batch, h, w, kh, kw, xcol_buf);
            xcol = tensor_from_array(batch * patches, patch_dim, xcol_buf, 0);
            y = make_one_hot_batch(train.labels + i, batch);

            logits = forward_logits(
                xcol, Wc, bc, Wcls, bcls, pool, &conv_lin, &conv_bias, &conv_act, &patch_logits, &patch_logits_bias);
            probs = tensor_softmax(logits);
            loss = tensor_cross_entropy(probs, y);

            tensor_backward(loss);
            tensor_sgd_step(params, 4, lr);

            loss_sum += loss->data[0];
            loss_count++;

            tensor_free(xcol);
            tensor_free(y);
            tensor_free(conv_lin);
            tensor_free(conv_bias);
            tensor_free(conv_act);
            tensor_free(patch_logits);
            tensor_free(patch_logits_bias);
            tensor_free(logits);
            tensor_free(probs);
            tensor_free(loss);
        }

        {
            float train_acc = evaluate_accuracy(&train, batch, kh, kw, Wc, bc, Wcls, bcls, pool, xcol_buf);
            float test_acc = evaluate_accuracy(&test, batch, kh, kw, Wc, bc, Wcls, bcls, pool, xcol_buf);
            float avg_loss = loss_count > 0 ? loss_sum / (float)loss_count : 0.0f;
            float train_error = 1.0f - train_acc;
            float test_error = 1.0f - test_acc;

            printf("epoch %d/%d loss %.4f train_acc %.3f test_acc %.3f\n",
                   epoch, epochs, avg_loss, train_acc, test_acc);
            fprintf(log_file, "%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    epoch, avg_loss, train_acc, train_error, test_acc, test_error);
            fflush(log_file);
        }
    }

    fclose(log_file);
    free(xcol_buf);
    tensor_free(Wc);
    tensor_free(bc);
    tensor_free(Wcls);
    tensor_free(bcls);
    tensor_free(pool);
    free_mnist(&train);
    free_mnist(&test);

    return 0;
}
