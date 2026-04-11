#include "tensor.h"

/*
 * autoencoder.c
 *
 * Minimal MNIST autoencoder demo using nanotensor.
 * - Flattens each 28x28 digit into a 784-wide input vector
 * - Encodes through a hidden layer and low-dimensional latent bottleneck
 * - Decodes back to pixel space with a sigmoid reconstruction head
 * - Trains with mean-squared reconstruction loss
 *
 * Usage:
 *   ./autoencoder [--epochs=N] [--batch=N] [--hidden=N]
 *                 [--latent=N] [--lr=FLOAT] [--log=PATH]
 *                 [--recon=PATH]
 */
#include <sys/time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MNIST_ROWS 28
#define MNIST_COLS 28
#define MNIST_PIXELS (MNIST_ROWS * MNIST_COLS)

typedef struct {
    int n;
    int rows;
    int cols;
    float *images;
    unsigned char *labels;
} MnistSet;

/* Architecture:
 * input [batch, 784]
 * -> encoder hidden [batch, hidden] with relu
 * -> latent [batch, latent] with relu
 * -> decoder hidden [batch, hidden] with relu
 * -> reconstruction logits [batch, 784]
 * -> sigmoid reconstruction in [0, 1]
 */
typedef struct {
    int batch;
    int hidden;
    int latent;
    Tensor *W_enc1;
    Tensor *b_enc1;
    Tensor *W_latent;
    Tensor *b_latent;
    Tensor *W_dec1;
    Tensor *b_dec1;
    Tensor *W_out;
    Tensor *b_out;
    Tensor *params[8];
    Tensor *velocity[8];
    Tensor *tmp_enc1_lin;
    Tensor *tmp_enc1_bias;
    Tensor *tmp_enc1_act;
    Tensor *tmp_latent_lin;
    Tensor *tmp_latent_bias;
    Tensor *tmp_latent_act;
    Tensor *tmp_dec1_lin;
    Tensor *tmp_dec1_bias;
    Tensor *tmp_dec1_act;
    Tensor *tmp_out_lin;
    Tensor *tmp_out_bias;
    Tensor *tmp_recon;
} MnistAutoencoder;

typedef struct {
    int epochs;
    int batch;
    int hidden;
    int latent;
    float lr;
    const char *log_path;
    const char *recon_path;
} AutoencoderOptions;

static void autoencoder_init(MnistAutoencoder *model, const AutoencoderOptions *opt, unsigned int *seed);
static void autoencoder_free(MnistAutoencoder *model);
static void autoencoder_clear_forward_cache(MnistAutoencoder *model);
static Tensor *autoencoder_forward(MnistAutoencoder *model, Tensor *x);
static float autoencoder_train_epoch(MnistAutoencoder *model,
                                     const MnistSet *train,
                                     const int *indices,
                                     float *batch_images,
                                     float lr);
static float autoencoder_eval(MnistAutoencoder *model, const MnistSet *ds, float *batch_images);
static void autoencoder_save_reconstructions(MnistAutoencoder *model,
                                             const MnistSet *ds,
                                             const char *path,
                                             int n_examples);

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

static float rand_uniform(unsigned int *seed) {
    *seed = (1103515245U * (*seed) + 12345U);
    return (float)((*seed) & 0x7fffffffU) / 2147483648.0f;
}

static void print_usage(const char *prog) {
    printf("usage: %s [options]\n", prog);
    printf("  --epochs=N\n");
    printf("  --batch=N\n");
    printf("  --hidden=N\n");
    printf("  --latent=N\n");
    printf("  --lr=FLOAT\n");
    printf("  --log=PATH\n");
    printf("  --recon=PATH\n");
}

static void parse_args(int argc, char **argv, AutoencoderOptions *opt) {
    opt->epochs = 10;
    opt->batch = 32;
    opt->hidden = 128;
    opt->latent = 32;
    opt->lr = 0.01f;
    opt->log_path = "out/autoencoder_training_log.csv";
    opt->recon_path = "out/autoencoder_recon.csv";

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else if (sscanf(arg, "--epochs=%d", &opt->epochs) == 1) {
            continue;
        } else if (sscanf(arg, "--batch=%d", &opt->batch) == 1) {
            continue;
        } else if (sscanf(arg, "--hidden=%d", &opt->hidden) == 1) {
            continue;
        } else if (sscanf(arg, "--latent=%d", &opt->latent) == 1) {
            continue;
        } else if (sscanf(arg, "--lr=%f", &opt->lr) == 1) {
            continue;
        } else if (strncmp(arg, "--log=", 6) == 0) {
            opt->log_path = argv[i] + 6;
        } else if (strncmp(arg, "--recon=", 8) == 0) {
            opt->recon_path = argv[i] + 8;
        } else {
            die("unknown option");
        }
    }
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
    if (!ds) {
        return;
    }
    free(ds->images);
    free(ds->labels);
    ds->images = NULL;
    ds->labels = NULL;
    ds->n = 0;
}

static void shuffle_indices(int *indices, int n, unsigned int *seed) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(rand_uniform(seed) * (float)(i + 1));
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

static void gather_batch_images(const MnistSet *ds,
                                const int *indices,
                                int start,
                                int batch,
                                float *batch_images) {
    for (int b = 0; b < batch; b++) {
        int idx = indices[start + b];
        memcpy(batch_images + (size_t)b * MNIST_PIXELS,
               ds->images + (size_t)idx * MNIST_PIXELS,
               sizeof(float) * MNIST_PIXELS);
    }
}

static void print_architecture_summary(FILE *out, const MnistAutoencoder *model) {
    fprintf(out, "arch: input=784 hidden=%d latent=%d\n", model->hidden, model->latent);
    fprintf(out, "arch: encoder=784->%d->%d relu\n", model->hidden, model->latent);
    fprintf(out, "arch: decoder=%d->%d->784 relu+sigmoid\n", model->latent, model->hidden);
    fprintf(out, "arch: loss=mse over pixel reconstructions\n");
}

static void autoencoder_clear_forward_cache(MnistAutoencoder *model) {
    if (!model) {
        return;
    }
    tensor_free(model->tmp_enc1_lin);
    tensor_free(model->tmp_enc1_bias);
    tensor_free(model->tmp_enc1_act);
    tensor_free(model->tmp_latent_lin);
    tensor_free(model->tmp_latent_bias);
    tensor_free(model->tmp_latent_act);
    tensor_free(model->tmp_dec1_lin);
    tensor_free(model->tmp_dec1_bias);
    tensor_free(model->tmp_dec1_act);
    tensor_free(model->tmp_out_lin);
    tensor_free(model->tmp_out_bias);
    tensor_free(model->tmp_recon);
    model->tmp_enc1_lin = NULL;
    model->tmp_enc1_bias = NULL;
    model->tmp_enc1_act = NULL;
    model->tmp_latent_lin = NULL;
    model->tmp_latent_bias = NULL;
    model->tmp_latent_act = NULL;
    model->tmp_dec1_lin = NULL;
    model->tmp_dec1_bias = NULL;
    model->tmp_dec1_act = NULL;
    model->tmp_out_lin = NULL;
    model->tmp_out_bias = NULL;
    model->tmp_recon = NULL;
}

static void autoencoder_init(MnistAutoencoder *model, const AutoencoderOptions *opt, unsigned int *seed) {
    memset(model, 0, sizeof(*model));
    model->batch = opt->batch;
    model->hidden = opt->hidden;
    model->latent = opt->latent;

    model->W_enc1 = tensor_create(MNIST_PIXELS, opt->hidden, 1);
    model->b_enc1 = tensor_create(1, opt->hidden, 1);
    model->W_latent = tensor_create(opt->hidden, opt->latent, 1);
    model->b_latent = tensor_create(1, opt->latent, 1);
    model->W_dec1 = tensor_create(opt->latent, opt->hidden, 1);
    model->b_dec1 = tensor_create(1, opt->hidden, 1);
    model->W_out = tensor_create(opt->hidden, MNIST_PIXELS, 1);
    model->b_out = tensor_create(1, MNIST_PIXELS, 1);

    if (!model->W_enc1 || !model->b_enc1 || !model->W_latent || !model->b_latent || !model->W_dec1 ||
        !model->b_dec1 || !model->W_out || !model->b_out) {
        die("autoencoder parameter allocation failed");
    }

    tensor_fill_randn(model->W_enc1, 0.0f, 0.05f, seed);
    tensor_fill_randn(model->W_latent, 0.0f, 0.05f, seed);
    tensor_fill_randn(model->W_dec1, 0.0f, 0.05f, seed);
    tensor_fill_randn(model->W_out, 0.0f, 0.05f, seed);
    tensor_fill(model->b_enc1, 0.0f);
    tensor_fill(model->b_latent, 0.0f);
    tensor_fill(model->b_dec1, 0.0f);
    tensor_fill(model->b_out, 0.0f);

    model->params[0] = model->W_enc1;
    model->params[1] = model->b_enc1;
    model->params[2] = model->W_latent;
    model->params[3] = model->b_latent;
    model->params[4] = model->W_dec1;
    model->params[5] = model->b_dec1;
    model->params[6] = model->W_out;
    model->params[7] = model->b_out;

    for (int i = 0; i < 8; i++) {
        model->velocity[i] = tensor_create(model->params[i]->rows, model->params[i]->cols, 0);
        if (!model->velocity[i]) {
            die("autoencoder velocity allocation failed");
        }
    }
}

static void autoencoder_free(MnistAutoencoder *model) {
    if (!model) {
        return;
    }
    autoencoder_clear_forward_cache(model);
    for (int i = 0; i < 8; i++) {
        tensor_free(model->params[i]);
        tensor_free(model->velocity[i]);
    }
    memset(model, 0, sizeof(*model));
}

static Tensor *autoencoder_forward(MnistAutoencoder *model, Tensor *x) {
    model->tmp_enc1_lin = tensor_matmul(x, model->W_enc1);
    model->tmp_enc1_bias = tensor_add_bias(model->tmp_enc1_lin, model->b_enc1);
    model->tmp_enc1_act = tensor_relu(model->tmp_enc1_bias);
    model->tmp_latent_lin = tensor_matmul(model->tmp_enc1_act, model->W_latent);
    model->tmp_latent_bias = tensor_add_bias(model->tmp_latent_lin, model->b_latent);
    model->tmp_latent_act = tensor_relu(model->tmp_latent_bias);
    model->tmp_dec1_lin = tensor_matmul(model->tmp_latent_act, model->W_dec1);
    model->tmp_dec1_bias = tensor_add_bias(model->tmp_dec1_lin, model->b_dec1);
    model->tmp_dec1_act = tensor_relu(model->tmp_dec1_bias);
    model->tmp_out_lin = tensor_matmul(model->tmp_dec1_act, model->W_out);
    model->tmp_out_bias = tensor_add_bias(model->tmp_out_lin, model->b_out);
    model->tmp_recon = tensor_sigmoid(model->tmp_out_bias);
    return model->tmp_recon;
}

static float autoencoder_train_epoch(MnistAutoencoder *model,
                                     const MnistSet *train,
                                     const int *indices,
                                     float *batch_images,
                                     float lr) {
    int n_train = (train->n / model->batch) * model->batch;
    float loss_sum = 0.0f;
    int loss_count = 0;

    for (int i = 0; i < n_train; i += model->batch) {
        Tensor *x;
        Tensor *recon;
        Tensor *loss;

        gather_batch_images(train, indices, i, model->batch, batch_images);
        x = tensor_from_array(model->batch, MNIST_PIXELS, batch_images, 0);
        recon = autoencoder_forward(model, x);
        loss = tensor_mse_loss(recon, x);

        tensor_backward(loss);
        tensor_sgd_momentum_step(model->params, model->velocity, 8, lr, 0.9f);

        loss_sum += loss->data[0];
        loss_count++;

        tensor_free(x);
        tensor_free(loss);
        autoencoder_clear_forward_cache(model);
    }

    return loss_count > 0 ? loss_sum / (float)loss_count : 0.0f;
}

static float autoencoder_eval(MnistAutoencoder *model, const MnistSet *ds, float *batch_images) {
    int n_eval = (ds->n / model->batch) * model->batch;
    float loss_sum = 0.0f;
    int loss_count = 0;

    for (int i = 0; i < n_eval; i += model->batch) {
        Tensor *x;
        Tensor *recon;
        Tensor *loss;

        memcpy(batch_images,
               ds->images + (size_t)i * MNIST_PIXELS,
               sizeof(float) * (size_t)model->batch * MNIST_PIXELS);
        x = tensor_from_array(model->batch, MNIST_PIXELS, batch_images, 0);
        recon = autoencoder_forward(model, x);
        loss = tensor_mse_loss(recon, x);

        loss_sum += loss->data[0];
        loss_count++;

        tensor_free(x);
        tensor_free(loss);
        autoencoder_clear_forward_cache(model);
    }

    return loss_count > 0 ? loss_sum / (float)loss_count : 0.0f;
}

static void autoencoder_save_reconstructions(MnistAutoencoder *model,
                                             const MnistSet *ds,
                                             const char *path,
                                             int n_examples) {
    FILE *f;
    Tensor *x;
    Tensor *recon;
    int count = n_examples;

    if (count > model->batch) {
        count = model->batch;
    }
    if (count > ds->n) {
        count = ds->n;
    }
    if (count <= 0) {
        return;
    }

    f = fopen(path, "w");
    if (!f) {
        die("failed to open reconstruction output");
    }

    x = tensor_from_array(model->batch, MNIST_PIXELS, ds->images, 0);
    recon = autoencoder_forward(model, x);

    fprintf(f, "kind,index,pixel,value\n");
    for (int i = 0; i < count; i++) {
        for (int p = 0; p < MNIST_PIXELS; p++) {
            fprintf(f, "input,%d,%d,%.6f\n", i, p, ds->images[(size_t)i * MNIST_PIXELS + p]);
        }
        for (int p = 0; p < MNIST_PIXELS; p++) {
            fprintf(f, "recon,%d,%d,%.6f\n", i, p, recon->data[(size_t)i * MNIST_PIXELS + p]);
        }
    }

    fclose(f);
    tensor_free(x);
    autoencoder_clear_forward_cache(model);
}

int main(int argc, char **argv) {
    const char *train_images = "data/mnist/train-images-idx3-ubyte";
    const char *train_labels = "data/mnist/train-labels-idx1-ubyte";
    const char *test_images = "data/mnist/t10k-images-idx3-ubyte";
    const char *test_labels = "data/mnist/t10k-labels-idx1-ubyte";
    const int train_limit = 10000;
    const int test_limit = 2000;

    AutoencoderOptions opt;
    MnistSet train;
    MnistSet test;
    MnistAutoencoder model;
    FILE *logf;
    int *indices;
    float *batch_images;
    unsigned int seed = 1337u;
    double start_time;

    parse_args(argc, argv, &opt);
    if (opt.epochs <= 0) opt.epochs = 10;
    if (opt.batch <= 0) opt.batch = 32;
    if (opt.hidden <= 0) opt.hidden = 128;
    if (opt.latent <= 0) opt.latent = 32;
    if (opt.lr <= 0.0f) opt.lr = 0.01f;

    train = load_mnist(train_images, train_labels, train_limit);
    test = load_mnist(test_images, test_labels, test_limit);

    if (train.n < opt.batch || test.n < opt.batch) {
        die("batch size larger than loaded MNIST subset");
    }

    autoencoder_init(&model, &opt, &seed);
    indices = (int *)malloc(sizeof(int) * (size_t)train.n);
    batch_images = (float *)malloc(sizeof(float) * (size_t)opt.batch * MNIST_PIXELS);
    if (!indices || !batch_images) {
        die("allocation failed");
    }
    for (int i = 0; i < train.n; i++) {
        indices[i] = i;
    }

    printf("loaded MNIST train=%d test=%d batch=%d\n", train.n, test.n, opt.batch);
    print_architecture_summary(stdout, &model);
    printf("opt: epochs=%d lr=%.4f recon=%s\n", opt.epochs, opt.lr, opt.recon_path);

    logf = fopen(opt.log_path, "w");
    if (!logf) {
        die("failed to open log file");
    }
    fprintf(logf, "epoch,train_loss,eval_loss,elapsed_s\n");

    start_time = now_seconds();
    for (int epoch = 1; epoch <= opt.epochs; epoch++) {
        float train_loss;
        float eval_loss;
        double elapsed;

        shuffle_indices(indices, train.n, &seed);
        train_loss = autoencoder_train_epoch(&model, &train, indices, batch_images, opt.lr);
        eval_loss = autoencoder_eval(&model, &test, batch_images);
        elapsed = now_seconds() - start_time;

        printf("epoch %3d train_loss %.6f eval_loss %.6f elapsed %.1fs\n",
               epoch,
               train_loss,
               eval_loss,
               elapsed);
        fprintf(logf, "%d,%.6f,%.6f,%.3f\n", epoch, train_loss, eval_loss, elapsed);
        fflush(logf);
    }

    fclose(logf);
    autoencoder_save_reconstructions(&model, &test, opt.recon_path, 8);
    printf("saved reconstructions: %s\n", opt.recon_path);
    autoencoder_free(&model);
    free(indices);
    free(batch_images);
    free_mnist(&train);
    free_mnist(&test);
    return 0;
}
