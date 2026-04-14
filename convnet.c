#include "mnist.h"
#include "patch.h"
#include "tensor.h"

/*
 * convnet.c
 *
 * Minimal MNIST conv-like classifier demo using nanotensor.
 * - Extracts 5x5 image patches with im2col
 * - Projects patches through a small relu feature stack and class head
 * - Mean-pools patch logits per image and trains with softmax cross entropy
 *
 * Usage:
 *   ./mnist_conv_demo [--epochs=N] [--batch=N] [--channels=N]
 *                     [--lr=FLOAT] [--momentum=FLOAT] [--log=PATH]
 */
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_CLASSES 10

typedef enum {
    OPT_SGD = 0,
    OPT_MOMENTUM,
    OPT_ADAM
} OptimizerKind;

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
    PatchLayout patch_layout;
    int channels;
    int batch;
    Tensor *Wc;
    Tensor *bc;
    Tensor *Wcls;
    Tensor *bcls;
    Tensor *params[4];
    Tensor *velocity[4];
    Tensor *adam_m1[4];
    Tensor *adam_m2[4];
    int adam_step;
    PatchBatch patch_batch;
} MnistConvModel;

static void mnist_model_init(MnistConvModel *model, int batch, int channels, unsigned int *seed);
static void mnist_model_free(MnistConvModel *model);
static float mnist_model_train_epoch(MnistConvModel *model,
                                     const MnistSet *train,
                                     const int *indices,
                                     float *batch_images,
                                     unsigned char *batch_labels,
                                     float lr,
                                     OptimizerKind opt_kind,
                                     float momentum);
static Tensor *mnist_model_forward(TensorList *temps, MnistConvModel *model, Tensor *xcol);
static float evaluate_accuracy(const MnistSet *ds, MnistConvModel *model);

typedef struct {
    int epochs;
    int batch;
    int channels;
    int train_limit;
    int test_limit;
    OptimizerKind opt_kind;
    float lr;
    float momentum;
    const char *log_path;
} MnistOptions;

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

static void print_usage(const char *prog) {
    printf("usage: %s [options]\n", prog);
    printf("  --epochs=N\n");
    printf("  --batch=N\n");
    printf("  --channels=N\n");
    printf("  --train-limit=N\n");
    printf("  --test-limit=N\n");
    printf("  --opt=sgd|momentum|adam\n");
    printf("  --lr=FLOAT\n");
    printf("  --momentum=FLOAT\n");
    printf("  --log=PATH\n");
}

static const char *optimizer_name(OptimizerKind kind) {
    switch (kind) {
        case OPT_SGD: return "sgd";
        case OPT_MOMENTUM: return "momentum";
        case OPT_ADAM: return "adam";
    }
    return "unknown";
}

static void parse_args(int argc, char **argv, MnistOptions *opt) {
    char log_path_buf[1024];

    opt->epochs = 5;
    opt->batch = 32;
    opt->channels = 8;
    opt->train_limit = 10000;
    opt->test_limit = 2000;
    opt->opt_kind = OPT_MOMENTUM;
    opt->lr = 0.03f;
    opt->momentum = 0.9f;
    opt->log_path = "out/mnist_training_log.csv";

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else if (sscanf(arg, "--epochs=%d", &opt->epochs) == 1) {
            continue;
        } else if (sscanf(arg, "--batch=%d", &opt->batch) == 1) {
            continue;
        } else if (sscanf(arg, "--channels=%d", &opt->channels) == 1) {
            continue;
        } else if (sscanf(arg, "--train-limit=%d", &opt->train_limit) == 1) {
            continue;
        } else if (sscanf(arg, "--test-limit=%d", &opt->test_limit) == 1) {
            continue;
        } else if (strcmp(arg, "--opt=sgd") == 0) {
            opt->opt_kind = OPT_SGD;
            continue;
        } else if (strcmp(arg, "--opt=momentum") == 0) {
            opt->opt_kind = OPT_MOMENTUM;
            continue;
        } else if (strcmp(arg, "--opt=adam") == 0) {
            opt->opt_kind = OPT_ADAM;
            continue;
        } else if (sscanf(arg, "--lr=%f", &opt->lr) == 1) {
            continue;
        } else if (sscanf(arg, "--momentum=%f", &opt->momentum) == 1) {
            continue;
        } else if (sscanf(arg, "--log=%1023s", log_path_buf) == 1) {
            opt->log_path = argv[i] + 6;
        } else {
            die("unknown option");
        }
    }
}

static Tensor *make_one_hot_batch(const unsigned char *labels, int batch_size) {
    int *tmp = (int *)malloc(sizeof(int) * (size_t)batch_size);
    Tensor *y;

    if (!tmp) {
        die("allocation failed for label buffer");
    }
    for (int i = 0; i < batch_size; i++) {
        tmp[i] = (int)labels[i];
    }
    y = tensor_one_hot(tmp, batch_size, N_CLASSES);
    free(tmp);
    return y;
}

static void mnist_model_init(MnistConvModel *model, int batch, int channels, unsigned int *seed) {
    if (!model) {
        die("mnist_model_init: model is null");
    }

    memset(model, 0, sizeof(*model));
    model->patch_layout = patch_layout_make(MNIST_ROWS, MNIST_COLS, 5, 5);
    model->channels = channels;
    model->batch = batch;

    model->Wc = tensor_create(model->patch_layout.patch_dim, channels, 1);
    model->bc = tensor_create(1, channels, 1);
    model->Wcls = tensor_create(channels, N_CLASSES, 1);
    model->bcls = tensor_create(1, N_CLASSES, 1);
    model->patch_batch = patch_batch_create(model->patch_layout, batch);

    tensor_fill_randn(model->Wc, 0.0f, 0.08f, seed);
    tensor_fill_randn(model->Wcls, 0.0f, 0.08f, seed);
    tensor_fill(model->bc, 0.0f);
    tensor_fill(model->bcls, 0.0f);

    model->params[0] = model->Wc;
    model->params[1] = model->bc;
    model->params[2] = model->Wcls;
    model->params[3] = model->bcls;
    model->velocity[0] = tensor_create(model->patch_layout.patch_dim, channels, 0);
    model->velocity[1] = tensor_create(1, channels, 0);
    model->velocity[2] = tensor_create(channels, N_CLASSES, 0);
    model->velocity[3] = tensor_create(1, N_CLASSES, 0);
    model->adam_m1[0] = tensor_create(model->patch_layout.patch_dim, channels, 0);
    model->adam_m1[1] = tensor_create(1, channels, 0);
    model->adam_m1[2] = tensor_create(channels, N_CLASSES, 0);
    model->adam_m1[3] = tensor_create(1, N_CLASSES, 0);
    model->adam_m2[0] = tensor_create(model->patch_layout.patch_dim, channels, 0);
    model->adam_m2[1] = tensor_create(1, channels, 0);
    model->adam_m2[2] = tensor_create(channels, N_CLASSES, 0);
    model->adam_m2[3] = tensor_create(1, N_CLASSES, 0);
    model->adam_step = 0;
}

static void mnist_model_free(MnistConvModel *model) {
    if (!model) {
        return;
    }
    patch_batch_free(&model->patch_batch);
    tensor_free(model->Wc);
    tensor_free(model->bc);
    tensor_free(model->Wcls);
    tensor_free(model->bcls);
    tensor_free(model->velocity[0]);
    tensor_free(model->velocity[1]);
    tensor_free(model->velocity[2]);
    tensor_free(model->velocity[3]);
    tensor_free(model->adam_m1[0]);
    tensor_free(model->adam_m1[1]);
    tensor_free(model->adam_m1[2]);
    tensor_free(model->adam_m1[3]);
    tensor_free(model->adam_m2[0]);
    tensor_free(model->adam_m2[1]);
    tensor_free(model->adam_m2[2]);
    tensor_free(model->adam_m2[3]);
    memset(model, 0, sizeof(*model));
}

static Tensor *mnist_model_forward(TensorList *temps, MnistConvModel *model, Tensor *xcol) {
    Tensor *conv_lin = tensor_list_add(temps, tensor_matmul(xcol, model->Wc));
    Tensor *conv_bias = tensor_list_add(temps, tensor_add_bias(conv_lin, model->bc));
    Tensor *conv_act = tensor_list_add(temps, tensor_relu(conv_bias));
    Tensor *patch_logits = tensor_list_add(temps, tensor_matmul(conv_act, model->Wcls));
    Tensor *patch_logits_bias = tensor_list_add(temps, tensor_add_bias(patch_logits, model->bcls));
    return patch_mean_pool_rows(temps, patch_logits_bias, model->batch, &model->patch_layout);
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
        TensorList temps = {0};
        Tensor *xcol;
        Tensor *logits;

        patch_extract_batch(&model->patch_layout,
                            ds->images + (size_t)i * MNIST_PIXELS,
                            model->batch,
                            model->patch_batch.buffer);
        xcol = tensor_list_add(&temps, patch_batch_to_tensor(&model->patch_batch));
        logits = mnist_model_forward(&temps, model, xcol);

        for (int b = 0; b < model->batch; b++) {
            int pred = tensor_argmax_row(logits, b);
            if (pred == (int)ds->labels[i + b]) {
                correct++;
            }
        }

        tensor_list_free(&temps);
    }

    if (n_eval == 0) {
        return 0.0f;
    }
    return (float)correct / (float)n_eval;
}

static float mnist_model_train_epoch(MnistConvModel *model,
                                     const MnistSet *train,
                                     const int *indices,
                                     float *batch_images,
                                     unsigned char *batch_labels,
                                     float lr,
                                     OptimizerKind opt_kind,
                                     float momentum) {
    int n_train = (train->n / model->batch) * model->batch;
    float loss_sum = 0.0f;
    int loss_count = 0;

    for (int i = 0; i < n_train; i += model->batch) {
        TensorList temps = {0};
        Tensor *xcol;
        Tensor *y;
        Tensor *logits;
        Tensor *probs;
        Tensor *loss;

        mnist_gather_batch(train, indices, i, model->batch, batch_images, batch_labels);
        patch_extract_batch(&model->patch_layout, batch_images, model->batch, model->patch_batch.buffer);
        xcol = tensor_list_add(&temps, patch_batch_to_tensor(&model->patch_batch));
        y = tensor_list_add(&temps, make_one_hot_batch(batch_labels, model->batch));

        logits = mnist_model_forward(&temps, model, xcol);
        probs = tensor_list_add(&temps, tensor_softmax(logits));
        loss = tensor_list_add(&temps, tensor_cross_entropy(probs, y));

        tensor_backward(loss);
        if (opt_kind == OPT_ADAM) {
            TensorAdamOptions adam = {0};
            adam.lr = lr;
            adam.beta1 = 0.9f;
            adam.beta2 = 0.999f;
            adam.eps = 1e-8f;
            adam.timestep = ++model->adam_step;
            tensor_adam_step(model->params, model->adam_m1, model->adam_m2, 4, &adam);
        } else if (opt_kind == OPT_MOMENTUM) {
            tensor_sgd_momentum_step(model->params, model->velocity, 4, lr, momentum);
        } else {
            tensor_sgd_step(model->params, 4, lr);
        }

        loss_sum += loss->data[0];
        loss_count++;

        tensor_list_free(&temps);
    }

    return loss_count > 0 ? loss_sum / (float)loss_count : 0.0f;
}

int main(int argc, char **argv) {
    MnistOptions opt;
    unsigned int seed = 1337U;
    const char *train_images = "data/mnist/train-images-idx3-ubyte";
    const char *train_labels = "data/mnist/train-labels-idx1-ubyte";
    const char *test_images = "data/mnist/t10k-images-idx3-ubyte";
    const char *test_labels = "data/mnist/t10k-labels-idx1-ubyte";
    MnistSet train;
    MnistSet test;
    MnistConvModel model;
    int *train_indices;
    float *batch_images;
    unsigned char *batch_labels;
    FILE *log_file;

    parse_args(argc, argv, &opt);

    if (opt.epochs <= 0 || opt.batch <= 0 || opt.channels <= 0 || opt.momentum < 0.0f || opt.momentum >= 1.0f) {
        die("invalid epochs/batch/channels/momentum");
    }

    train = mnist_load(train_images, train_labels, opt.train_limit);
    test = mnist_load(test_images, test_labels, opt.test_limit);

    if (train.n < opt.batch || test.n < opt.batch) {
        die("dataset too small for chosen batch size");
    }

    mnist_model_init(&model, opt.batch, opt.channels, &seed);
    train_indices = (int *)malloc(sizeof(int) * (size_t)train.n);
    batch_images = (float *)malloc(sizeof(float) * (size_t)opt.batch * MNIST_PIXELS);
    batch_labels = (unsigned char *)malloc((size_t)opt.batch);
    if (!train_indices || !batch_images || !batch_labels) {
        die("allocation failed for training buffers");
    }
    for (int i = 0; i < train.n; i++) {
        train_indices[i] = i;
    }

    log_file = fopen(opt.log_path, "w");
    if (!log_file) {
        die("failed to open log file");
    }
    print_architecture_summary(log_file,
                               "# ",
                               model.patch_layout.kernel_h,
                               model.patch_layout.kernel_w,
                               model.channels,
                               model.patch_layout.patches_per_image);
    fprintf(log_file, "# train=%d test=%d batch=%d epochs=%d opt=%s lr=%.4f momentum=%.3f channels=%d\n",
            train.n, test.n, opt.batch, opt.epochs, optimizer_name(opt.opt_kind), opt.lr, opt.momentum, opt.channels);
    fprintf(log_file, "epoch,train_loss,train_acc,train_error,test_acc,test_error\n");

    printf("MNIST conv-matmul demo\n");
    printf("train=%d test=%d batch=%d epochs=%d opt=%s lr=%.4f momentum=%.3f channels=%d\n",
           train.n,
           test.n,
           opt.batch,
           opt.epochs,
           optimizer_name(opt.opt_kind),
           opt.lr,
           opt.momentum,
           opt.channels);
    print_architecture_summary(
        stdout, NULL, model.patch_layout.kernel_h, model.patch_layout.kernel_w, model.channels, model.patch_layout.patches_per_image);
    printf("logging metrics to %s\n", opt.log_path);

    {
        double start_time = now_seconds();

    for (int epoch = 1; epoch <= opt.epochs; epoch++) {
        {
            mnist_shuffle_indices(train_indices, train.n, &seed);
            float avg_loss = mnist_model_train_epoch(
                &model, &train, train_indices, batch_images, batch_labels, opt.lr, opt.opt_kind, opt.momentum);
            float train_acc = evaluate_accuracy(&train, &model);
            float test_acc = evaluate_accuracy(&test, &model);
            float train_error = 1.0f - train_acc;
            float test_error = 1.0f - test_acc;
            double elapsed = now_seconds() - start_time;

            printf("epoch %d/%d loss %.4f train_acc %.3f test_acc %.3f elapsed %.2fs\n",
                   epoch, opt.epochs, avg_loss, train_acc, test_acc, elapsed);
            fprintf(log_file, "%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    epoch, avg_loss, train_acc, train_error, test_acc, test_error);
            fflush(log_file);
        }
    }
    }

    fclose(log_file);
    free(train_indices);
    free(batch_images);
    free(batch_labels);
    mnist_model_free(&model);
    mnist_free(&train);
    mnist_free(&test);

    return 0;
}
