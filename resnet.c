#include "mnist.h"
#include "patch.h"
#include "tensor.h"

/*
 * resnet.c
 *
 * Minimal MNIST residual patch-network demo using nanotensor.
 * - Extracts 5x5 image patches with the shared patch helpers
 * - Projects each patch into a learned feature space with a small stem
 * - Refines patch features with two residual MLP blocks using layernorm
 * - Mean-pools patch features per image and trains a 10-way classifier
 *
 * Usage:
 *   ./resnet_demo [--epochs=N] [--batch=N] [--dim=N] [--hidden=N]
 *                       [--opt=sgd|momentum|adam] [--lr=FLOAT]
 *                       [--momentum=FLOAT] [--train-limit=N]
 *                       [--test-limit=N] [--log=PATH] [--snapshot=PATH]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#define N_CLASSES 10
#define N_BLOCKS 2
#define PARAM_COUNT 16

typedef enum {
    OPT_SGD = 0,
    OPT_MOMENTUM,
    OPT_ADAM
} OptimizerKind;

typedef struct {
    Tensor *ln_gamma;
    Tensor *ln_beta;
    Tensor *W1;
    Tensor *b1;
    Tensor *W2;
    Tensor *b2;
} ResidualBlock;

typedef struct {
    PatchLayout patch_layout;
    PatchBatch patch_batch;
    int batch;
    int dim;
    int hidden;
    Tensor *W_in;
    Tensor *b_in;
    ResidualBlock blocks[N_BLOCKS];
    Tensor *W_out;
    Tensor *b_out;
    TensorList params;
    Tensor *velocity[PARAM_COUNT];
    Tensor *adam_m1[PARAM_COUNT];
    Tensor *adam_m2[PARAM_COUNT];
    int adam_step;
} MnistResNetModel;

typedef struct {
    int epochs;
    int batch;
    int dim;
    int hidden;
    int train_limit;
    int test_limit;
    OptimizerKind opt_kind;
    float lr;
    float momentum;
    const char *log_path;
    const char *snapshot_path;
} ResNetOptions;

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

static const char *optimizer_name(OptimizerKind kind) {
    switch (kind) {
        case OPT_SGD: return "sgd";
        case OPT_MOMENTUM: return "momentum";
        case OPT_ADAM: return "adam";
    }
    return "unknown";
}

static void print_usage(const char *prog) {
    printf("usage: %s [options]\n", prog);
    printf("  --epochs=N\n");
    printf("  --batch=N\n");
    printf("  --dim=N\n");
    printf("  --hidden=N\n");
    printf("  --train-limit=N\n");
    printf("  --test-limit=N\n");
    printf("  --opt=sgd|momentum|adam\n");
    printf("  --lr=FLOAT\n");
    printf("  --momentum=FLOAT\n");
    printf("  --log=PATH\n");
    printf("  --snapshot=PATH\n");
}

static void parse_args(int argc, char **argv, ResNetOptions *opt) {
    char log_path_buf[1024];

    opt->epochs = 5;
    opt->batch = 32;
    opt->dim = 32;
    opt->hidden = 64;
    opt->train_limit = 2000;
    opt->test_limit = 1000;
    opt->opt_kind = OPT_ADAM;
    opt->lr = 0.001f;
    opt->momentum = 0.9f;
    opt->log_path = "out/resnet_training_log.csv";
    opt->snapshot_path = NULL;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else if (sscanf(arg, "--epochs=%d", &opt->epochs) == 1) {
            continue;
        } else if (sscanf(arg, "--batch=%d", &opt->batch) == 1) {
            continue;
        } else if (sscanf(arg, "--dim=%d", &opt->dim) == 1) {
            continue;
        } else if (sscanf(arg, "--hidden=%d", &opt->hidden) == 1) {
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
        } else if (strncmp(arg, "--snapshot=", 11) == 0) {
            opt->snapshot_path = argv[i] + 11;
        } else {
            die("unknown option");
        }
    }
}

static int file_exists(const char *path) {
    return path && access(path, F_OK) == 0;
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

static Tensor *resnet_add_param(MnistResNetModel *model, Tensor *param) {
    return tensor_list_add(&model->params, param);
}

static void residual_block_init(MnistResNetModel *model, ResidualBlock *block, int dim, int hidden, unsigned int *seed) {
    memset(block, 0, sizeof(*block));
    block->ln_gamma = resnet_add_param(model, tensor_create(1, dim, 1));
    block->ln_beta = resnet_add_param(model, tensor_create(1, dim, 1));
    block->W1 = resnet_add_param(model, tensor_create(dim, hidden, 1));
    block->b1 = resnet_add_param(model, tensor_create(1, hidden, 1));
    block->W2 = resnet_add_param(model, tensor_create(hidden, dim, 1));
    block->b2 = resnet_add_param(model, tensor_create(1, dim, 1));

    tensor_fill(block->ln_gamma, 1.0f);
    tensor_fill(block->ln_beta, 0.0f);
    tensor_fill_randn(block->W1, 0.0f, 0.05f, seed);
    tensor_fill_randn(block->W2, 0.0f, 0.05f, seed);
    tensor_fill(block->b1, 0.0f);
    tensor_fill(block->b2, 0.0f);
}

static Tensor *residual_block_forward(TensorList *temps, ResidualBlock *block, Tensor *x) {
    Tensor *norm = tensor_list_add(temps, tensor_layernorm(x, block->ln_gamma, block->ln_beta, 1e-5f));
    Tensor *ff1_lin = tensor_list_add(temps, tensor_matmul(norm, block->W1));
    Tensor *ff1_bias = tensor_list_add(temps, tensor_add_bias(ff1_lin, block->b1));
    Tensor *ff1_act = tensor_list_add(temps, tensor_relu(ff1_bias));
    Tensor *ff2_lin = tensor_list_add(temps, tensor_matmul(ff1_act, block->W2));
    Tensor *ff2_bias = tensor_list_add(temps, tensor_add_bias(ff2_lin, block->b2));
    return tensor_list_add(temps, tensor_add(x, ff2_bias));
}

static void resnet_init(MnistResNetModel *model, const ResNetOptions *opt, unsigned int *seed) {
    memset(model, 0, sizeof(*model));
    tensor_list_init(&model->params);
    model->patch_layout = patch_layout_make(MNIST_ROWS, MNIST_COLS, 5, 5);
    model->patch_batch = patch_batch_create(model->patch_layout, opt->batch);
    model->batch = opt->batch;
    model->dim = opt->dim;
    model->hidden = opt->hidden;

    model->W_in = resnet_add_param(model, tensor_create(model->patch_layout.patch_dim, opt->dim, 1));
    model->b_in = resnet_add_param(model, tensor_create(1, opt->dim, 1));
    tensor_fill_randn(model->W_in, 0.0f, 0.08f, seed);
    tensor_fill(model->b_in, 0.0f);

    for (int i = 0; i < N_BLOCKS; i++) {
        residual_block_init(model, &model->blocks[i], opt->dim, opt->hidden, seed);
    }

    model->W_out = resnet_add_param(model, tensor_create(opt->dim, N_CLASSES, 1));
    model->b_out = resnet_add_param(model, tensor_create(1, N_CLASSES, 1));
    tensor_fill_randn(model->W_out, 0.0f, 0.08f, seed);
    tensor_fill(model->b_out, 0.0f);

    if (model->params.len != PARAM_COUNT) {
        die("resnet_init: unexpected param count");
    }

    for (int i = 0; i < PARAM_COUNT; i++) {
        Tensor *param = model->params.items[i];
        model->velocity[i] = tensor_create(param->rows, param->cols, 0);
        model->adam_m1[i] = tensor_create(param->rows, param->cols, 0);
        model->adam_m2[i] = tensor_create(param->rows, param->cols, 0);
    }
}

static int resnet_save(const MnistResNetModel *model, const char *path) {
    return tensor_list_save(&model->params, path);
}

static int resnet_load(MnistResNetModel *model, const char *path) {
    return tensor_list_load(&model->params, path);
}

static void resnet_free(MnistResNetModel *model) {
    if (!model) {
        return;
    }

    patch_batch_free(&model->patch_batch);
    tensor_list_free(&model->params);
    for (int i = 0; i < PARAM_COUNT; i++) {
        tensor_free(model->velocity[i]);
        tensor_free(model->adam_m1[i]);
        tensor_free(model->adam_m2[i]);
    }
    memset(model, 0, sizeof(*model));
}

static Tensor *resnet_forward(TensorList *temps, MnistResNetModel *model, Tensor *xcol) {
    Tensor *x = tensor_list_add(temps, tensor_matmul(xcol, model->W_in));
    Tensor *x_bias = tensor_list_add(temps, tensor_add_bias(x, model->b_in));
    Tensor *x_act = tensor_list_add(temps, tensor_relu(x_bias));
    Tensor *features = x_act;

    for (int i = 0; i < N_BLOCKS; i++) {
        features = residual_block_forward(temps, &model->blocks[i], features);
    }

    {
        Tensor *pooled = patch_mean_pool_rows(temps, features, model->batch, &model->patch_layout);
        Tensor *head_lin = tensor_list_add(temps, tensor_matmul(pooled, model->W_out));
        return tensor_list_add(temps, tensor_add_bias(head_lin, model->b_out));
    }
}

static void print_architecture_summary(FILE *out, const char *prefix, const MnistResNetModel *model) {
    const char *p = prefix ? prefix : "";

    fprintf(out, "%sarch: input=28x28x1 im2col=%dx%d\n",
            p,
            model->patch_layout.kernel_h,
            model->patch_layout.kernel_w);
    fprintf(out, "%sarch: patch_dim=%d patches=%d\n",
            p,
            model->patch_layout.patch_dim,
            model->patch_layout.patches_per_image);
    fprintf(out, "%sarch: stem=matmul(%d->%d)+bias+relu\n",
            p,
            model->patch_layout.patch_dim,
            model->dim);
    fprintf(out, "%sarch: residual_blocks=%d block_width=%d ff_hidden=%d\n", p, N_BLOCKS, model->dim, model->hidden);
    fprintf(out, "%sarch: block=x+linear(relu(linear(layernorm(x))))\n", p);
    fprintf(out, "%sarch: pool=mean_over_patches head=matmul(%d->10)+bias loss=softmax_ce\n", p, model->dim);
}

static float evaluate_accuracy(const MnistSet *ds, MnistResNetModel *model) {
    int n_eval = (ds->n / model->batch) * model->batch;
    int correct = 0;

    for (int i = 0; i < n_eval; i += model->batch) {
        TensorList temps;
        Tensor *xcol;
        Tensor *logits;

        tensor_list_init(&temps);

        patch_extract_batch(&model->patch_layout,
                            ds->images + (size_t)i * MNIST_PIXELS,
                            model->batch,
                            model->patch_batch.buffer);
        xcol = tensor_list_add(&temps, patch_batch_to_tensor(&model->patch_batch));
        logits = resnet_forward(&temps, model, xcol);

        for (int b = 0; b < model->batch; b++) {
            if (tensor_argmax_row(logits, b) == (int)ds->labels[i + b]) {
                correct++;
            }
        }

        tensor_list_free(&temps);
    }

    return n_eval > 0 ? (float)correct / (float)n_eval : 0.0f;
}

static float train_epoch(MnistResNetModel *model,
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
        TensorList temps;
        Tensor *xcol;
        Tensor *y;
        Tensor *logits;
        Tensor *probs;
        Tensor *loss;

        tensor_list_init(&temps);

        mnist_gather_batch(train, indices, i, model->batch, batch_images, batch_labels);
        patch_extract_batch(&model->patch_layout, batch_images, model->batch, model->patch_batch.buffer);
        xcol = tensor_list_add(&temps, patch_batch_to_tensor(&model->patch_batch));
        y = tensor_list_add(&temps, make_one_hot_batch(batch_labels, model->batch));

        logits = resnet_forward(&temps, model, xcol);
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
            tensor_adam_step(model->params.items, model->adam_m1, model->adam_m2, (size_t)model->params.len, &adam);
        } else if (opt_kind == OPT_MOMENTUM) {
            tensor_sgd_momentum_step(model->params.items, model->velocity, (size_t)model->params.len, lr, momentum);
        } else {
            tensor_sgd_step(model->params.items, (size_t)model->params.len, lr);
        }

        loss_sum += loss->data[0];
        loss_count++;

        tensor_list_free(&temps);
    }

    return loss_count > 0 ? loss_sum / (float)loss_count : 0.0f;
}

int main(int argc, char **argv) {
    ResNetOptions opt;
    const char *train_images = "data/mnist/train-images-idx3-ubyte";
    const char *train_labels = "data/mnist/train-labels-idx1-ubyte";
    const char *test_images = "data/mnist/t10k-images-idx3-ubyte";
    const char *test_labels = "data/mnist/t10k-labels-idx1-ubyte";
    unsigned int seed = 1337U;
    MnistSet train;
    MnistSet test;
    MnistResNetModel model;
    int *train_indices;
    float *batch_images;
    unsigned char *batch_labels;
    FILE *log_file;

    parse_args(argc, argv, &opt);
    if (opt.epochs <= 0 || opt.batch <= 0 || opt.dim <= 0 || opt.hidden <= 0 ||
        opt.momentum < 0.0f || opt.momentum >= 1.0f) {
        die("invalid epochs/batch/dim/hidden/momentum");
    }

    train = mnist_load(train_images, train_labels, opt.train_limit);
    test = mnist_load(test_images, test_labels, opt.test_limit);
    if (train.n < opt.batch || test.n < opt.batch) {
        die("dataset too small for chosen batch size");
    }

    resnet_init(&model, &opt, &seed);
    if (file_exists(opt.snapshot_path) && resnet_load(&model, opt.snapshot_path) != 0) {
        die("failed to load snapshot");
    }
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
    print_architecture_summary(log_file, "# ", &model);
    fprintf(log_file,
            "# train=%d test=%d batch=%d epochs=%d opt=%s lr=%.4f momentum=%.3f dim=%d hidden=%d\n",
            train.n,
            test.n,
            opt.batch,
            opt.epochs,
            optimizer_name(opt.opt_kind),
            opt.lr,
            opt.momentum,
            opt.dim,
            opt.hidden);
    fprintf(log_file, "epoch,train_loss,train_acc,train_error,test_acc,test_error\n");

    printf("MNIST residual patch demo\n");
    printf("train=%d test=%d batch=%d epochs=%d opt=%s lr=%.4f momentum=%.3f dim=%d hidden=%d\n",
           train.n,
           test.n,
           opt.batch,
           opt.epochs,
           optimizer_name(opt.opt_kind),
           opt.lr,
           opt.momentum,
           opt.dim,
           opt.hidden);
    print_architecture_summary(stdout, NULL, &model);
    printf("logging metrics to %s\n", opt.log_path);
    if (file_exists(opt.snapshot_path)) {
        printf("loaded snapshot from %s\n", opt.snapshot_path);
    }

    {
        double start_time = now_seconds();

        for (int epoch = 1; epoch <= opt.epochs; epoch++) {
            float avg_loss;
            float train_acc;
            float test_acc;
            float train_error;
            float test_error;
            double elapsed;

            mnist_shuffle_indices(train_indices, train.n, &seed);
            avg_loss = train_epoch(
                &model, &train, train_indices, batch_images, batch_labels, opt.lr, opt.opt_kind, opt.momentum);
            train_acc = evaluate_accuracy(&train, &model);
            test_acc = evaluate_accuracy(&test, &model);
            train_error = 1.0f - train_acc;
            test_error = 1.0f - test_acc;
            elapsed = now_seconds() - start_time;

            printf("epoch %d/%d loss %.4f train_acc %.3f test_acc %.3f elapsed %.2fs\n",
                   epoch, opt.epochs, avg_loss, train_acc, test_acc, elapsed);
            fprintf(log_file, "%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    epoch, avg_loss, train_acc, train_error, test_acc, test_error);
            fflush(log_file);
        }
    }

    if (opt.snapshot_path && resnet_save(&model, opt.snapshot_path) != 0) {
        die("failed to save snapshot");
    }
    if (opt.snapshot_path) {
        printf("saved snapshot to %s\n", opt.snapshot_path);
    }

    fclose(log_file);
    free(train_indices);
    free(batch_images);
    free(batch_labels);
    resnet_free(&model);
    mnist_free(&train);
    mnist_free(&test);
    return 0;
}
