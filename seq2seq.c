#include "tensor.h"

/*
 * seq2seq.c
 *
 * Minimal seq2seq demo scaffold using nanotensor.
 * - Uses a synthetic digit-sequence reversal task
 * - Plans for a tanh RNN encoder and tanh RNN decoder
 * - Keeps the first version intentionally small and readable
 *
 * Usage:
 *   ./seq2seq [--steps=N] [--batch=N] [--embed=N]
 *             [--hidden=N] [--lr=FLOAT] [--min-len=N]
 *             [--max-len=N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIGIT_TOKENS 10
#define BOS_TOKEN 10
#define EOS_TOKEN 11
#define VOCAB_SIZE 12

typedef struct {
    int steps;
    int batch;
    int embed;
    int hidden;
    int min_len;
    int max_len;
    float lr;
} Seq2SeqOptions;

/* Planned architecture:
 * input tokens [seq]
 * -> embeddings [seq, embed]
 * -> encoder tanh RNN hidden states [seq, hidden]
 * -> final encoder hidden initializes decoder state
 * -> decoder tanh RNN unrolled with teacher forcing
 * -> vocab logits per output position [seq + 1, vocab]
 */
typedef struct {
    int embed;
    int hidden;
    Tensor *E;
    Tensor *W_enc_x;
    Tensor *W_enc_h;
    Tensor *b_enc;
    Tensor *W_dec_x;
    Tensor *W_dec_h;
    Tensor *b_dec;
    Tensor *W_out;
    Tensor *b_out;
    Tensor *params[9];
    Tensor *velocity[9];
} Seq2SeqModel;

static void seq2seq_print_usage(const char *prog);
static void seq2seq_parse_args(int argc, char **argv, Seq2SeqOptions *opt);
static void seq2seq_model_init(Seq2SeqModel *model, const Seq2SeqOptions *opt, unsigned int *seed);
static void seq2seq_model_free(Seq2SeqModel *model);
static void seq2seq_print_architecture(const Seq2SeqModel *model);

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static void seq2seq_print_usage(const char *prog) {
    printf("usage: %s [options]\n", prog);
    printf("  --steps=N\n");
    printf("  --batch=N\n");
    printf("  --embed=N\n");
    printf("  --hidden=N\n");
    printf("  --lr=FLOAT\n");
    printf("  --min-len=N\n");
    printf("  --max-len=N\n");
}

static void seq2seq_parse_args(int argc, char **argv, Seq2SeqOptions *opt) {
    opt->steps = 2000;
    opt->batch = 32;
    opt->embed = 16;
    opt->hidden = 32;
    opt->min_len = 3;
    opt->max_len = 8;
    opt->lr = 0.03f;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            seq2seq_print_usage(argv[0]);
            exit(0);
        } else if (sscanf(arg, "--steps=%d", &opt->steps) == 1) {
            continue;
        } else if (sscanf(arg, "--batch=%d", &opt->batch) == 1) {
            continue;
        } else if (sscanf(arg, "--embed=%d", &opt->embed) == 1) {
            continue;
        } else if (sscanf(arg, "--hidden=%d", &opt->hidden) == 1) {
            continue;
        } else if (sscanf(arg, "--lr=%f", &opt->lr) == 1) {
            continue;
        } else if (sscanf(arg, "--min-len=%d", &opt->min_len) == 1) {
            continue;
        } else if (sscanf(arg, "--max-len=%d", &opt->max_len) == 1) {
            continue;
        } else {
            die("unknown option");
        }
    }
}

static void seq2seq_model_init(Seq2SeqModel *model, const Seq2SeqOptions *opt, unsigned int *seed) {
    memset(model, 0, sizeof(*model));
    model->embed = opt->embed;
    model->hidden = opt->hidden;

    model->E = tensor_create(VOCAB_SIZE, opt->embed, 1);
    model->W_enc_x = tensor_create(opt->embed, opt->hidden, 1);
    model->W_enc_h = tensor_create(opt->hidden, opt->hidden, 1);
    model->b_enc = tensor_create(1, opt->hidden, 1);
    model->W_dec_x = tensor_create(opt->embed, opt->hidden, 1);
    model->W_dec_h = tensor_create(opt->hidden, opt->hidden, 1);
    model->b_dec = tensor_create(1, opt->hidden, 1);
    model->W_out = tensor_create(opt->hidden, VOCAB_SIZE, 1);
    model->b_out = tensor_create(1, VOCAB_SIZE, 1);

    if (!model->E || !model->W_enc_x || !model->W_enc_h || !model->b_enc || !model->W_dec_x ||
        !model->W_dec_h || !model->b_dec || !model->W_out || !model->b_out) {
        die("seq2seq parameter allocation failed");
    }

    tensor_fill_randn(model->E, 0.0f, 0.05f, seed);
    tensor_fill_randn(model->W_enc_x, 0.0f, 0.05f, seed);
    tensor_fill_randn(model->W_enc_h, 0.0f, 0.05f, seed);
    tensor_fill_randn(model->W_dec_x, 0.0f, 0.05f, seed);
    tensor_fill_randn(model->W_dec_h, 0.0f, 0.05f, seed);
    tensor_fill_randn(model->W_out, 0.0f, 0.05f, seed);
    tensor_fill(model->b_enc, 0.0f);
    tensor_fill(model->b_dec, 0.0f);
    tensor_fill(model->b_out, 0.0f);

    model->params[0] = model->E;
    model->params[1] = model->W_enc_x;
    model->params[2] = model->W_enc_h;
    model->params[3] = model->b_enc;
    model->params[4] = model->W_dec_x;
    model->params[5] = model->W_dec_h;
    model->params[6] = model->b_dec;
    model->params[7] = model->W_out;
    model->params[8] = model->b_out;

    for (int i = 0; i < 9; i++) {
        model->velocity[i] = tensor_create(model->params[i]->rows, model->params[i]->cols, 0);
        if (!model->velocity[i]) {
            die("seq2seq velocity allocation failed");
        }
    }
}

static void seq2seq_model_free(Seq2SeqModel *model) {
    if (!model) {
        return;
    }
    for (int i = 0; i < 9; i++) {
        tensor_free(model->params[i]);
        tensor_free(model->velocity[i]);
    }
    memset(model, 0, sizeof(*model));
}

static void seq2seq_print_architecture(const Seq2SeqModel *model) {
    printf("arch: vocab=%d embed=%d hidden=%d\n", VOCAB_SIZE, model->embed, model->hidden);
    printf("arch: task=reverse digit sequences with BOS/EOS\n");
    printf("arch: encoder=tanh_rnn decoder=tanh_rnn\n");
    printf("arch: bridge=final_encoder_hidden -> decoder_h0\n");
    printf("arch: output=decoder_hidden -> vocab logits\n");
}

int main(int argc, char **argv) {
    Seq2SeqOptions opt;
    Seq2SeqModel model;
    unsigned int seed = 1337u;

    seq2seq_parse_args(argc, argv, &opt);
    if (opt.steps <= 0) opt.steps = 2000;
    if (opt.batch <= 0) opt.batch = 32;
    if (opt.embed <= 0) opt.embed = 16;
    if (opt.hidden <= 0) opt.hidden = 32;
    if (opt.min_len <= 0) opt.min_len = 3;
    if (opt.max_len < opt.min_len) opt.max_len = opt.min_len;
    if (opt.lr <= 0.0f) opt.lr = 0.03f;

    seq2seq_model_init(&model, &opt, &seed);

    printf("seq2seq scaffold\n");
    seq2seq_print_architecture(&model);
    printf("opt: steps=%d batch=%d min_len=%d max_len=%d lr=%.4f\n",
           opt.steps,
           opt.batch,
           opt.min_len,
           opt.max_len,
           opt.lr);
    printf("status: model scaffold is ready; training loop comes next\n");

    seq2seq_model_free(&model);
    return 0;
}
