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

/* Architecture:
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

typedef struct {
    Tensor **items;
    int len;
    int cap;
} TensorTemps;

static void seq2seq_print_usage(const char *prog);
static void seq2seq_parse_args(int argc, char **argv, Seq2SeqOptions *opt);
static void seq2seq_model_init(Seq2SeqModel *model, const Seq2SeqOptions *opt, unsigned int *seed);
static void seq2seq_model_free(Seq2SeqModel *model);
static void seq2seq_print_architecture(const Seq2SeqModel *model);
static void seq2seq_sample_batch(int *enc_tokens,
                                 int *dec_in_tokens,
                                 int *dec_tgt_tokens,
                                 int batch,
                                 int seq_len,
                                 unsigned int *seed);
static float seq2seq_train_step(Seq2SeqModel *model,
                                const Seq2SeqOptions *opt,
                                int seq_len,
                                int *enc_tokens,
                                int *dec_in_tokens,
                                int *dec_tgt_tokens,
                                unsigned int *seed,
                                float *out_token_acc);
static void seq2seq_decode_example(Seq2SeqModel *model, int seq_len, unsigned int *seed);

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static float rand_uniform(unsigned int *seed) {
    *seed = (1103515245U * (*seed) + 12345U);
    return (float)((*seed) & 0x7fffffffU) / 2147483648.0f;
}

static int rand_int(unsigned int *seed, int lo, int hi_inclusive) {
    return lo + (int)(rand_uniform(seed) * (float)(hi_inclusive - lo + 1));
}

static void temps_push(TensorTemps *temps, Tensor *t) {
    if (temps->len == temps->cap) {
        int next_cap = temps->cap == 0 ? 64 : temps->cap * 2;
        Tensor **next = (Tensor **)realloc(temps->items, sizeof(Tensor *) * (size_t)next_cap);
        if (!next) {
            die("seq2seq temp allocation failed");
        }
        temps->items = next;
        temps->cap = next_cap;
    }
    temps->items[temps->len++] = t;
}

static void temps_free_all(TensorTemps *temps) {
    for (int i = 0; i < temps->len; i++) {
        tensor_free(temps->items[i]);
    }
    free(temps->items);
    temps->items = NULL;
    temps->len = 0;
    temps->cap = 0;
}

static char token_to_char(int token) {
    if (token >= 0 && token < DIGIT_TOKENS) {
        return (char)('0' + token);
    }
    if (token == BOS_TOKEN) {
        return '^';
    }
    if (token == EOS_TOKEN) {
        return '$';
    }
    return '?';
}

static void tokens_to_string(const int *tokens, int n, char *out, size_t out_size) {
    int k = 0;
    if (out_size == 0) {
        return;
    }
    for (int i = 0; i < n && k + 1 < (int)out_size; i++) {
        if (tokens[i] == EOS_TOKEN) {
            break;
        }
        if (tokens[i] == BOS_TOKEN) {
            continue;
        }
        out[k++] = token_to_char(tokens[i]);
    }
    out[k] = '\0';
}

static Tensor *seq2seq_rnn_step(TensorTemps *temps,
                                Seq2SeqModel *model,
                                const int *token_ids,
                                int batch,
                                Tensor *prev_h,
                                Tensor *W_x,
                                Tensor *W_h,
                                Tensor *b) {
    Tensor *x_onehot = tensor_one_hot(token_ids, batch, VOCAB_SIZE);
    Tensor *embed = tensor_matmul(x_onehot, model->E);
    Tensor *x_proj = tensor_matmul(embed, W_x);
    Tensor *h_proj = tensor_matmul(prev_h, W_h);
    Tensor *sum = tensor_add(x_proj, h_proj);
    Tensor *bias = tensor_add_bias(sum, b);
    Tensor *next_h = tensor_tanh(bias);

    temps_push(temps, x_onehot);
    temps_push(temps, embed);
    temps_push(temps, x_proj);
    temps_push(temps, h_proj);
    temps_push(temps, sum);
    temps_push(temps, bias);
    temps_push(temps, next_h);
    return next_h;
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

static void seq2seq_sample_batch(int *enc_tokens,
                                 int *dec_in_tokens,
                                 int *dec_tgt_tokens,
                                 int batch,
                                 int seq_len,
                                 unsigned int *seed) {
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < seq_len; t++) {
            int digit = rand_int(seed, 0, 9);
            enc_tokens[t * batch + b] = digit;
            dec_tgt_tokens[t * batch + b] = enc_tokens[(seq_len - 1 - t) * batch + b];
        }
        dec_in_tokens[b] = BOS_TOKEN;
        for (int t = 1; t < seq_len + 1; t++) {
            dec_in_tokens[t * batch + b] = dec_tgt_tokens[(t - 1) * batch + b];
        }
        dec_tgt_tokens[seq_len * batch + b] = EOS_TOKEN;
    }
}

static float seq2seq_train_step(Seq2SeqModel *model,
                                const Seq2SeqOptions *opt,
                                int seq_len,
                                int *enc_tokens,
                                int *dec_in_tokens,
                                int *dec_tgt_tokens,
                                unsigned int *seed,
                                float *out_token_acc) {
    TensorTemps temps = {0};
    Tensor *h = tensor_create(opt->batch, model->hidden, 0);
    Tensor *total_loss = NULL;
    int total = 0;
    int correct = 0;

    (void)seed;
    tensor_fill(h, 0.0f);
    temps_push(&temps, h);

    for (int t = 0; t < seq_len; t++) {
        h = seq2seq_rnn_step(&temps,
                             model,
                             enc_tokens + t * opt->batch,
                             opt->batch,
                             h,
                             model->W_enc_x,
                             model->W_enc_h,
                             model->b_enc);
    }

    for (int t = 0; t < seq_len + 1; t++) {
        Tensor *dec_h = seq2seq_rnn_step(&temps,
                                         model,
                                         dec_in_tokens + t * opt->batch,
                                         opt->batch,
                                         h,
                                         model->W_dec_x,
                                         model->W_dec_h,
                                         model->b_dec);
        Tensor *out_lin = tensor_matmul(dec_h, model->W_out);
        Tensor *logits = tensor_add_bias(out_lin, model->b_out);
        Tensor *probs = tensor_softmax(logits);
        Tensor *target = tensor_one_hot(dec_tgt_tokens + t * opt->batch, opt->batch, VOCAB_SIZE);
        Tensor *loss_t = tensor_cross_entropy(probs, target);

        temps_push(&temps, out_lin);
        temps_push(&temps, logits);
        temps_push(&temps, probs);
        temps_push(&temps, target);
        temps_push(&temps, loss_t);

        if (!total_loss) {
            total_loss = loss_t;
        } else {
            total_loss = tensor_add(total_loss, loss_t);
            temps_push(&temps, total_loss);
        }
        h = dec_h;

        for (int b = 0; b < opt->batch; b++) {
            int pred = tensor_argmax_row(logits, b);
            if (pred == dec_tgt_tokens[t * opt->batch + b]) {
                correct++;
            }
            total++;
        }
    }

    if (seq_len + 1 > 1) {
        total_loss = tensor_scalar_mul(total_loss, 1.0f / (float)(seq_len + 1));
        temps_push(&temps, total_loss);
    }

    tensor_backward(total_loss);
    tensor_sgd_momentum_step(model->params, model->velocity, 9, opt->lr, 0.9f);

    {
        float loss_value = total_loss->data[0];
        float token_acc = total > 0 ? (float)correct / (float)total : 0.0f;
        if (out_token_acc) {
            *out_token_acc = token_acc;
        }
        temps_free_all(&temps);
        return loss_value;
    }
}

static void seq2seq_decode_example(Seq2SeqModel *model, int seq_len, unsigned int *seed) {
    TensorTemps temps = {0};
    Tensor *h = tensor_create(1, model->hidden, 0);
    int *enc_tokens = (int *)malloc(sizeof(int) * (size_t)seq_len);
    int *pred_tokens = (int *)malloc(sizeof(int) * (size_t)(seq_len + 1));
    int *tgt_tokens = (int *)malloc(sizeof(int) * (size_t)(seq_len + 1));
    int bos = BOS_TOKEN;
    char src[64];
    char tgt[64];
    char pred[64];

    if (!enc_tokens || !pred_tokens || !tgt_tokens) {
        die("seq2seq decode allocation failed");
    }

    tensor_fill(h, 0.0f);
    temps_push(&temps, h);
    for (int t = 0; t < seq_len; t++) {
        enc_tokens[t] = rand_int(seed, 0, 9);
    }
    for (int t = 0; t < seq_len; t++) {
        h = seq2seq_rnn_step(&temps, model, &enc_tokens[t], 1, h, model->W_enc_x, model->W_enc_h, model->b_enc);
    }

    {
        int prev = bos;
        for (int t = 0; t < seq_len + 1; t++) {
            Tensor *dec_h = seq2seq_rnn_step(&temps, model, &prev, 1, h, model->W_dec_x, model->W_dec_h, model->b_dec);
            Tensor *out_lin = tensor_matmul(dec_h, model->W_out);
            Tensor *logits = tensor_add_bias(out_lin, model->b_out);

            temps_push(&temps, out_lin);
            temps_push(&temps, logits);
            pred_tokens[t] = tensor_argmax_row(logits, 0);
            prev = pred_tokens[t];
            h = dec_h;
        }
    }

    tokens_to_string(enc_tokens, seq_len, src, sizeof(src));
    for (int t = 0; t < seq_len; t++) {
        tgt_tokens[t] = enc_tokens[seq_len - 1 - t];
    }
    tgt_tokens[seq_len] = EOS_TOKEN;
    tokens_to_string(tgt_tokens, seq_len + 1, tgt, sizeof(tgt));
    tokens_to_string(pred_tokens, seq_len + 1, pred, sizeof(pred));

    printf(" sample: %s -> %s (target %s)\n", src, pred, tgt);

    free(enc_tokens);
    free(pred_tokens);
    free(tgt_tokens);
    temps_free_all(&temps);
}

int main(int argc, char **argv) {
    Seq2SeqOptions opt;
    Seq2SeqModel model;
    unsigned int seed = 1337u;
    int *enc_tokens;
    int *dec_in_tokens;
    int *dec_tgt_tokens;

    seq2seq_parse_args(argc, argv, &opt);
    if (opt.steps <= 0) opt.steps = 2000;
    if (opt.batch <= 0) opt.batch = 32;
    if (opt.embed <= 0) opt.embed = 16;
    if (opt.hidden <= 0) opt.hidden = 32;
    if (opt.min_len <= 0) opt.min_len = 3;
    if (opt.max_len < opt.min_len) opt.max_len = opt.min_len;
    if (opt.lr <= 0.0f) opt.lr = 0.03f;

    seq2seq_model_init(&model, &opt, &seed);
    enc_tokens = (int *)malloc(sizeof(int) * (size_t)opt.batch * (size_t)opt.max_len);
    dec_in_tokens = (int *)malloc(sizeof(int) * (size_t)opt.batch * (size_t)(opt.max_len + 1));
    dec_tgt_tokens = (int *)malloc(sizeof(int) * (size_t)opt.batch * (size_t)(opt.max_len + 1));
    if (!enc_tokens || !dec_in_tokens || !dec_tgt_tokens) {
        die("seq2seq batch allocation failed");
    }

    seq2seq_print_architecture(&model);
    printf("opt: steps=%d batch=%d min_len=%d max_len=%d lr=%.4f\n",
           opt.steps,
           opt.batch,
           opt.min_len,
           opt.max_len,
           opt.lr);
    for (int step = 1; step <= opt.steps; step++) {
        int seq_len = rand_int(&seed, opt.min_len, opt.max_len);
        float train_loss;
        float token_acc;
        seq2seq_sample_batch(enc_tokens, dec_in_tokens, dec_tgt_tokens, opt.batch, seq_len, &seed);
        train_loss = seq2seq_train_step(
            &model, &opt, seq_len, enc_tokens, dec_in_tokens, dec_tgt_tokens, &seed, &token_acc);
        if (step == 1 || step % 100 == 0 || step == opt.steps) {
            printf("step %4d len %d loss %.6f token_acc %.3f",
                   step,
                   seq_len,
                   train_loss,
                   token_acc);
            seq2seq_decode_example(&model, seq_len, &seed);
        }
    }

    free(enc_tokens);
    free(dec_in_tokens);
    free(dec_tgt_tokens);
    seq2seq_model_free(&model);
    return 0;
}
