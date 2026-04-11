#include "tensor.h"

/*
 * seq2seq.c
 *
 * Minimal seq2seq demo using nanotensor.
 * - Trains on a synthetic digit-sequence translation task
 * - Learns to map each input digit string to its reversed output string
 * - Uses a tanh RNN encoder and tanh RNN decoder over a tiny BOS/EOS vocabulary
 * - Keeps the first version intentionally small and readable
 *
 * Usage:
 *   ./seq2seq [--steps=N] [--batch=N] [--embed=N]
 *             [--hidden=N] [--lr=FLOAT] [--min-len=N]
 *             [--max-len=N] [--log=PATH]
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
    const char *log_path;
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

typedef struct {
    int batch;
    int seq_len;
    int *enc_tokens;
    int *dec_in_tokens;
    int *dec_tgt_tokens;
} Seq2SeqBatch;

static void seq2seq_print_usage(const char *prog);
static void seq2seq_parse_args(int argc, char **argv, Seq2SeqOptions *opt);
static void seq2seq_model_init(Seq2SeqModel *model, const Seq2SeqOptions *opt, unsigned int *seed);
static void seq2seq_model_free(Seq2SeqModel *model);
static void seq2seq_print_architecture(const Seq2SeqModel *model);
static void seq2seq_batch_init(Seq2SeqBatch *batch, int batch_size, int max_len);
static void seq2seq_batch_free(Seq2SeqBatch *batch);
static int seq2seq_curriculum_max_len(const Seq2SeqOptions *opt, int step);
static void seq2seq_sample_batch(Seq2SeqBatch *batch, unsigned int *seed);
static float seq2seq_train_step(Seq2SeqModel *model,
                                const Seq2SeqOptions *opt,
                                const Seq2SeqBatch *batch,
                                unsigned int *seed,
                                float *out_token_acc);
static void seq2seq_predict_batch(Seq2SeqModel *model,
                                  const Seq2SeqBatch *batch,
                                  int *pred_tokens);
static void seq2seq_eval(Seq2SeqModel *model,
                         const Seq2SeqOptions *opt,
                         unsigned int *seed,
                         float *out_token_acc,
                         float *out_exact_acc);
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
    /* Shared tanh RNN cell used by both the encoder and decoder:
     * token ids -> one-hot -> embedding -> input/hidden projections -> tanh hidden state.
     */
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
    printf("  --log=PATH\n");
}

static void seq2seq_parse_args(int argc, char **argv, Seq2SeqOptions *opt) {
    opt->steps = 2000;
    opt->batch = 32;
    opt->embed = 16;
    opt->hidden = 32;
    opt->min_len = 3;
    opt->max_len = 8;
    opt->lr = 0.03f;
    opt->log_path = "out/seq2seq_training_log.csv";

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
        } else if (strncmp(arg, "--log=", 6) == 0) {
            opt->log_path = argv[i] + 6;
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

static void seq2seq_batch_init(Seq2SeqBatch *batch, int batch_size, int max_len) {
    memset(batch, 0, sizeof(*batch));
    batch->batch = batch_size;
    batch->enc_tokens = (int *)malloc(sizeof(int) * (size_t)batch_size * (size_t)max_len);
    batch->dec_in_tokens = (int *)malloc(sizeof(int) * (size_t)batch_size * (size_t)(max_len + 1));
    batch->dec_tgt_tokens = (int *)malloc(sizeof(int) * (size_t)batch_size * (size_t)(max_len + 1));
    if (!batch->enc_tokens || !batch->dec_in_tokens || !batch->dec_tgt_tokens) {
        die("seq2seq batch allocation failed");
    }
}

static void seq2seq_batch_free(Seq2SeqBatch *batch) {
    if (!batch) {
        return;
    }
    free(batch->enc_tokens);
    free(batch->dec_in_tokens);
    free(batch->dec_tgt_tokens);
    memset(batch, 0, sizeof(*batch));
}

static int seq2seq_curriculum_max_len(const Seq2SeqOptions *opt, int step) {
    int span = opt->max_len - opt->min_len;
    int stage1_max = opt->min_len + (span >= 1 ? span / 3 : 0);
    int stage2_max = opt->min_len + (span >= 2 ? (2 * span) / 3 : 0);

    if (stage1_max < opt->min_len) {
        stage1_max = opt->min_len;
    }
    if (stage2_max < stage1_max) {
        stage2_max = stage1_max;
    }
    if (stage1_max > opt->max_len) {
        stage1_max = opt->max_len;
    }
    if (stage2_max > opt->max_len) {
        stage2_max = opt->max_len;
    }

    if (step <= opt->steps / 3) {
        return stage1_max;
    }
    if (step <= (2 * opt->steps) / 3) {
        return stage2_max;
    }
    return opt->max_len;
}

static void seq2seq_sample_batch(Seq2SeqBatch *batch, unsigned int *seed) {
    for (int b = 0; b < batch->batch; b++) {
        for (int t = 0; t < batch->seq_len; t++) {
            int digit = rand_int(seed, 0, 9);
            batch->enc_tokens[t * batch->batch + b] = digit;
            batch->dec_tgt_tokens[t * batch->batch + b] =
                batch->enc_tokens[(batch->seq_len - 1 - t) * batch->batch + b];
        }
        batch->dec_in_tokens[b] = BOS_TOKEN;
        for (int t = 1; t < batch->seq_len + 1; t++) {
            batch->dec_in_tokens[t * batch->batch + b] = batch->dec_tgt_tokens[(t - 1) * batch->batch + b];
        }
        batch->dec_tgt_tokens[batch->seq_len * batch->batch + b] = EOS_TOKEN;
    }
}

static float seq2seq_train_step(Seq2SeqModel *model,
                                const Seq2SeqOptions *opt,
                                const Seq2SeqBatch *batch,
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

    /* Encode the source sequence left-to-right. The final hidden state becomes
     * the fixed context passed into the decoder.
     */
    for (int t = 0; t < batch->seq_len; t++) {
        h = seq2seq_rnn_step(&temps,
                             model,
                             batch->enc_tokens + t * batch->batch,
                             opt->batch,
                             h,
                             model->W_enc_x,
                             model->W_enc_h,
                             model->b_enc);
    }

    /* Decode with teacher forcing: each decoder step consumes the previous
     * ground-truth output token and predicts the next target token.
     */
    for (int t = 0; t < batch->seq_len + 1; t++) {
        Tensor *dec_h = seq2seq_rnn_step(&temps,
                                         model,
                                         batch->dec_in_tokens + t * batch->batch,
                                         opt->batch,
                                         h,
                                         model->W_dec_x,
                                         model->W_dec_h,
                                         model->b_dec);
        Tensor *out_lin = tensor_matmul(dec_h, model->W_out);
        Tensor *logits = tensor_add_bias(out_lin, model->b_out);
        Tensor *probs = tensor_softmax(logits);
        Tensor *target = tensor_one_hot(batch->dec_tgt_tokens + t * batch->batch, opt->batch, VOCAB_SIZE);
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
            if (pred == batch->dec_tgt_tokens[t * batch->batch + b]) {
                correct++;
            }
            total++;
        }
    }

    if (batch->seq_len + 1 > 1) {
        total_loss = tensor_scalar_mul(total_loss, 1.0f / (float)(batch->seq_len + 1));
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

static void seq2seq_predict_batch(Seq2SeqModel *model, const Seq2SeqBatch *batch, int *pred_tokens) {
    TensorTemps temps = {0};
    Tensor *h = tensor_create(batch->batch, model->hidden, 0);
    int *prev_tokens = (int *)malloc(sizeof(int) * (size_t)batch->batch);

    if (!h || !prev_tokens) {
        die("seq2seq predict allocation failed");
    }

    tensor_fill(h, 0.0f);
    temps_push(&temps, h);
    /* Rebuild the encoder final state for this batch before greedy decoding. */
    for (int t = 0; t < batch->seq_len; t++) {
        h = seq2seq_rnn_step(&temps,
                             model,
                             batch->enc_tokens + t * batch->batch,
                             batch->batch,
                             h,
                             model->W_enc_x,
                             model->W_enc_h,
                             model->b_enc);
    }

    for (int b = 0; b < batch->batch; b++) {
        prev_tokens[b] = BOS_TOKEN;
    }
    /* Greedy autoregressive decode: feed back each predicted token into the
     * next decoder step.
     */
    for (int t = 0; t < batch->seq_len + 1; t++) {
        Tensor *dec_h = seq2seq_rnn_step(
            &temps, model, prev_tokens, batch->batch, h, model->W_dec_x, model->W_dec_h, model->b_dec);
        Tensor *out_lin = tensor_matmul(dec_h, model->W_out);
        Tensor *logits = tensor_add_bias(out_lin, model->b_out);

        temps_push(&temps, out_lin);
        temps_push(&temps, logits);
        for (int b = 0; b < batch->batch; b++) {
            int pred = tensor_argmax_row(logits, b);
            pred_tokens[t * batch->batch + b] = pred;
            prev_tokens[b] = pred;
        }
        h = dec_h;
    }

    free(prev_tokens);
    temps_free_all(&temps);
}

static void seq2seq_eval(Seq2SeqModel *model,
                         const Seq2SeqOptions *opt,
                         unsigned int *seed,
                         float *out_token_acc,
                         float *out_exact_acc) {
    const int eval_batches = 8;
    Seq2SeqBatch batch = {0};
    int *pred_tokens = (int *)malloc(sizeof(int) * (size_t)opt->batch * (size_t)(opt->max_len + 1));
    int total = 0;
    int correct = 0;
    int exact_total = 0;
    int exact_correct = 0;

    seq2seq_batch_init(&batch, opt->batch, opt->max_len);
    if (!pred_tokens) {
        die("seq2seq eval allocation failed");
    }

    for (int i = 0; i < eval_batches; i++) {
        batch.seq_len = rand_int(seed, opt->min_len, opt->max_len);
        seq2seq_sample_batch(&batch, seed);
        seq2seq_predict_batch(model, &batch, pred_tokens);

        for (int b = 0; b < opt->batch; b++) {
            int match = 1;
            for (int t = 0; t < batch.seq_len + 1; t++) {
                int pred = pred_tokens[t * opt->batch + b];
                int tgt = batch.dec_tgt_tokens[t * opt->batch + b];
                if (pred == tgt) {
                    correct++;
                } else {
                    match = 0;
                }
                total++;
            }
            if (match) {
                exact_correct++;
            }
            exact_total++;
        }
    }

    if (out_token_acc) {
        *out_token_acc = total > 0 ? (float)correct / (float)total : 0.0f;
    }
    if (out_exact_acc) {
        *out_exact_acc = exact_total > 0 ? (float)exact_correct / (float)exact_total : 0.0f;
    }

    seq2seq_batch_free(&batch);
    free(pred_tokens);
}

static void seq2seq_decode_example(Seq2SeqModel *model, int seq_len, unsigned int *seed) {
    Seq2SeqBatch batch = {0};
    int *pred_tokens = (int *)malloc(sizeof(int) * (size_t)(seq_len + 1));
    int *tgt_tokens = (int *)malloc(sizeof(int) * (size_t)(seq_len + 1));
    char src[64];
    char tgt[64];
    char pred[64];

    seq2seq_batch_init(&batch, 1, seq_len);
    batch.seq_len = seq_len;

    if (!pred_tokens || !tgt_tokens) {
        die("seq2seq decode allocation failed");
    }

    for (int t = 0; t < seq_len; t++) {
        batch.enc_tokens[t] = rand_int(seed, 0, 9);
    }
    seq2seq_predict_batch(model, &batch, pred_tokens);

    tokens_to_string(batch.enc_tokens, seq_len, src, sizeof(src));
    for (int t = 0; t < seq_len; t++) {
        tgt_tokens[t] = batch.enc_tokens[seq_len - 1 - t];
    }
    tgt_tokens[seq_len] = EOS_TOKEN;
    tokens_to_string(tgt_tokens, seq_len + 1, tgt, sizeof(tgt));
    tokens_to_string(pred_tokens, seq_len + 1, pred, sizeof(pred));

    printf(" sample: %s -> %s (target %s)\n", src, pred, tgt);

    seq2seq_batch_free(&batch);
    free(pred_tokens);
    free(tgt_tokens);
}

int main(int argc, char **argv) {
    Seq2SeqOptions opt;
    Seq2SeqModel model;
    Seq2SeqBatch batch = {0};
    FILE *logf;
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
    seq2seq_batch_init(&batch, opt.batch, opt.max_len);

    seq2seq_print_architecture(&model);
    printf("opt: steps=%d batch=%d min_len=%d max_len=%d lr=%.4f\n",
           opt.steps,
           opt.batch,
           opt.min_len,
           opt.max_len,
           opt.lr);
    logf = fopen(opt.log_path, "w");
    if (!logf) {
        die("failed to open seq2seq log file");
    }
    fprintf(logf, "step,seq_len,curriculum_max_len,train_loss,train_tok,eval_tok,eval_seq\n");
    for (int step = 1; step <= opt.steps; step++) {
        int curriculum_max_len;
        float train_loss;
        float token_acc;
        float eval_token_acc;
        float eval_exact_acc;
        curriculum_max_len = seq2seq_curriculum_max_len(&opt, step);
        batch.seq_len = rand_int(&seed, opt.min_len, curriculum_max_len);
        seq2seq_sample_batch(&batch, &seed);
        train_loss = seq2seq_train_step(&model, &opt, &batch, &seed, &token_acc);
        if (step == 1 || step % 100 == 0 || step == opt.steps) {
            unsigned int eval_seed = 4242u + (unsigned int)step;
            seq2seq_eval(&model, &opt, &eval_seed, &eval_token_acc, &eval_exact_acc);
            printf("step %4d len %d cur_max %d loss %.6f train_tok %.3f eval_tok %.3f eval_seq %.3f",
                   step,
                   batch.seq_len,
                   curriculum_max_len,
                   train_loss,
                   token_acc,
                   eval_token_acc,
                   eval_exact_acc);
            fprintf(logf, "%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
                    step,
                    batch.seq_len,
                    curriculum_max_len,
                    train_loss,
                    token_acc,
                    eval_token_acc,
                    eval_exact_acc);
            fflush(logf);
            seq2seq_decode_example(&model, batch.seq_len, &seed);
        }
    }

    fclose(logf);
    seq2seq_batch_free(&batch);
    seq2seq_model_free(&model);
    return 0;
}
