#include "tensor.h"

/*
 * gpt_char.c
 *
 * Minimal GPT-like character language model demo using nanotensor.
 * - Trains a single-block causal self-attention model on raw text
 * - Uses explicit Q/K/V projections with autoregressive next-char loss
 * - Adds residual connections, layernorm, and a small feed-forward block
 * - Generates text from fixed prompts and can save a report/log artifact
 *
 * Usage:
 *   ./gpt_char [--text=PATH] [--steps=N] [--context=N] [--dim=N]
 *              [--hidden=N] [--heads=N] [--lr=FLOAT] [--temperature=FLOAT]
 *              [--prompt=TEXT] [--log=PATH] [--report=PATH]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_VOCAB 256
#define GPT_PARAM_COUNT 16

typedef struct {
    char text_path[1024];
    int steps;
    int context;
    int dim;
    int hidden;
    int heads;
    float lr;
    float temperature;
    char prompt[256];
    char log_path[1024];
    char report_path[1024];
} GPTCharOptions;

/* Architecture:
 * input chars [seq]
 * -> token embeddings + learned positional embeddings [seq, dim]
 * -> layernorm
 * -> causal self-attention via Q/K/V projections [seq, dim]
 * -> residual add
 * -> layernorm
 * -> feed-forward MLP dim -> hidden -> dim
 * -> residual add
 * -> vocab logits per position [seq, vocab]
 */
typedef struct {
    int vocab;
    int context;
    int dim;
    int hidden;
    int heads;
    Tensor *E;
    Tensor *P;
    Tensor *ln1_gamma;
    Tensor *ln1_beta;
    Tensor *W_q;
    Tensor *W_k;
    Tensor *W_v;
    Tensor *W_o;
    Tensor *ln2_gamma;
    Tensor *ln2_beta;
    Tensor *W1;
    Tensor *b1;
    Tensor *W2;
    Tensor *b2;
    Tensor *W_vocab;
    Tensor *b_vocab;
    Tensor *params[GPT_PARAM_COUNT];
    Tensor *adam_m1[GPT_PARAM_COUNT];
    Tensor *adam_m2[GPT_PARAM_COUNT];
    int adam_step;
} GPTCharModel;

static void gpt_char_print_usage(const char *prog);
static void gpt_char_parse_args(int argc, char **argv, GPTCharOptions *opt);
static char *gpt_char_read_file(const char *path, size_t *out_size);
static void gpt_char_sanitize_text(char *text, size_t n);
static int gpt_char_build_vocab(const char *text,
                                size_t n,
                                int char_to_id[256],
                                unsigned char id_to_char[MAX_VOCAB]);
static int *gpt_char_encode_text(const char *text, size_t n, const int char_to_id[256]);
static void gpt_char_model_init(GPTCharModel *model,
                                int vocab,
                                const GPTCharOptions *opt,
                                unsigned int *seed);
static void gpt_char_model_free(GPTCharModel *model);
static void gpt_char_print_architecture(const GPTCharModel *model, const GPTCharOptions *opt);
static Tensor *gpt_char_forward_logits(TensorList *temps,
                                       GPTCharModel *model,
                                       const int *tokens,
                                       int seq_len);
static float gpt_char_eval_window(GPTCharModel *model,
                                  const int *tokens,
                                  int start,
                                  int seq_len,
                                  float *out_acc);
static void gpt_char_eval(GPTCharModel *model,
                          const int *tokens,
                          int n_tokens,
                          int seq_len,
                          float *out_loss,
                          float *out_acc);
static void gpt_char_generate(FILE *out,
                              GPTCharModel *model,
                              const GPTCharOptions *opt,
                              const int char_to_id[256],
                              const unsigned char id_to_char[MAX_VOCAB],
                              const int *fallback_tokens,
                              const char *prompt,
                              int gen_len,
                              unsigned int *seed);
static void gpt_char_write_report(const GPTCharOptions *opt,
                                  GPTCharModel *model,
                                  const int char_to_id[256],
                                  const unsigned char id_to_char[MAX_VOCAB],
                                  const int *fallback_tokens,
                                  unsigned int *seed);

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static float rand_uniform(unsigned int *seed) {
    *seed = (1103515245U * (*seed) + 12345U);
    return (float)((*seed) & 0x7fffffffU) / 2147483648.0f;
}

static int sample_from_probs(const float *probs, int n, unsigned int *seed) {
    float r = rand_uniform(seed);
    float acc = 0.0f;

    for (int i = 0; i < n; i++) {
        acc += probs[i];
        if (r <= acc) {
            return i;
        }
    }
    return n - 1;
}

static char gpt_char_output_char(unsigned char ch) {
    if (ch == '\r' || ch == '\n' || ch == '\t') {
        return ' ';
    }
    if (ch >= 32 && ch <= 126) {
        return (char)ch;
    }
    return '?';
}

static int id_for_char(const int char_to_id[256], char c, int fallback_id) {
    int id = char_to_id[(unsigned char)c];
    return id >= 0 ? id : fallback_id;
}

static void gpt_char_print_usage(const char *prog) {
    printf("usage: %s [options]\n", prog);
    printf("  --text=PATH\n");
    printf("  --steps=N\n");
    printf("  --context=N\n");
    printf("  --dim=N\n");
    printf("  --hidden=N\n");
    printf("  --heads=N\n");
    printf("  --lr=FLOAT\n");
    printf("  --temperature=FLOAT\n");
    printf("  --prompt=TEXT\n");
    printf("  --log=PATH\n");
    printf("  --report=PATH\n");
}

static void gpt_char_parse_args(int argc, char **argv, GPTCharOptions *opt) {
    snprintf(opt->text_path, sizeof(opt->text_path), "%s", "data/shakespeare/shakespeare_gutenberg.txt");
    opt->steps = 2000;
    opt->context = 32;
    opt->dim = 32;
    opt->hidden = 64;
    opt->heads = 1;
    opt->lr = 0.003f;
    opt->temperature = 0.9f;
    snprintf(opt->prompt, sizeof(opt->prompt), "%s", "To be,");
    snprintf(opt->log_path, sizeof(opt->log_path), "%s", "out/gpt_char_training_log.csv");
    snprintf(opt->report_path, sizeof(opt->report_path), "%s", "out/gpt_char_report.txt");

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            gpt_char_print_usage(argv[0]);
            exit(0);
        } else if (sscanf(arg, "--text=%1023s", opt->text_path) == 1) {
            continue;
        } else if (sscanf(arg, "--steps=%d", &opt->steps) == 1) {
            continue;
        } else if (sscanf(arg, "--context=%d", &opt->context) == 1) {
            continue;
        } else if (sscanf(arg, "--dim=%d", &opt->dim) == 1) {
            continue;
        } else if (sscanf(arg, "--hidden=%d", &opt->hidden) == 1) {
            continue;
        } else if (sscanf(arg, "--heads=%d", &opt->heads) == 1) {
            continue;
        } else if (sscanf(arg, "--lr=%f", &opt->lr) == 1) {
            continue;
        } else if (sscanf(arg, "--temperature=%f", &opt->temperature) == 1) {
            continue;
        } else if (sscanf(arg, "--prompt=%255[^\n]", opt->prompt) == 1) {
            continue;
        } else if (sscanf(arg, "--log=%1023s", opt->log_path) == 1) {
            continue;
        } else if (sscanf(arg, "--report=%1023s", opt->report_path) == 1) {
            continue;
        } else {
            die("unknown option");
        }
    }
}

static char *gpt_char_read_file(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    char *buf;
    long sz;

    if (!f) {
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    sz = ftell(f);
    if (sz < 0) {
        fclose(f);
        return NULL;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }

    buf = (char *)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    buf[sz] = '\0';
    *out_size = (size_t)sz;
    return buf;
}

static void gpt_char_sanitize_text(char *text, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char ch = (unsigned char)text[i];
        if (ch == '\r') {
            text[i] = '\n';
        } else if (ch == '\n' || ch == '\t') {
            continue;
        } else if (ch < 32 || ch > 126) {
            text[i] = ' ';
        }
    }
}

static int gpt_char_build_vocab(const char *text,
                                size_t n,
                                int char_to_id[256],
                                unsigned char id_to_char[MAX_VOCAB]) {
    int vocab = 0;

    for (int i = 0; i < 256; i++) {
        char_to_id[i] = -1;
    }
    for (size_t i = 0; i < n; i++) {
        unsigned char ch = (unsigned char)text[i];
        if (char_to_id[ch] == -1) {
            if (vocab >= MAX_VOCAB) {
                return -1;
            }
            char_to_id[ch] = vocab;
            id_to_char[vocab] = ch;
            vocab++;
        }
    }
    return vocab;
}

static int *gpt_char_encode_text(const char *text, size_t n, const int char_to_id[256]) {
    int *tok = (int *)malloc(sizeof(int) * n);

    if (!tok) {
        return NULL;
    }
    for (size_t i = 0; i < n; i++) {
        tok[i] = char_to_id[(unsigned char)text[i]];
    }
    return tok;
}

static void gpt_char_model_init(GPTCharModel *model,
                                int vocab,
                                const GPTCharOptions *opt,
                                unsigned int *seed) {
    memset(model, 0, sizeof(*model));
    model->vocab = vocab;
    model->context = opt->context;
    model->dim = opt->dim;
    model->hidden = opt->hidden;
    model->heads = opt->heads;

    model->E = tensor_create(vocab, opt->dim, 1);
    model->P = tensor_create(opt->context, opt->dim, 1);
    model->ln1_gamma = tensor_create(1, opt->dim, 1);
    model->ln1_beta = tensor_create(1, opt->dim, 1);
    model->W_q = tensor_create(opt->dim, opt->dim, 1);
    model->W_k = tensor_create(opt->dim, opt->dim, 1);
    model->W_v = tensor_create(opt->dim, opt->dim, 1);
    model->W_o = tensor_create(opt->dim, opt->dim, 1);
    model->ln2_gamma = tensor_create(1, opt->dim, 1);
    model->ln2_beta = tensor_create(1, opt->dim, 1);
    model->W1 = tensor_create(opt->dim, opt->hidden, 1);
    model->b1 = tensor_create(1, opt->hidden, 1);
    model->W2 = tensor_create(opt->hidden, opt->dim, 1);
    model->b2 = tensor_create(1, opt->dim, 1);
    model->W_vocab = tensor_create(opt->dim, vocab, 1);
    model->b_vocab = tensor_create(1, vocab, 1);

    tensor_fill_randn(model->E, 0.0f, 0.05f, seed);
    tensor_fill_randn(model->P, 0.0f, 0.05f, seed);
    tensor_fill(model->ln1_gamma, 1.0f);
    tensor_fill(model->ln1_beta, 0.0f);
    tensor_fill_randn(model->W_q, 0.0f, 0.05f, seed);
    tensor_fill_randn(model->W_k, 0.0f, 0.05f, seed);
    tensor_fill_randn(model->W_v, 0.0f, 0.05f, seed);
    tensor_fill_randn(model->W_o, 0.0f, 0.05f, seed);
    tensor_fill(model->ln2_gamma, 1.0f);
    tensor_fill(model->ln2_beta, 0.0f);
    tensor_fill_randn(model->W1, 0.0f, 0.05f, seed);
    tensor_fill(model->b1, 0.0f);
    tensor_fill_randn(model->W2, 0.0f, 0.05f, seed);
    tensor_fill(model->b2, 0.0f);
    tensor_fill_randn(model->W_vocab, 0.0f, 0.05f, seed);
    tensor_fill(model->b_vocab, 0.0f);

    model->params[0] = model->E;
    model->params[1] = model->P;
    model->params[2] = model->ln1_gamma;
    model->params[3] = model->ln1_beta;
    model->params[4] = model->W_q;
    model->params[5] = model->W_k;
    model->params[6] = model->W_v;
    model->params[7] = model->W_o;
    model->params[8] = model->ln2_gamma;
    model->params[9] = model->ln2_beta;
    model->params[10] = model->W1;
    model->params[11] = model->b1;
    model->params[12] = model->W2;
    model->params[13] = model->b2;
    model->params[14] = model->W_vocab;
    model->params[15] = model->b_vocab;

    for (int i = 0; i < GPT_PARAM_COUNT; i++) {
        model->adam_m1[i] = tensor_create(model->params[i]->rows, model->params[i]->cols, 0);
        model->adam_m2[i] = tensor_create(model->params[i]->rows, model->params[i]->cols, 0);
    }
}

static void gpt_char_model_free(GPTCharModel *model) {
    for (int i = 0; i < GPT_PARAM_COUNT; i++) {
        tensor_free(model->params[i]);
        tensor_free(model->adam_m1[i]);
        tensor_free(model->adam_m2[i]);
    }
    memset(model, 0, sizeof(*model));
}

static void gpt_char_print_architecture(const GPTCharModel *model, const GPTCharOptions *opt) {
    printf("arch: vocab=%d context=%d dim=%d hidden=%d heads=%d\n",
           model->vocab,
           opt->context,
           model->dim,
           model->hidden,
           model->heads);
    printf("arch: block=ln -> qkv self_attn(%d heads) -> resid\n", model->heads);
    printf("arch: block=ln -> ff dim->hidden->dim -> resid\n");
    printf("arch: head=causal char lm\n");
}

static Tensor *gpt_char_forward_logits(TensorList *temps,
                                       GPTCharModel *model,
                                       const int *tokens,
                                       int seq_len) {
    Tensor *x_onehot = tensor_one_hot(tokens, seq_len, model->vocab);
    Tensor *tok_embed = tensor_matmul(x_onehot, model->E);
    Tensor *pos_embed = tensor_slice(model->P, 0, seq_len, 0, model->dim);
    Tensor *x = tensor_add(tok_embed, pos_embed);
    Tensor *x_norm = tensor_layernorm(x, model->ln1_gamma, model->ln1_beta, 1e-5f);
    Tensor *q = tensor_matmul(x_norm, model->W_q);
    Tensor *k = tensor_matmul(x_norm, model->W_k);
    Tensor *v = tensor_matmul(x_norm, model->W_v);
    Tensor *head_ctx = NULL;
    Tensor *head_ctx_cols = NULL;
    Tensor *attn_heads = NULL;
    Tensor *attn_ctx;
    Tensor *attn_proj;
    Tensor *resid1;
    Tensor *resid1_norm;
    Tensor *ff1_lin;
    Tensor *ff1_bias;
    Tensor *ff1;
    Tensor *ff2_lin;
    Tensor *ff2_bias;
    Tensor *resid2;
    Tensor *logits_lin;
    Tensor *logits;
    const int head_dim = model->dim / model->heads;
    const float scale = 1.0f / sqrtf((float)head_dim);

    tensor_list_add(temps, x_onehot);
    tensor_list_add(temps, tok_embed);
    tensor_list_add(temps, pos_embed);
    tensor_list_add(temps, x);
    tensor_list_add(temps, x_norm);
    tensor_list_add(temps, q);
    tensor_list_add(temps, k);
    tensor_list_add(temps, v);

    /* Multi-head causal attention is built one head at a time and one query
     * row at a time so each position only attends to the visible prefix.
     */
    for (int h = 0; h < model->heads; h++) {
        int col0 = h * head_dim;
        int col1 = col0 + head_dim;
        Tensor *q_head = tensor_slice(q, 0, seq_len, col0, col1);
        Tensor *k_head = tensor_slice(k, 0, seq_len, col0, col1);
        Tensor *v_head = tensor_slice(v, 0, seq_len, col0, col1);

        tensor_list_add(temps, q_head);
        tensor_list_add(temps, k_head);
        tensor_list_add(temps, v_head);

        head_ctx = NULL;
        head_ctx_cols = NULL;
        for (int t = 0; t < seq_len; t++) {
            Tensor *q_t = tensor_slice(q_head, t, t + 1, 0, head_dim);
            Tensor *k_prefix = tensor_slice(k_head, 0, t + 1, 0, head_dim);
            Tensor *v_prefix = tensor_slice(v_head, 0, t + 1, 0, head_dim);
            Tensor *k_prefix_t = tensor_transpose(k_prefix);
            Tensor *scores = tensor_matmul(q_t, k_prefix_t);
            Tensor *scaled = tensor_scalar_mul(scores, scale);
            Tensor *weights = tensor_softmax(scaled);
            Tensor *ctx_t = tensor_matmul(weights, v_prefix);
            Tensor *ctx_col = tensor_transpose(ctx_t);

            tensor_list_add(temps, q_t);
            tensor_list_add(temps, k_prefix);
            tensor_list_add(temps, v_prefix);
            tensor_list_add(temps, k_prefix_t);
            tensor_list_add(temps, scores);
            tensor_list_add(temps, scaled);
            tensor_list_add(temps, weights);
            if (!head_ctx_cols) {
                head_ctx_cols = ctx_col;
                tensor_list_add(temps, head_ctx_cols);
            } else {
                tensor_list_add(temps, ctx_t);
                head_ctx_cols = tensor_concat_cols(head_ctx_cols, ctx_col);
                tensor_list_add(temps, ctx_col);
                tensor_list_add(temps, head_ctx_cols);
            }
        }

        head_ctx = tensor_transpose(head_ctx_cols);
        tensor_list_add(temps, head_ctx);
        if (!attn_heads) {
            attn_heads = head_ctx;
        } else {
            attn_heads = tensor_concat_cols(attn_heads, head_ctx);
            tensor_list_add(temps, attn_heads);
        }
    }

    attn_ctx = attn_heads;
    attn_proj = tensor_matmul(attn_ctx, model->W_o);
    resid1 = tensor_add(x, attn_proj);
    resid1_norm = tensor_layernorm(resid1, model->ln2_gamma, model->ln2_beta, 1e-5f);
    ff1_lin = tensor_matmul(resid1_norm, model->W1);
    ff1_bias = tensor_add_bias(ff1_lin, model->b1);
    ff1 = tensor_relu(ff1_bias);
    ff2_lin = tensor_matmul(ff1, model->W2);
    ff2_bias = tensor_add_bias(ff2_lin, model->b2);
    resid2 = tensor_add(resid1, ff2_bias);
    logits_lin = tensor_matmul(resid2, model->W_vocab);
    logits = tensor_add_bias(logits_lin, model->b_vocab);

    tensor_list_add(temps, attn_proj);
    tensor_list_add(temps, resid1);
    tensor_list_add(temps, resid1_norm);
    tensor_list_add(temps, ff1_lin);
    tensor_list_add(temps, ff1_bias);
    tensor_list_add(temps, ff1);
    tensor_list_add(temps, ff2_lin);
    tensor_list_add(temps, ff2_bias);
    tensor_list_add(temps, resid2);
    tensor_list_add(temps, logits_lin);
    tensor_list_add(temps, logits);
    return logits;
}

static float gpt_char_eval_window(GPTCharModel *model,
                                  const int *tokens,
                                  int start,
                                  int seq_len,
                                  float *out_acc) {
    TensorList temps = {0};
    Tensor *logits = gpt_char_forward_logits(&temps, model, tokens + start, seq_len);
    Tensor *targets = tensor_one_hot(tokens + start + 1, seq_len, model->vocab);
    Tensor *probs = tensor_softmax(logits);
    Tensor *loss = tensor_cross_entropy(probs, targets);
    int correct = 0;

    tensor_list_add(&temps, targets);
    tensor_list_add(&temps, probs);
    tensor_list_add(&temps, loss);

    for (int i = 0; i < seq_len; i++) {
        if (tensor_argmax_row(logits, i) == tokens[start + 1 + i]) {
            correct++;
        }
    }
    if (out_acc) {
        *out_acc = seq_len > 0 ? (float)correct / (float)seq_len : 0.0f;
    }

    {
        float loss_value = loss->data[0];
        tensor_list_free(&temps);
        return loss_value;
    }
}

static void gpt_char_eval(GPTCharModel *model,
                          const int *tokens,
                          int n_tokens,
                          int seq_len,
                          float *out_loss,
                          float *out_acc) {
    const int eval_windows = 8;
    float total_loss = 0.0f;
    float total_acc = 0.0f;
    int counted = 0;
    int max_start = n_tokens - seq_len - 1;

    if (max_start <= 0) {
        if (out_loss) *out_loss = -1.0f;
        if (out_acc) *out_acc = 0.0f;
        return;
    }

    for (int i = 0; i < eval_windows; i++) {
        int start = (i * max_start) / eval_windows;
        float acc = 0.0f;
        float loss = gpt_char_eval_window(model, tokens, start, seq_len, &acc);
        total_loss += loss;
        total_acc += acc;
        counted++;
    }

    if (out_loss) {
        *out_loss = counted > 0 ? total_loss / (float)counted : -1.0f;
    }
    if (out_acc) {
        *out_acc = counted > 0 ? total_acc / (float)counted : 0.0f;
    }
}

static void gpt_char_generate(FILE *out,
                              GPTCharModel *model,
                              const GPTCharOptions *opt,
                              const int char_to_id[256],
                              const unsigned char id_to_char[MAX_VOCAB],
                              const int *fallback_tokens,
                              const char *prompt,
                              int gen_len,
                              unsigned int *seed) {
    int total_cap = (int)strlen(prompt) + gen_len + opt->context + 1;
    int *gen = (int *)malloc(sizeof(int) * (size_t)total_cap);
    int gen_len_tokens = 0;

    fprintf(out, "%s", prompt);
    for (size_t i = 0; i < strlen(prompt); i++) {
        gen[gen_len_tokens++] = id_for_char(char_to_id, prompt[i], fallback_tokens[0]);
    }
    if (gen_len_tokens == 0) {
        gen[gen_len_tokens++] = fallback_tokens[0];
    }

    for (int step = 0; step < gen_len; step++) {
        int window = gen_len_tokens < opt->context ? gen_len_tokens : opt->context;
        int start = gen_len_tokens - window;
        TensorList temps = {0};
        Tensor *logits = gpt_char_forward_logits(&temps, model, gen + start, window);
        Tensor *last = tensor_slice(logits, window - 1, window, 0, model->vocab);
        Tensor *scaled = tensor_scalar_mul(last, 1.0f / (opt->temperature > 1e-6f ? opt->temperature : 1e-6f));
        Tensor *probs = tensor_softmax(scaled);
        int next = sample_from_probs(probs->data, model->vocab, seed);

        tensor_list_add(&temps, last);
        tensor_list_add(&temps, scaled);
        tensor_list_add(&temps, probs);
        gen[gen_len_tokens++] = next;
        fputc(gpt_char_output_char(id_to_char[next]), out);
        tensor_list_free(&temps);
    }
    fputc('\n', out);
    free(gen);
}

static void gpt_char_write_report(const GPTCharOptions *opt,
                                  GPTCharModel *model,
                                  const int char_to_id[256],
                                  const unsigned char id_to_char[MAX_VOCAB],
                                  const int *fallback_tokens,
                                  unsigned int *seed) {
    static const char *prompts[] = {"To be,", "What ", "KING ", "ROMEO "};
    FILE *out = fopen(opt->report_path, "w");

    if (!out) {
        fprintf(stderr, "failed to open report '%s'\n", opt->report_path);
        return;
    }

    fprintf(out, "gpt_char report\n");
    fprintf(out,
            "context=%d dim=%d hidden=%d heads=%d steps=%d lr=%.4f\n\n",
            opt->context,
            opt->dim,
            opt->hidden,
            opt->heads,
            opt->steps,
            opt->lr);
    for (size_t i = 0; i < sizeof(prompts) / sizeof(prompts[0]); i++) {
        fprintf(out, "prompt: ");
        gpt_char_generate(out, model, opt, char_to_id, id_to_char, fallback_tokens, prompts[i], 96, seed);
        fprintf(out, "\n");
    }
    fclose(out);
}

int main(int argc, char **argv) {
    GPTCharOptions opt;
    size_t text_len = 0;
    char *text;
    int char_to_id[256];
    unsigned char id_to_char[MAX_VOCAB];
    int *tokens;
    int vocab;
    int split;
    int train_len;
    int eval_len;
    int *train_tokens;
    int *eval_tokens;
    unsigned int seed = 1337u;
    GPTCharModel model;
    FILE *logf;

    gpt_char_parse_args(argc, argv, &opt);
    text = gpt_char_read_file(opt.text_path, &text_len);
    if (!text) {
        fprintf(stderr, "failed to read '%s'\n", opt.text_path);
        return 1;
    }
    gpt_char_sanitize_text(text, text_len);
    if (opt.steps <= 0) opt.steps = 2000;
    if (opt.context <= 1) opt.context = 32;
    if (opt.dim <= 0) opt.dim = 32;
    if (opt.hidden <= 0) opt.hidden = 64;
    if (opt.heads <= 0) opt.heads = 1;
    if (opt.lr <= 0.0f) opt.lr = 0.003f;
    if (opt.dim % opt.heads != 0) {
        fprintf(stderr, "dim=%d must be divisible by heads=%d\n", opt.dim, opt.heads);
        free(text);
        return 1;
    }
    if (text_len < (size_t)(opt.context + 2)) {
        fprintf(stderr, "text corpus is too small for context=%d\n", opt.context);
        free(text);
        return 1;
    }

    vocab = gpt_char_build_vocab(text, text_len, char_to_id, id_to_char);
    if (vocab <= 1) {
        fprintf(stderr, "failed to build vocab\n");
        free(text);
        return 1;
    }
    tokens = gpt_char_encode_text(text, text_len, char_to_id);
    if (!tokens) {
        fprintf(stderr, "failed to encode text\n");
        free(text);
        return 1;
    }

    split = (int)((float)text_len * 0.9f);
    if (split < opt.context + 2) {
        split = (int)text_len - (opt.context + 2);
    }
    if (split < opt.context + 2) {
        split = opt.context + 2;
    }
    train_len = split;
    eval_len = (int)text_len - split;
    train_tokens = tokens;
    eval_tokens = tokens + split;

    gpt_char_model_init(&model, vocab, &opt, &seed);
    gpt_char_print_architecture(&model, &opt);
    printf("opt: text=%s\n", opt.text_path);
    printf("opt: steps=%d context=%d heads=%d lr=%.4f prompt=%s\n",
           opt.steps,
           opt.context,
           opt.heads,
           opt.lr,
           opt.prompt);
    printf("opt: train=%d eval=%d vocab=%d log=%s\n", train_len, eval_len, vocab, opt.log_path);

    logf = fopen(opt.log_path, "w");
    if (!logf) {
        die("failed to open gpt_char log file");
    }
    fprintf(logf, "step,train_loss,train_acc,eval_loss,eval_acc\n");

    for (int step = 1; step <= opt.steps; step++) {
        int start = (int)(rand_uniform(&seed) * (float)(train_len - opt.context - 1));
        TensorList temps = {0};
        Tensor *logits = gpt_char_forward_logits(&temps, &model, train_tokens + start, opt.context);
        Tensor *targets = tensor_one_hot(train_tokens + start + 1, opt.context, vocab);
        Tensor *probs = tensor_softmax(logits);
        Tensor *loss = tensor_cross_entropy(probs, targets);
        int correct = 0;

        tensor_list_add(&temps, targets);
        tensor_list_add(&temps, probs);
        tensor_list_add(&temps, loss);

        for (int i = 0; i < opt.context; i++) {
            if (tensor_argmax_row(logits, i) == train_tokens[start + 1 + i]) {
                correct++;
            }
        }

        tensor_backward(loss);
        {
            TensorAdamOptions adam = {0};
            adam.lr = opt.lr;
            adam.beta1 = 0.9f;
            adam.beta2 = 0.999f;
            adam.eps = 1e-8f;
            adam.timestep = ++model.adam_step;
            tensor_adam_step(model.params, model.adam_m1, model.adam_m2, GPT_PARAM_COUNT, &adam);
        }

        if (step == 1 || step % 100 == 0 || step == opt.steps) {
            float train_acc = (float)correct / (float)opt.context;
            float eval_loss = -1.0f;
            float eval_acc = 0.0f;
            gpt_char_eval(&model, eval_tokens, eval_len, opt.context, &eval_loss, &eval_acc);
            printf("step %4d loss %.6f train_acc %.3f eval_loss %.6f eval_acc %.3f\n",
                   step,
                   loss->data[0],
                   train_acc,
                   eval_loss,
                   eval_acc);
            fprintf(logf, "%d,%.6f,%.6f,%.6f,%.6f\n", step, loss->data[0], train_acc, eval_loss, eval_acc);
            fflush(logf);
        }

        tensor_list_free(&temps);
    }

    printf("\n--- generation ---\n");
    gpt_char_generate(stdout, &model, &opt, char_to_id, id_to_char, train_tokens, opt.prompt, 200, &seed);
    gpt_char_write_report(&opt, &model, char_to_id, id_to_char, train_tokens, &seed);
    printf("saved report: %s\n", opt.report_path);

    fclose(logf);
    gpt_char_model_free(&model);
    free(tokens);
    free(text);
    return 0;
}
