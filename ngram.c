#include "tensor.h"
#include "vocab.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * ngram.c
 *
 * Minimal neural n-gram language model demo using nanotensor.
 * - Builds a word vocabulary from a text corpus
 * - Uses a fixed context window of previous words
 * - Learns shared embeddings and predicts the next word
 * - Reports held-out loss/perplexity and prints a short greedy sample
 *
 * Usage:
 *   ./ngram [--text=PATH] [--steps=N] [--batch=N] [--context=N]
 *           [--lr=FLOAT] [--embed=N] [--hidden=N] [--vocab=N]
 *           [--snapshot=PATH] [--vocab-out=PATH]
 */

#define DEFAULT_VOCAB 1000

typedef struct {
    char text_path[1024];
    int steps;
    int batch;
    int context;
    float lr;
    int embed;
    int hidden;
    int vocab_limit;
    char snapshot_path[1024];
    char vocab_out_path[1024];
} NGramOptions;

typedef struct {
    int vocab;
    int embed;
    int hidden;
    int context;
    Tensor *E;
    Tensor **W_ctx;
    Tensor *b_h;
    Tensor *W_out;
    Tensor *b_out;
    Tensor **velocity;
    Tensor **params;
    size_t n_params;
    Tensor **tmp_x_list;
    Tensor **tmp_embed_list;
    Tensor **tmp_proj_list;
    Tensor **tmp_sum_list;
    int tmp_n_sums;
    Tensor *tmp_hidden_sum;
    Tensor *tmp_hidden_bias;
    Tensor *tmp_hidden_act;
    Tensor *tmp_logits;
} NGramModel;

static int ngram_init(NGramModel *m, const NGramOptions *opt, int vocab, unsigned int *seed);
static void ngram_free(NGramModel *m);
static void ngram_clear_forward_temps(NGramModel *model);
static Tensor *ngram_forward(NGramModel *model, const int *context_ids, int batch);
static float ngram_eval(NGramModel *model, const int *tokens, int n_tokens, int batch, int context);
static void ngram_predict_topk(NGramModel *model, const EncodedCorpus *corpus, const char *prompt, int top_k);
static void ngram_generate_sample(NGramModel *model,
                                  const EncodedCorpus *corpus,
                                  const char *prompt,
                                  int length);

static float rand_uniform(unsigned int *seed) {
    *seed = (1103515245U * (*seed) + 12345U);
    return (float)((*seed) & 0x7fffffffU) / 2147483648.0f;
}

static void print_usage(const char *prog) {
    printf("usage: %s [options]\n", prog);
    printf("  --text=PATH\n");
    printf("  --steps=N\n");
    printf("  --batch=N\n");
    printf("  --context=N\n");
    printf("  --lr=FLOAT\n");
    printf("  --embed=N\n");
    printf("  --hidden=N\n");
    printf("  --vocab=N\n");
    printf("  --snapshot=PATH\n");
    printf("  --vocab-out=PATH\n");
}

static void parse_args(int argc, char **argv, NGramOptions *opt) {
    snprintf(opt->text_path, sizeof(opt->text_path), "%s", "data/shakespeare/shakespeare_gutenberg.txt");
    opt->steps = 2000;
    opt->batch = 64;
    opt->context = 3;
    opt->lr = 0.03f;
    opt->embed = 32;
    opt->hidden = 64;
    opt->vocab_limit = DEFAULT_VOCAB;
    snprintf(opt->snapshot_path, sizeof(opt->snapshot_path), "%s", "out/ngram_snapshot.bin");
    snprintf(opt->vocab_out_path, sizeof(opt->vocab_out_path), "%s", "out/ngram_vocab.txt");

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else if (sscanf(arg, "--text=%1023s", opt->text_path) == 1) {
            continue;
        } else if (sscanf(arg, "--steps=%d", &opt->steps) == 1) {
            continue;
        } else if (sscanf(arg, "--batch=%d", &opt->batch) == 1) {
            continue;
        } else if (sscanf(arg, "--context=%d", &opt->context) == 1) {
            continue;
        } else if (sscanf(arg, "--lr=%f", &opt->lr) == 1) {
            continue;
        } else if (sscanf(arg, "--embed=%d", &opt->embed) == 1) {
            continue;
        } else if (sscanf(arg, "--hidden=%d", &opt->hidden) == 1) {
            continue;
        } else if (sscanf(arg, "--vocab=%d", &opt->vocab_limit) == 1) {
            continue;
        } else if (sscanf(arg, "--snapshot=%1023s", opt->snapshot_path) == 1) {
            continue;
        } else if (sscanf(arg, "--vocab-out=%1023s", opt->vocab_out_path) == 1) {
            continue;
        } else {
            fprintf(stderr, "unknown option: %s\n", arg);
            exit(1);
        }
    }
}

static Tensor *make_one_hot(const int *idx, int n, int vocab) {
    Tensor *t = tensor_create(n, vocab, 0);
    tensor_fill(t, 0.0f);
    for (int i = 0; i < n; i++) {
        t->data[i * vocab + idx[i]] = 1.0f;
    }
    return t;
}

static void ngram_clear_forward_temps(NGramModel *model) {
    if (!model) {
        return;
    }
    if (model->tmp_x_list) {
        for (int i = 0; i < model->context; i++) {
            tensor_free(model->tmp_x_list[i]);
            model->tmp_x_list[i] = NULL;
        }
    }
    if (model->tmp_embed_list) {
        for (int i = 0; i < model->context; i++) {
            tensor_free(model->tmp_embed_list[i]);
            model->tmp_embed_list[i] = NULL;
        }
    }
    if (model->tmp_proj_list) {
        for (int i = 0; i < model->context; i++) {
            if (model->tmp_proj_list[i] != model->tmp_hidden_sum) {
                tensor_free(model->tmp_proj_list[i]);
            }
            model->tmp_proj_list[i] = NULL;
        }
    }
    if (model->tmp_sum_list) {
        for (int i = 0; i < model->tmp_n_sums; i++) {
            if (model->tmp_sum_list[i] != model->tmp_hidden_sum) {
                tensor_free(model->tmp_sum_list[i]);
            }
            model->tmp_sum_list[i] = NULL;
        }
    }
    tensor_free(model->tmp_hidden_sum);
    tensor_free(model->tmp_hidden_bias);
    tensor_free(model->tmp_hidden_act);
    tensor_free(model->tmp_logits);
    model->tmp_n_sums = 0;
    model->tmp_hidden_sum = NULL;
    model->tmp_hidden_bias = NULL;
    model->tmp_hidden_act = NULL;
    model->tmp_logits = NULL;
}

static int ngram_init(NGramModel *m, const NGramOptions *opt, int vocab, unsigned int *seed) {
    memset(m, 0, sizeof(*m));
    m->vocab = vocab;
    m->embed = opt->embed;
    m->hidden = opt->hidden;
    m->context = opt->context;
    m->E = tensor_create(vocab, opt->embed, 1);
    m->W_ctx = (Tensor **)calloc((size_t)opt->context, sizeof(Tensor *));
    m->b_h = tensor_create(1, opt->hidden, 1);
    m->W_out = tensor_create(opt->hidden, vocab, 1);
    m->b_out = tensor_create(1, vocab, 1);
    m->params = (Tensor **)calloc((size_t)(opt->context + 4), sizeof(Tensor *));
    m->velocity = (Tensor **)calloc((size_t)(opt->context + 4), sizeof(Tensor *));
    m->tmp_x_list = (Tensor **)calloc((size_t)opt->context, sizeof(Tensor *));
    m->tmp_embed_list = (Tensor **)calloc((size_t)opt->context, sizeof(Tensor *));
    m->tmp_proj_list = (Tensor **)calloc((size_t)opt->context, sizeof(Tensor *));
    m->tmp_sum_list = (Tensor **)calloc((size_t)opt->context, sizeof(Tensor *));
    if (!m->E || !m->W_ctx || !m->b_h || !m->W_out || !m->b_out || !m->params || !m->velocity ||
        !m->tmp_x_list || !m->tmp_embed_list || !m->tmp_proj_list || !m->tmp_sum_list) {
        return -1;
    }

    tensor_fill_randn(m->E, 0.0f, 0.05f, seed);
    for (int i = 0; i < opt->context; i++) {
        m->W_ctx[i] = tensor_create(opt->embed, opt->hidden, 1);
        if (!m->W_ctx[i]) {
            return -1;
        }
        tensor_fill_randn(m->W_ctx[i], 0.0f, 0.05f, seed);
    }
    tensor_fill(m->b_h, 0.0f);
    tensor_fill_randn(m->W_out, 0.0f, 0.05f, seed);
    tensor_fill(m->b_out, 0.0f);

    m->params[0] = m->E;
    for (int i = 0; i < opt->context; i++) {
        m->params[1 + i] = m->W_ctx[i];
    }
    m->params[1 + opt->context] = m->b_h;
    m->params[2 + opt->context] = m->W_out;
    m->params[3 + opt->context] = m->b_out;
    m->n_params = (size_t)(opt->context + 4);

    for (size_t i = 0; i < m->n_params; i++) {
        m->velocity[i] = tensor_create(m->params[i]->rows, m->params[i]->cols, 0);
        if (!m->velocity[i]) {
            return -1;
        }
    }
    return 0;
}

static void ngram_free(NGramModel *m) {
    if (!m) {
        return;
    }
    ngram_clear_forward_temps(m);
    tensor_free(m->E);
    if (m->W_ctx) {
        for (int i = 0; i < m->context; i++) {
            tensor_free(m->W_ctx[i]);
        }
    }
    tensor_free(m->b_h);
    tensor_free(m->W_out);
    tensor_free(m->b_out);
    if (m->velocity) {
        for (size_t i = 0; i < m->n_params; i++) {
            tensor_free(m->velocity[i]);
        }
    }
    free(m->W_ctx);
    free(m->params);
    free(m->velocity);
    free(m->tmp_x_list);
    free(m->tmp_embed_list);
    free(m->tmp_proj_list);
    free(m->tmp_sum_list);
    memset(m, 0, sizeof(*m));
}

static void pick_training_batch(const EncodedCorpus *corpus,
                                int batch,
                                int context,
                                int *context_ids,
                                int *target_ids,
                                unsigned int *seed) {
    for (int b = 0; b < batch; b++) {
        int pos = context + (int)(rand_uniform(seed) * (float)(corpus->n_tokens - context - 1));
        for (int c = 0; c < context; c++) {
            context_ids[b * context + c] = corpus->tokens[pos - context + c];
        }
        target_ids[b] = corpus->tokens[pos];
    }
}

static Tensor *ngram_forward(NGramModel *model, const int *context_ids, int batch) {
    Tensor *hidden_sum = NULL;
    Tensor *hidden_bias;
    Tensor *hidden_act;
    Tensor *logits_lin;
    Tensor *logits;

    ngram_clear_forward_temps(model);

    for (int c = 0; c < model->context; c++) {
        int *col_idx = (int *)malloc(sizeof(int) * (size_t)batch);
        if (!col_idx) {
            fprintf(stderr, "allocation failed\n");
            exit(1);
        }
        for (int b = 0; b < batch; b++) {
            col_idx[b] = context_ids[b * model->context + c];
        }
        model->tmp_x_list[c] = make_one_hot(col_idx, batch, model->vocab);
        model->tmp_embed_list[c] = tensor_matmul(model->tmp_x_list[c], model->E);
        model->tmp_proj_list[c] = tensor_matmul(model->tmp_embed_list[c], model->W_ctx[c]);
        free(col_idx);

        if (!hidden_sum) {
            hidden_sum = model->tmp_proj_list[c];
        } else {
            hidden_sum = tensor_add(hidden_sum, model->tmp_proj_list[c]);
            model->tmp_sum_list[model->tmp_n_sums++] = hidden_sum;
        }
    }

    hidden_bias = tensor_add_bias(hidden_sum, model->b_h);
    hidden_act = tensor_tanh(hidden_bias);
    logits_lin = tensor_matmul(hidden_act, model->W_out);
    logits = tensor_add_bias(logits_lin, model->b_out);

    model->tmp_hidden_sum = hidden_sum;
    model->tmp_hidden_bias = hidden_bias;
    model->tmp_hidden_act = hidden_act;
    model->tmp_logits = logits;
    tensor_free(logits_lin);
    return logits;
}

static int argmax_row(const Tensor *t, int row) {
    int best = 0;
    float best_value = t->data[row * t->cols];

    for (int i = 1; i < t->cols; i++) {
        float value = t->data[row * t->cols + i];
        if (value > best_value) {
            best_value = value;
            best = i;
        }
    }
    return best;
}

static void build_batch_from_tokens(const int *tokens,
                                    int batch,
                                    int context,
                                    int start,
                                    int *context_ids,
                                    int *target_ids) {
    for (int b = 0; b < batch; b++) {
        int pos = start + b;
        for (int c = 0; c < context; c++) {
            context_ids[b * context + c] = tokens[pos - context + c];
        }
        target_ids[b] = tokens[pos];
    }
}

static float ngram_eval(NGramModel *model, const int *tokens, int n_tokens, int batch, int context) {
    int *context_ids;
    int *target_ids;
    int batches = 0;
    float total_loss = 0.0f;

    if (n_tokens < context + batch) {
        return -1.0f;
    }
    context_ids = (int *)malloc(sizeof(int) * (size_t)batch * (size_t)context);
    target_ids = (int *)malloc(sizeof(int) * (size_t)batch);
    if (!context_ids || !target_ids) {
        fprintf(stderr, "allocation failed\n");
        exit(1);
    }

    for (int start = context; start + batch <= n_tokens; start += batch) {
        Tensor *logits;
        Tensor *targets;
        Tensor *probs;
        Tensor *loss;

        build_batch_from_tokens(tokens, batch, context, start, context_ids, target_ids);
        logits = ngram_forward(model, context_ids, batch);
        targets = make_one_hot(target_ids, batch, model->vocab);
        probs = tensor_softmax(logits);
        loss = tensor_cross_entropy(probs, targets);
        total_loss += loss->data[0];
        batches++;

        tensor_free(targets);
        tensor_free(probs);
        tensor_free(loss);
        ngram_clear_forward_temps(model);
    }

    free(context_ids);
    free(target_ids);
    return batches > 0 ? total_loss / (float)batches : -1.0f;
}

static void ngram_predict_topk(NGramModel *model, const EncodedCorpus *corpus, const char *prompt, int top_k) {
    int ids[16];
    int n_ids = vocab_encode_prompt(corpus, prompt, ids, 16);
    int context_ids[16];
    Tensor *logits;
    Tensor *probs;
    float best_prob[8];
    int best_id[8];

    if (top_k > 8) {
        top_k = 8;
    }
    if (n_ids < model->context) {
        return;
    }
    for (int i = 0; i < model->context; i++) {
        context_ids[i] = ids[n_ids - model->context + i];
    }

    logits = ngram_forward(model, context_ids, 1);
    probs = tensor_softmax(logits);

    for (int i = 0; i < top_k; i++) {
        best_prob[i] = -1.0f;
        best_id[i] = -1;
    }
    for (int v = 0; v < corpus->vocab; v++) {
        float p = probs->data[v];
        int slot = -1;
        for (int i = 0; i < top_k; i++) {
            if (p > best_prob[i]) {
                slot = i;
                break;
            }
        }
        if (slot >= 0) {
            for (int i = top_k - 1; i > slot; i--) {
                best_prob[i] = best_prob[i - 1];
                best_id[i] = best_id[i - 1];
            }
            best_prob[slot] = p;
            best_id[slot] = v;
        }
    }

    printf("%s ->", prompt);
    for (int i = 0; i < top_k; i++) {
        if (best_id[i] >= 0) {
            printf(" %s(%.2f)", corpus->entries[best_id[i]].word, best_prob[i]);
        }
    }
    printf("\n");

    tensor_free(probs);
    ngram_clear_forward_temps(model);
}

static void ngram_generate_sample(NGramModel *model, const EncodedCorpus *corpus, const char *prompt, int length) {
    int ids[128];
    int n_ids = vocab_encode_prompt(corpus, prompt, ids, 128);

    printf("sample: %s", prompt);
    if (n_ids < model->context) {
        printf(" ... (need %d in-vocab prompt words)\n", model->context);
        return;
    }

    for (int step = 0; step < length && n_ids < (int)(sizeof(ids) / sizeof(ids[0])); step++) {
        int context_ids[16];
        Tensor *logits;
        Tensor *probs;
        int next_id;

        for (int i = 0; i < model->context; i++) {
            context_ids[i] = ids[n_ids - model->context + i];
        }
        logits = ngram_forward(model, context_ids, 1);
        probs = tensor_softmax(logits);
        next_id = argmax_row(probs, 0);
        ids[n_ids++] = next_id;
        printf(" %s", corpus->entries[next_id].word);

        tensor_free(probs);
        ngram_clear_forward_temps(model);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    NGramOptions opt;
    size_t text_len = 0;
    char *text;
    EncodedCorpus corpus;
    int split_tokens;
    int train_tokens;
    int eval_tokens;
    const int *train_data;
    const int *eval_data;
    int *context_ids;
    int *target_ids;
    unsigned int seed = 1337u;
    NGramModel model;

    parse_args(argc, argv, &opt);
    text = vocab_read_file(opt.text_path, &text_len);
    if (!text) {
        fprintf(stderr, "failed to read '%s'\n", opt.text_path);
        return 1;
    }
    if (opt.steps <= 0) opt.steps = 2000;
    if (opt.batch <= 0) opt.batch = 64;
    if (opt.context <= 0) opt.context = 3;
    if (opt.embed <= 0) opt.embed = 32;
    if (opt.hidden <= 0) opt.hidden = 64;
    if (opt.vocab_limit <= 0) opt.vocab_limit = DEFAULT_VOCAB;

    if (vocab_build_corpus(text, text_len, opt.vocab_limit, &corpus) != 0 || corpus.n_tokens <= opt.context + 1) {
        fprintf(stderr, "failed to build corpus/vocab\n");
        free(text);
        return 1;
    }
    split_tokens = (int)((float)corpus.n_tokens * 0.9f);
    if (split_tokens < opt.context + opt.batch + 1) {
        split_tokens = corpus.n_tokens;
    }
    train_tokens = split_tokens;
    eval_tokens = corpus.n_tokens - split_tokens;
    train_data = corpus.tokens;
    eval_data = corpus.tokens + split_tokens;
    context_ids = (int *)malloc(sizeof(int) * (size_t)opt.batch * (size_t)opt.context);
    target_ids = (int *)malloc(sizeof(int) * (size_t)opt.batch);
    if (!context_ids || !target_ids) {
        fprintf(stderr, "allocation failed\n");
        free(context_ids);
        free(target_ids);
        vocab_free_corpus(&corpus);
        free(text);
        return 1;
    }
    if (ngram_init(&model, &opt, corpus.vocab, &seed) != 0) {
        fprintf(stderr, "failed to initialize ngram model\n");
        free(context_ids);
        free(target_ids);
        vocab_free_corpus(&corpus);
        free(text);
        return 1;
    }

    printf("loaded %zu bytes, total_words=%d kept_tokens=%d vocab=%d\n",
           text_len, corpus.total_words, corpus.n_tokens, corpus.vocab);
    printf("training neural ngram: steps=%d batch=%d context=%d embed=%d hidden=%d lr=%.4f\n",
           opt.steps, opt.batch, opt.context, opt.embed, opt.hidden, opt.lr);
    printf("split: train=%d eval=%d tokens\n", train_tokens, eval_tokens);

    for (int step = 1; step <= opt.steps; step++) {
        Tensor *logits;
        Tensor *targets;
        Tensor *probs;
        Tensor *loss;
        float train_loss;

        if (train_tokens < corpus.n_tokens) {
            for (int b = 0; b < opt.batch; b++) {
                int pos = opt.context +
                          (int)(rand_uniform(&seed) * (float)(train_tokens - opt.context - 1));
                for (int c = 0; c < opt.context; c++) {
                    context_ids[b * opt.context + c] = train_data[pos - opt.context + c];
                }
                target_ids[b] = train_data[pos];
            }
        } else {
            pick_training_batch(&corpus, opt.batch, opt.context, context_ids, target_ids, &seed);
        }
        logits = ngram_forward(&model, context_ids, opt.batch);
        targets = make_one_hot(target_ids, opt.batch, corpus.vocab);
        probs = tensor_softmax(logits);
        loss = tensor_cross_entropy(probs, targets);

        tensor_backward(loss);
        tensor_sgd_momentum_step(model.params, model.velocity, model.n_params, opt.lr, 0.9f);
        train_loss = loss->data[0];

        tensor_free(targets);
        tensor_free(probs);
        tensor_free(loss);
        ngram_clear_forward_temps(&model);

        if (step == 1 || step % 100 == 0 || step == opt.steps) {
            float eval_loss = ngram_eval(&model, eval_data, eval_tokens, opt.batch, opt.context);
            if (eval_loss > 0.0f) {
                printf("step %4d train_loss %.6f eval_loss %.6f ppl %.2f\n",
                       step,
                       train_loss,
                       eval_loss,
                       expf(eval_loss));
            } else {
                printf("step %4d train_loss %.6f\n", step, train_loss);
            }
        }
    }

    printf("\n--- predictions ---\n");
    ngram_predict_topk(&model, &corpus, "to be or", 5);
    ngram_predict_topk(&model, &corpus, "my lord i", 5);
    ngram_predict_topk(&model, &corpus, "i will not", 5);
    ngram_predict_topk(&model, &corpus, "for the king", 5);
    printf("\n--- generation ---\n");
    ngram_generate_sample(&model, &corpus, "to be or", 12);
    ngram_generate_sample(&model, &corpus, "my lord i", 12);

    if (tensor_snapshot_save(model.params, model.n_params, opt.snapshot_path) != 0) {
        fprintf(stderr, "failed to save snapshot\n");
        ngram_free(&model);
        free(context_ids);
        free(target_ids);
        vocab_free_corpus(&corpus);
        free(text);
        return 1;
    }
    if (vocab_save(&corpus, opt.vocab_out_path) != 0) {
        fprintf(stderr, "failed to save vocab\n");
        ngram_free(&model);
        free(context_ids);
        free(target_ids);
        vocab_free_corpus(&corpus);
        free(text);
        return 1;
    }
    printf("\nsaved snapshot: %s\n", opt.snapshot_path);
    printf("saved vocab: %s\n", opt.vocab_out_path);

    ngram_free(&model);
    free(context_ids);
    free(target_ids);
    vocab_free_corpus(&corpus);
    free(text);
    return 0;
}
