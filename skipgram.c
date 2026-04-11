#include "tensor.h"
#include "vocab.h"

/*
 * skipgram.c
 *
 * Minimal word2vec-style skip-gram word embedding demo using nanotensor.
 * - Tokenizes a text corpus into lowercase words
 * - Trains a center-word -> context-word predictor with a softmax loss
 * - Prints nearest-neighbor words from the learned embedding table
 * - Saves the trained embedding weights as a tensor snapshot
 * - Exports the retained vocabulary as a plain text file
 *
 * Usage:
 *   ./skipgram [--text=PATH] [--steps=N] [--batch=N] [--window=N]
 *              [--lr=FLOAT] [--embed=N] [--vocab=N]
 *              [--snapshot=PATH] [--vocab-out=PATH]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_VOCAB 800

typedef struct {
    int vocab;
    int embed;
    Tensor *W_in;
    Tensor *W_out;
    Tensor *velocity[2];
} SkipGramModel;

typedef struct {
    char text_path[1024];
    int steps;
    int batch;
    int window;
    float lr;
    int embed;
    int vocab_limit;
    char snapshot_path[1024];
    char vocab_out_path[1024];
    char report_path[1024];
} SkipGramOptions;

static float rand_uniform(unsigned int *seed) {
    *seed = (1103515245U * (*seed) + 12345U);
    return (float)((*seed) & 0x7fffffffU) / 2147483648.0f;
}

static int word_eq_nocase(const char *a, const char *b) {
    while (*a && *b) {
        char ca = *a;
        char cb = *b;
        if (ca >= 'A' && ca <= 'Z') ca = (char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (char)(cb - 'A' + 'a');
        if (ca != cb) {
            return 0;
        }
        a++;
        b++;
    }
    return *a == '\0' && *b == '\0';
}

static void print_usage(const char *prog) {
    printf("usage: %s [options]\n", prog);
    printf("  --text=PATH\n");
    printf("  --steps=N\n");
    printf("  --batch=N\n");
    printf("  --window=N\n");
    printf("  --lr=FLOAT\n");
    printf("  --embed=N\n");
    printf("  --vocab=N\n");
    printf("  --snapshot=PATH\n");
    printf("  --vocab-out=PATH\n");
    printf("  --report=PATH\n");
}

static void parse_args(int argc, char **argv, SkipGramOptions *opt) {
    snprintf(opt->text_path, sizeof(opt->text_path), "%s", "data/shakespeare/shakespeare_gutenberg.txt");
    opt->steps = 2000;
    opt->batch = 64;
    opt->window = 2;
    opt->lr = 0.05f;
    opt->embed = 32;
    opt->vocab_limit = DEFAULT_VOCAB;
    snprintf(opt->snapshot_path, sizeof(opt->snapshot_path), "%s", "out/skipgram_snapshot.bin");
    snprintf(opt->vocab_out_path, sizeof(opt->vocab_out_path), "%s", "out/skipgram_vocab.txt");
    snprintf(opt->report_path, sizeof(opt->report_path), "%s", "out/skipgram_neighbors.txt");

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
        } else if (sscanf(arg, "--window=%d", &opt->window) == 1) {
            continue;
        } else if (sscanf(arg, "--lr=%f", &opt->lr) == 1) {
            continue;
        } else if (sscanf(arg, "--embed=%d", &opt->embed) == 1) {
            continue;
        } else if (sscanf(arg, "--vocab=%d", &opt->vocab_limit) == 1) {
            continue;
        } else if (sscanf(arg, "--snapshot=%1023s", opt->snapshot_path) == 1) {
            continue;
        } else if (sscanf(arg, "--vocab-out=%1023s", opt->vocab_out_path) == 1) {
            continue;
        } else if (sscanf(arg, "--report=%1023s", opt->report_path) == 1) {
            continue;
        } else {
            fprintf(stderr, "unknown option: %s\n", arg);
            exit(1);
        }
    }
}

static int skipgram_init(SkipGramModel *m, int vocab, int embed, unsigned int *seed) {
    m->vocab = vocab;
    m->embed = embed;
    m->W_in = tensor_create(vocab, embed, 1);
    m->W_out = tensor_create(embed, vocab, 1);
    m->velocity[0] = tensor_create(vocab, embed, 0);
    m->velocity[1] = tensor_create(embed, vocab, 0);
    if (!m->W_in || !m->W_out || !m->velocity[0] || !m->velocity[1]) {
        return -1;
    }
    tensor_fill_randn(m->W_in, 0.0f, 0.05f, seed);
    tensor_fill_randn(m->W_out, 0.0f, 0.05f, seed);
    return 0;
}

static void skipgram_free(SkipGramModel *m) {
    tensor_free(m->W_in);
    tensor_free(m->W_out);
    tensor_free(m->velocity[0]);
    tensor_free(m->velocity[1]);
}

static void pick_training_batch(const int *tokens,
                                int n_tokens,
                                int batch,
                                int window,
                                int *center_idx,
                                int *context_idx,
                                unsigned int *seed) {
    for (int b = 0; b < batch; b++) {
        int pos;
        int offset;
        do {
            pos = (int)(rand_uniform(seed) * (float)n_tokens);
        } while (pos < window || pos >= n_tokens - window);

        do {
            offset = (int)(rand_uniform(seed) * (float)(2 * window + 1)) - window;
        } while (offset == 0);

        center_idx[b] = tokens[pos];
        context_idx[b] = tokens[pos + offset];
    }
}

static float dot_row(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static float norm_row(const float *a, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * a[i];
    }
    return sqrtf(sum > 1e-12f ? sum : 1e-12f);
}

static void write_nearest_neighbors(FILE *out,
                                    SkipGramModel *model,
                                    const EncodedCorpus *corpus,
                                    const char *word,
                                    int k) {
    int id = vocab_id_for_word(corpus, word);
    float best_score[8];
    int best_id[8];
    const float *query;
    float query_norm;
    int vocab = corpus->vocab;

    if (k > 8) {
        k = 8;
    }
    if (id < 0) {
        fprintf(out, "%s: <not in vocab>\n", word);
        return;
    }

    for (int i = 0; i < k; i++) {
        best_score[i] = -1e30f;
        best_id[i] = -1;
    }

    query = model->W_in->data + (size_t)id * model->embed;
    query_norm = norm_row(query, model->embed);

    for (int i = 0; i < vocab; i++) {
        const float *cand;
        float score;
        int slot = -1;

        if (i == id) {
            continue;
        }
        cand = model->W_in->data + (size_t)i * model->embed;
        score = dot_row(query, cand, model->embed) / (query_norm * norm_row(cand, model->embed));
        for (int j = 0; j < k; j++) {
            if (score > best_score[j]) {
                slot = j;
                break;
            }
        }
        if (slot >= 0) {
            for (int j = k - 1; j > slot; j--) {
                best_score[j] = best_score[j - 1];
                best_id[j] = best_id[j - 1];
            }
            best_score[slot] = score;
            best_id[slot] = i;
        }
    }

    fprintf(out, "%s:", word);
    for (int i = 0; i < k; i++) {
        if (best_id[i] >= 0) {
            fprintf(out, " %s(%.2f)", corpus->entries[best_id[i]].word, best_score[i]);
        }
    }
    fprintf(out, "\n");
}

static void print_nearest_neighbors(SkipGramModel *model, const EncodedCorpus *corpus, const char *word, int k) {
    write_nearest_neighbors(stdout, model, corpus, word, k);
}

static void write_skipgram_report(const char *path, SkipGramModel *model, const EncodedCorpus *corpus) {
    static const char *queries[] = {"king", "queen", "love", "death", "man", "woman"};
    FILE *out = fopen(path, "w");

    if (!out) {
        fprintf(stderr, "failed to open report '%s'\n", path);
        return;
    }

    fprintf(out, "skipgram nearest neighbors\n");
    fprintf(out, "vocab=%d embed=%d\n\n", corpus->vocab, model->embed);
    for (size_t i = 0; i < sizeof(queries) / sizeof(queries[0]); i++) {
        write_nearest_neighbors(out, model, corpus, queries[i], 5);
    }
    fprintf(out, "\nkept vocab examples:\n");
    {
        int shown = 0;
        for (int i = 0; i < corpus->vocab && shown < 12; i++) {
            const char *word = corpus->entries[i].word;
            if (word_eq_nocase(word, "king") || word_eq_nocase(word, "queen") || word_eq_nocase(word, "love") ||
                word_eq_nocase(word, "death") || word_eq_nocase(word, "man") || word_eq_nocase(word, "woman")) {
                continue;
            }
            fprintf(out, "%s%s", shown == 0 ? "" : " ", word);
            shown++;
        }
        fprintf(out, "\n");
    }
    fclose(out);
}

int main(int argc, char **argv) {
    SkipGramOptions opt;
    size_t text_len = 0;
    char *text;
    EncodedCorpus corpus;
    int *center_idx;
    int *context_idx;
    unsigned int seed = 1337u;
    SkipGramModel model;
    Tensor *params[2];

    parse_args(argc, argv, &opt);
    text = vocab_read_file(opt.text_path, &text_len);

    if (!text) {
        fprintf(stderr, "failed to read '%s'\n", opt.text_path);
        return 1;
    }
    if (opt.steps <= 0) opt.steps = 2000;
    if (opt.batch <= 0) opt.batch = 64;
    if (opt.window <= 0) opt.window = 2;
    if (opt.embed <= 0) opt.embed = 32;
    if (opt.vocab_limit <= 0) opt.vocab_limit = DEFAULT_VOCAB;

    if (vocab_build_corpus(text, text_len, opt.vocab_limit, &corpus) != 0 || corpus.vocab < 8 ||
        corpus.n_tokens < 16) {
        fprintf(stderr, "failed to build corpus/vocab\n");
        free(text);
        return 1;
    }

    center_idx = (int *)malloc(sizeof(int) * (size_t)opt.batch);
    context_idx = (int *)malloc(sizeof(int) * (size_t)opt.batch);
    if (!center_idx || !context_idx) {
        fprintf(stderr, "allocation failed\n");
        free(center_idx);
        free(context_idx);
        vocab_free_corpus(&corpus);
        free(text);
        return 1;
    }

    if (skipgram_init(&model, corpus.vocab, opt.embed, &seed) != 0) {
        fprintf(stderr, "failed to initialize model\n");
        free(center_idx);
        free(context_idx);
        vocab_free_corpus(&corpus);
        free(text);
        return 1;
    }

    params[0] = model.W_in;
    params[1] = model.W_out;

    printf("loaded %zu bytes, total_words=%d kept_tokens=%d vocab=%d embed=%d\n",
           text_len, corpus.total_words, corpus.n_tokens, corpus.vocab, opt.embed);
    printf("training skip-gram softmax: steps=%d batch=%d window=%d lr=%.4f momentum=0.9\n",
           opt.steps, opt.batch, opt.window, opt.lr);

    for (int step = 1; step <= opt.steps; step++) {
        Tensor *X;
        Tensor *Y;
        Tensor *hidden;
        Tensor *logits;
        Tensor *probs;
        Tensor *loss;

        pick_training_batch(corpus.tokens, corpus.n_tokens, opt.batch, opt.window, center_idx, context_idx, &seed);
        X = tensor_one_hot(center_idx, opt.batch, corpus.vocab);
        Y = tensor_one_hot(context_idx, opt.batch, corpus.vocab);
        hidden = tensor_matmul(X, model.W_in);
        logits = tensor_matmul(hidden, model.W_out);
        probs = tensor_softmax(logits);
        loss = tensor_cross_entropy(probs, Y);

        tensor_backward(loss);
        tensor_sgd_momentum_step(params, model.velocity, 2, opt.lr, 0.9f);

        if (step == 1 || step % 100 == 0) {
            printf("step %4d loss %.6f\n", step, loss->data[0]);
        }

        tensor_free(X);
        tensor_free(Y);
        tensor_free(hidden);
        tensor_free(logits);
        tensor_free(probs);
        tensor_free(loss);
    }

    printf("\n--- nearest neighbors ---\n");
    print_nearest_neighbors(&model, &corpus, "king", 5);
    print_nearest_neighbors(&model, &corpus, "queen", 5);
    print_nearest_neighbors(&model, &corpus, "love", 5);
    print_nearest_neighbors(&model, &corpus, "death", 5);
    print_nearest_neighbors(&model, &corpus, "man", 5);
    print_nearest_neighbors(&model, &corpus, "woman", 5);

    if (tensor_snapshot_save(params, 2, opt.snapshot_path) != 0) {
        fprintf(stderr, "failed to save snapshot to '%s'\n", opt.snapshot_path);
        skipgram_free(&model);
        free(center_idx);
        free(context_idx);
        vocab_free_corpus(&corpus);
        free(text);
        return 1;
    }
    if (vocab_save(&corpus, opt.vocab_out_path) != 0) {
        fprintf(stderr, "failed to save vocab to '%s'\n", opt.vocab_out_path);
        skipgram_free(&model);
        free(center_idx);
        free(context_idx);
        vocab_free_corpus(&corpus);
        free(text);
        return 1;
    }
    write_skipgram_report(opt.report_path, &model, &corpus);
    printf("\nsaved snapshot: %s\n", opt.snapshot_path);
    printf("saved vocab: %s\n", opt.vocab_out_path);
    printf("saved report: %s\n", opt.report_path);

    skipgram_free(&model);
    free(center_idx);
    free(context_idx);
    vocab_free_corpus(&corpus);
    free(text);
    return 0;
}
