#include "tensor.h"

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
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WORD_HASH_CAPACITY 65536
#define MAX_WORD_LEN 63
#define DEFAULT_VOCAB 800

typedef struct {
    char *word;
    int count;
} VocabEntry;

typedef struct {
    char *word;
    int count;
    int id;
} HashSlot;

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
} SkipGramOptions;

static float rand_uniform(unsigned int *seed) {
    *seed = (1103515245U * (*seed) + 12345U);
    return (float)((*seed) & 0x7fffffffU) / 2147483648.0f;
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
}

static void parse_args(int argc, char **argv, SkipGramOptions *opt) {
    snprintf(opt->text_path, sizeof(opt->text_path), "%s", "data/shakespeare/shakespeare_gutenberg.txt");
    opt->steps = 2000;
    opt->batch = 64;
    opt->window = 2;
    opt->lr = 0.05f;
    opt->embed = 32;
    opt->vocab_limit = DEFAULT_VOCAB;
    snprintf(opt->snapshot_path, sizeof(opt->snapshot_path), "%s", "skipgram_snapshot.bin");
    snprintf(opt->vocab_out_path, sizeof(opt->vocab_out_path), "%s", "skipgram_vocab.txt");

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
        } else {
            fprintf(stderr, "unknown option: %s\n", arg);
            exit(1);
        }
    }
}

static char *read_file(const char *path, size_t *out_size) {
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

static uint32_t hash_word(const char *word) {
    uint32_t h = 2166136261u;
    while (*word) {
        h ^= (unsigned char)(*word++);
        h *= 16777619u;
    }
    return h;
}

static char *dup_word(const char *src) {
    size_t n = strlen(src);
    char *dst = (char *)malloc(n + 1);
    if (!dst) {
        return NULL;
    }
    memcpy(dst, src, n + 1);
    return dst;
}

static int next_word(const char *text, size_t n, size_t *pos, char word[MAX_WORD_LEN + 1]) {
    size_t i = *pos;
    int len = 0;

    while (i < n) {
        unsigned char ch = (unsigned char)text[i];
        if (isalpha(ch) || ch == '\'') {
            break;
        }
        i++;
    }
    if (i >= n) {
        *pos = i;
        return 0;
    }

    while (i < n) {
        unsigned char ch = (unsigned char)text[i];
        if (!(isalpha(ch) || ch == '\'')) {
            break;
        }
        if (len < MAX_WORD_LEN) {
            word[len++] = (char)tolower(ch);
        }
        i++;
    }
    word[len] = '\0';
    *pos = i;
    return len > 0;
}

static void count_words(const char *text, size_t n, HashSlot *table, int *unique_words, int *total_words) {
    size_t pos = 0;
    char word[MAX_WORD_LEN + 1];

    *unique_words = 0;
    *total_words = 0;
    memset(table, 0, sizeof(HashSlot) * WORD_HASH_CAPACITY);

    while (next_word(text, n, &pos, word)) {
        uint32_t idx = hash_word(word) % WORD_HASH_CAPACITY;
        (*total_words)++;
        while (table[idx].word) {
            if (strcmp(table[idx].word, word) == 0) {
                table[idx].count++;
                break;
            }
            idx = (idx + 1u) % WORD_HASH_CAPACITY;
        }
        if (!table[idx].word) {
            table[idx].word = dup_word(word);
            if (!table[idx].word) {
                fprintf(stderr, "allocation failed building vocab\n");
                exit(1);
            }
            table[idx].count = 1;
            table[idx].id = -1;
            (*unique_words)++;
        }
    }
}

static int compare_vocab_desc(const void *a, const void *b) {
    const VocabEntry *va = (const VocabEntry *)a;
    const VocabEntry *vb = (const VocabEntry *)b;
    if (vb->count != va->count) {
        return vb->count - va->count;
    }
    return strcmp(va->word, vb->word);
}

static VocabEntry *build_vocab_from_counts(HashSlot *table, int unique_words, int vocab_limit, int *out_vocab) {
    VocabEntry *entries;
    int k = 0;
    int vocab;

    entries = (VocabEntry *)malloc(sizeof(VocabEntry) * (size_t)unique_words);
    if (!entries) {
        return NULL;
    }
    for (int i = 0; i < WORD_HASH_CAPACITY; i++) {
        if (table[i].word) {
            entries[k].word = table[i].word;
            entries[k].count = table[i].count;
            k++;
        }
    }
    qsort(entries, (size_t)k, sizeof(VocabEntry), compare_vocab_desc);
    vocab = k < vocab_limit ? k : vocab_limit;
    *out_vocab = vocab;
    return entries;
}

static void assign_vocab_ids(HashSlot *table, VocabEntry *vocab_entries, int vocab) {
    for (int i = 0; i < vocab; i++) {
        uint32_t idx = hash_word(vocab_entries[i].word) % WORD_HASH_CAPACITY;
        while (table[idx].word) {
            if (strcmp(table[idx].word, vocab_entries[i].word) == 0) {
                table[idx].id = i;
                break;
            }
            idx = (idx + 1u) % WORD_HASH_CAPACITY;
        }
    }
}

static int *encode_corpus(const char *text, size_t n, HashSlot *table, int *out_tokens) {
    size_t pos = 0;
    char word[MAX_WORD_LEN + 1];
    int cap = 1024;
    int len = 0;
    int *tokens = (int *)malloc(sizeof(int) * (size_t)cap);

    if (!tokens) {
        return NULL;
    }

    while (next_word(text, n, &pos, word)) {
        uint32_t idx = hash_word(word) % WORD_HASH_CAPACITY;
        int id = -1;
        while (table[idx].word) {
            if (strcmp(table[idx].word, word) == 0) {
                id = table[idx].id;
                break;
            }
            idx = (idx + 1u) % WORD_HASH_CAPACITY;
        }
        if (id < 0) {
            continue;
        }
        if (len == cap) {
            int next_cap = cap * 2;
            int *next = (int *)realloc(tokens, sizeof(int) * (size_t)next_cap);
            if (!next) {
                free(tokens);
                return NULL;
            }
            tokens = next;
            cap = next_cap;
        }
        tokens[len++] = id;
    }

    *out_tokens = len;
    return tokens;
}

static void free_hash_table(HashSlot *table) {
    for (int i = 0; i < WORD_HASH_CAPACITY; i++) {
        free(table[i].word);
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

static int vocab_id_for_word(VocabEntry *vocab_entries, int vocab, const char *word) {
    for (int i = 0; i < vocab; i++) {
        if (strcmp(vocab_entries[i].word, word) == 0) {
            return i;
        }
    }
    return -1;
}

static void print_nearest_neighbors(SkipGramModel *model, VocabEntry *vocab_entries, int vocab, const char *word, int k) {
    int id = vocab_id_for_word(vocab_entries, vocab, word);
    float best_score[8];
    int best_id[8];
    const float *query;
    float query_norm;

    if (k > 8) {
        k = 8;
    }
    if (id < 0) {
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

    printf("%s:", word);
    for (int i = 0; i < k; i++) {
        if (best_id[i] >= 0) {
            printf(" %s(%.2f)", vocab_entries[best_id[i]].word, best_score[i]);
        }
    }
    printf("\n");
}

static int save_vocab(VocabEntry *vocab_entries, int vocab, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) {
        return -1;
    }
    for (int i = 0; i < vocab; i++) {
        if (fprintf(f, "%d\t%s\t%d\n", i, vocab_entries[i].word, vocab_entries[i].count) < 0) {
            fclose(f);
            return -1;
        }
    }
    if (fclose(f) != 0) {
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    SkipGramOptions opt;
    size_t text_len = 0;
    char *text;
    HashSlot *table;
    VocabEntry *vocab_entries;
    int unique_words = 0;
    int total_words = 0;
    int vocab = 0;
    int n_tokens = 0;
    int *tokens;
    int *center_idx;
    int *context_idx;
    unsigned int seed = 1337u;
    SkipGramModel model;
    Tensor *params[2];

    parse_args(argc, argv, &opt);
    text = read_file(opt.text_path, &text_len);

    if (!text) {
        fprintf(stderr, "failed to read '%s'\n", opt.text_path);
        return 1;
    }
    if (opt.steps <= 0) opt.steps = 2000;
    if (opt.batch <= 0) opt.batch = 64;
    if (opt.window <= 0) opt.window = 2;
    if (opt.embed <= 0) opt.embed = 32;
    if (opt.vocab_limit <= 0) opt.vocab_limit = DEFAULT_VOCAB;

    table = (HashSlot *)calloc(WORD_HASH_CAPACITY, sizeof(HashSlot));
    if (!table) {
        fprintf(stderr, "allocation failed\n");
        free(text);
        return 1;
    }

    count_words(text, text_len, table, &unique_words, &total_words);
    vocab_entries = build_vocab_from_counts(table, unique_words, opt.vocab_limit, &vocab);
    if (!vocab_entries || vocab < 8) {
        fprintf(stderr, "failed to build vocab\n");
        free_hash_table(table);
        free(table);
        free(text);
        free(vocab_entries);
        return 1;
    }
    assign_vocab_ids(table, vocab_entries, vocab);
    tokens = encode_corpus(text, text_len, table, &n_tokens);
    if (!tokens || n_tokens < 16) {
        fprintf(stderr, "failed to encode corpus\n");
        free(tokens);
        free(vocab_entries);
        free_hash_table(table);
        free(table);
        free(text);
        return 1;
    }

    center_idx = (int *)malloc(sizeof(int) * (size_t)opt.batch);
    context_idx = (int *)malloc(sizeof(int) * (size_t)opt.batch);
    if (!center_idx || !context_idx) {
        fprintf(stderr, "allocation failed\n");
        free(center_idx);
        free(context_idx);
        free(tokens);
        free(vocab_entries);
        free_hash_table(table);
        free(table);
        free(text);
        return 1;
    }

    if (skipgram_init(&model, vocab, opt.embed, &seed) != 0) {
        fprintf(stderr, "failed to initialize model\n");
        free(center_idx);
        free(context_idx);
        free(tokens);
        free(vocab_entries);
        free_hash_table(table);
        free(table);
        free(text);
        return 1;
    }

    params[0] = model.W_in;
    params[1] = model.W_out;

    printf("loaded %zu bytes, total_words=%d kept_tokens=%d vocab=%d embed=%d\n",
           text_len, total_words, n_tokens, vocab, opt.embed);
    printf("training skip-gram softmax: steps=%d batch=%d window=%d lr=%.4f momentum=0.9\n",
           opt.steps, opt.batch, opt.window, opt.lr);

    for (int step = 1; step <= opt.steps; step++) {
        Tensor *X;
        Tensor *Y;
        Tensor *hidden;
        Tensor *logits;
        Tensor *probs;
        Tensor *loss;

        pick_training_batch(tokens, n_tokens, opt.batch, opt.window, center_idx, context_idx, &seed);
        X = make_one_hot(center_idx, opt.batch, vocab);
        Y = make_one_hot(context_idx, opt.batch, vocab);
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
    print_nearest_neighbors(&model, vocab_entries, vocab, "king", 5);
    print_nearest_neighbors(&model, vocab_entries, vocab, "queen", 5);
    print_nearest_neighbors(&model, vocab_entries, vocab, "love", 5);
    print_nearest_neighbors(&model, vocab_entries, vocab, "death", 5);
    print_nearest_neighbors(&model, vocab_entries, vocab, "man", 5);
    print_nearest_neighbors(&model, vocab_entries, vocab, "woman", 5);

    if (tensor_snapshot_save(params, 2, opt.snapshot_path) != 0) {
        fprintf(stderr, "failed to save snapshot to '%s'\n", opt.snapshot_path);
        skipgram_free(&model);
        free(center_idx);
        free(context_idx);
        free(tokens);
        free(vocab_entries);
        free_hash_table(table);
        free(table);
        free(text);
        return 1;
    }
    if (save_vocab(vocab_entries, vocab, opt.vocab_out_path) != 0) {
        fprintf(stderr, "failed to save vocab to '%s'\n", opt.vocab_out_path);
        skipgram_free(&model);
        free(center_idx);
        free(context_idx);
        free(tokens);
        free(vocab_entries);
        free_hash_table(table);
        free(table);
        free(text);
        return 1;
    }
    printf("\nsaved snapshot: %s\n", opt.snapshot_path);
    printf("saved vocab: %s\n", opt.vocab_out_path);

    skipgram_free(&model);
    free(center_idx);
    free(context_idx);
    free(tokens);
    free(vocab_entries);
    free_hash_table(table);
    free(table);
    free(text);
    return 0;
}
