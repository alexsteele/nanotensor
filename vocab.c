#include "vocab.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WORD_HASH_CAPACITY 65536
#define MAX_WORD_LEN 63

typedef struct {
    char *word;
    int count;
    int id;
} HashSlot;

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

static void free_hash_table(HashSlot *table) {
    for (int i = 0; i < WORD_HASH_CAPACITY; i++) {
        free(table[i].word);
    }
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
                return;
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
    for (int i = 0; i < vocab; i++) {
        char *copy = dup_word(entries[i].word);
        if (!copy) {
            for (int j = 0; j < i; j++) {
                free(entries[j].word);
            }
            free(entries);
            return NULL;
        }
        entries[i].word = copy;
    }
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

char *vocab_read_file(const char *path, size_t *out_size) {
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

int vocab_build_corpus(const char *text, size_t n, int vocab_limit, EncodedCorpus *out) {
    HashSlot *table;
    int unique_words = 0;

    if (!text || !out || vocab_limit <= 0) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    table = (HashSlot *)calloc(WORD_HASH_CAPACITY, sizeof(HashSlot));
    if (!table) {
        return -1;
    }

    count_words(text, n, table, &unique_words, &out->total_words);
    out->entries = build_vocab_from_counts(table, unique_words, vocab_limit, &out->vocab);
    if (!out->entries || out->vocab < 1) {
        free(out->entries);
        free_hash_table(table);
        free(table);
        memset(out, 0, sizeof(*out));
        return -1;
    }
    assign_vocab_ids(table, out->entries, out->vocab);
    out->tokens = encode_corpus(text, n, table, &out->n_tokens);
    free_hash_table(table);
    free(table);
    if (!out->tokens || out->n_tokens < 1) {
        vocab_free_corpus(out);
        return -1;
    }
    return 0;
}

void vocab_free_corpus(EncodedCorpus *corpus) {
    if (!corpus) {
        return;
    }
    for (int i = 0; i < corpus->vocab; i++) {
        free(corpus->entries[i].word);
    }
    free(corpus->entries);
    free(corpus->tokens);
    memset(corpus, 0, sizeof(*corpus));
}

int vocab_save(const EncodedCorpus *corpus, const char *path) {
    FILE *f;

    if (!corpus || !path) {
        return -1;
    }
    f = fopen(path, "w");
    if (!f) {
        return -1;
    }
    for (int i = 0; i < corpus->vocab; i++) {
        if (fprintf(f, "%d\t%s\t%d\n", i, corpus->entries[i].word, corpus->entries[i].count) < 0) {
            fclose(f);
            return -1;
        }
    }
    if (fclose(f) != 0) {
        return -1;
    }
    return 0;
}

int vocab_id_for_word(const EncodedCorpus *corpus, const char *word) {
    if (!corpus || !word) {
        return -1;
    }
    for (int i = 0; i < corpus->vocab; i++) {
        if (strcmp(corpus->entries[i].word, word) == 0) {
            return i;
        }
    }
    return -1;
}

int vocab_encode_prompt(const EncodedCorpus *corpus, const char *text, int *out_ids, int max_ids) {
    size_t pos = 0;
    size_t n;
    char word[MAX_WORD_LEN + 1];
    int count = 0;

    if (!corpus || !text || !out_ids || max_ids <= 0) {
        return 0;
    }
    n = strlen(text);
    while (count < max_ids && next_word(text, n, &pos, word)) {
        int id = vocab_id_for_word(corpus, word);
        if (id >= 0) {
            out_ids[count++] = id;
        }
    }
    return count;
}
