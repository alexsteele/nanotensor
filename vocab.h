#ifndef VOCAB_H
#define VOCAB_H

#include <stddef.h>

typedef struct {
    char *word;
    int count;
} VocabEntry;

typedef struct {
    VocabEntry *entries;
    int vocab;
    int total_words;
    int *tokens;
    int n_tokens;
} EncodedCorpus;

char *vocab_read_file(const char *path, size_t *out_size);
int vocab_build_corpus(const char *text, size_t n, int vocab_limit, EncodedCorpus *out);
void vocab_free_corpus(EncodedCorpus *corpus);
int vocab_save(const EncodedCorpus *corpus, const char *path);
int vocab_id_for_word(const EncodedCorpus *corpus, const char *word);
int vocab_encode_prompt(const EncodedCorpus *corpus, const char *text, int *out_ids, int max_ids);

#endif
