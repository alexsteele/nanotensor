#include "tensor.h"

/*
 * llm.c
 *
 * Minimal character-level language model demo using nanotensor.
 * - Trains a 2-layer MLP next-character predictor on a text corpus
 * - Generates text from a prompt with temperature sampling
 *
 * Usage:
 *   ./llm <text_path> [steps] [prompt] [batch] [gen_len] [lr] [temperature] [hidden]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_VOCAB 256

typedef struct {
    int vocab;
    int hidden;
    Tensor *W1;
    Tensor *b1;
    Tensor *W2;
    Tensor *b2;
} CharMLP;

static float rand_uniform(unsigned int *seed) {
    *seed = (1103515245U * (*seed) + 12345U);
    return (float)((*seed) & 0x7fffffffU) / 2147483648.0f;
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

static int build_vocab(const char *text, size_t n, int char_to_id[256], unsigned char id_to_char[MAX_VOCAB]) {
    int vocab = 0;
    size_t i;

    for (i = 0; i < 256; i++) {
        char_to_id[i] = -1;
    }

    for (i = 0; i < n; i++) {
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

static int *encode_text(const char *text, size_t n, const int char_to_id[256]) {
    int *tok = (int *)malloc(sizeof(int) * n);
    size_t i;
    if (!tok) {
        return NULL;
    }
    for (i = 0; i < n; i++) {
        tok[i] = char_to_id[(unsigned char)text[i]];
    }
    return tok;
}

static Tensor *make_one_hot(const int *idx, int n, int vocab) {
    Tensor *t = tensor_create(n, vocab, 0);
    int i;
    tensor_fill(t, 0.0f);
    for (i = 0; i < n; i++) {
        t->data[i * vocab + idx[i]] = 1.0f;
    }
    return t;
}

static int sample_from_probs(const float *probs, int n, unsigned int *seed) {
    float r = rand_uniform(seed);
    float acc = 0.0f;
    int i;

    for (i = 0; i < n; i++) {
        acc += probs[i];
        if (r <= acc) {
            return i;
        }
    }
    return n - 1;
}

static int charmlp_init(CharMLP *m, int vocab, int hidden, unsigned int *seed) {
    m->vocab = vocab;
    m->hidden = hidden;
    m->W1 = tensor_create(vocab, hidden, 1);
    m->b1 = tensor_create(1, hidden, 1);
    m->W2 = tensor_create(hidden, vocab, 1);
    m->b2 = tensor_create(1, vocab, 1);
    if (!m->W1 || !m->b1 || !m->W2 || !m->b2) {
        return -1;
    }

    tensor_fill_randn(m->W1, 0.0f, 0.03f, seed);
    tensor_fill_randn(m->W2, 0.0f, 0.03f, seed);
    tensor_fill(m->b1, 0.0f);
    tensor_fill(m->b2, 0.0f);
    return 0;
}

static void charmlp_free(CharMLP *m) {
    tensor_free(m->W1);
    tensor_free(m->b1);
    tensor_free(m->W2);
    tensor_free(m->b2);
}

static void charmlp_params(CharMLP *m, Tensor **params, size_t *n_params) {
    params[0] = m->W1;
    params[1] = m->b1;
    params[2] = m->W2;
    params[3] = m->b2;
    *n_params = 4;
}

static int sample_next_char(CharMLP *m, int current_id, float temperature, unsigned int *seed) {
    int idx[1] = {current_id};
    TensorList temps = {0};
    Tensor *x = tensor_list_add(&temps, make_one_hot(idx, 1, m->vocab));
    Tensor *h1_lin;
    Tensor *h1_bias;
    Tensor *h1;
    Tensor *h2_lin;
    Tensor *logits;
    Tensor *scaled;
    Tensor *probs;
    int next;

    if (temperature < 1e-6f) {
        temperature = 1e-6f;
    }

    h1_lin = tensor_list_add(&temps, tensor_matmul(x, m->W1));
    h1_bias = tensor_list_add(&temps, tensor_add_bias(h1_lin, m->b1));
    h1 = tensor_list_add(&temps, tensor_tanh(h1_bias));
    h2_lin = tensor_list_add(&temps, tensor_matmul(h1, m->W2));
    logits = tensor_list_add(&temps, tensor_add_bias(h2_lin, m->b2));
    scaled = tensor_list_add(&temps, tensor_scalar_mul(logits, 1.0f / temperature));
    probs = tensor_list_add(&temps, tensor_softmax(scaled));
    next = sample_from_probs(probs->data, probs->cols, seed);

    tensor_list_free(&temps);
    return next;
}

static int id_for_char(const int char_to_id[256], char c, int fallback_id) {
    int id = char_to_id[(unsigned char)c];
    return id >= 0 ? id : fallback_id;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "shakespeare.txt";
    int steps = argc > 2 ? atoi(argv[2]) : 3000;
    const char *prompt = argc > 3 ? argv[3] : "To be";
    int batch_size = argc > 4 ? atoi(argv[4]) : 64;
    int gen_len = argc > 5 ? atoi(argv[5]) : 300;
    float lr = argc > 6 ? (float)atof(argv[6]) : 0.2f;
    float temperature = argc > 7 ? (float)atof(argv[7]) : 0.9f;
    int hidden = argc > 8 ? atoi(argv[8]) : 96;

    size_t text_len = 0;
    char *text = read_file(path, &text_len);
    int char_to_id[256];
    unsigned char id_to_char[MAX_VOCAB];
    int *tokens;
    int vocab;
    CharMLP model;
    Tensor *params[4];
    size_t n_params = 0;
    unsigned int seed = 1337;
    int step;

    if (!text) {
        fprintf(stderr, "Failed to read '%s'\n", path);
        fprintf(stderr, "Usage: ./llm <text_path> [steps] [prompt] [batch] [gen_len] [lr] [temperature] [hidden]\n");
        return 1;
    }
    if (text_len < 2) {
        fprintf(stderr, "Text corpus is too small\n");
        free(text);
        return 1;
    }

    vocab = build_vocab(text, text_len, char_to_id, id_to_char);
    if (vocab <= 1) {
        fprintf(stderr, "Failed to build vocabulary\n");
        free(text);
        return 1;
    }

    tokens = encode_text(text, text_len, char_to_id);
    if (!tokens) {
        fprintf(stderr, "Failed to encode text\n");
        free(text);
        return 1;
    }

    if (batch_size <= 0) batch_size = 64;
    if (steps <= 0) steps = 3000;
    if (gen_len <= 0) gen_len = 300;
    if (hidden <= 0) hidden = 96;

    if (charmlp_init(&model, vocab, hidden, &seed) != 0) {
        fprintf(stderr, "Failed to initialize model\n");
        free(tokens);
        free(text);
        return 1;
    }
    charmlp_params(&model, params, &n_params);

    printf("loaded %zu bytes, vocab=%d\n", text_len, vocab);
    printf("training 2-layer char MLP: hidden=%d steps=%d batch=%d lr=%.4f\n", hidden, steps, batch_size, lr);

    for (step = 1; step <= steps; step++) {
        int b;
        int *x_idx = (int *)malloc(sizeof(int) * (size_t)batch_size);
        int *y_idx = (int *)malloc(sizeof(int) * (size_t)batch_size);
        TensorList temps = {0};
        Tensor *X;
        Tensor *Y;
        Tensor *h1_lin;
        Tensor *h1_bias;
        Tensor *h1;
        Tensor *h2_lin;
        Tensor *logits;
        Tensor *probs;
        Tensor *loss;

        if (!x_idx || !y_idx) {
            fprintf(stderr, "allocation failed\n");
            free(x_idx);
            free(y_idx);
            charmlp_free(&model);
            free(tokens);
            free(text);
            return 1;
        }

        for (b = 0; b < batch_size; b++) {
            size_t pos = (size_t)(rand_uniform(&seed) * (float)(text_len - 1));
            x_idx[b] = tokens[pos];
            y_idx[b] = tokens[pos + 1];
        }

        X = tensor_list_add(&temps, make_one_hot(x_idx, batch_size, vocab));
        Y = tensor_list_add(&temps, make_one_hot(y_idx, batch_size, vocab));

        h1_lin = tensor_list_add(&temps, tensor_matmul(X, model.W1));
        h1_bias = tensor_list_add(&temps, tensor_add_bias(h1_lin, model.b1));
        h1 = tensor_list_add(&temps, tensor_tanh(h1_bias));
        h2_lin = tensor_list_add(&temps, tensor_matmul(h1, model.W2));
        logits = tensor_list_add(&temps, tensor_add_bias(h2_lin, model.b2));
        probs = tensor_list_add(&temps, tensor_softmax(logits));
        loss = tensor_list_add(&temps, tensor_cross_entropy(probs, Y));

        tensor_backward(loss);
        tensor_sgd_step(params, n_params, lr);

        if (step == 1 || step % 100 == 0) {
            printf("step %4d loss %.6f\n", step, loss->data[0]);
        }

        tensor_list_free(&temps);
        free(x_idx);
        free(y_idx);
    }

    {
        size_t prompt_len = strlen(prompt);
        int current_id = id_for_char(char_to_id, prompt_len ? prompt[prompt_len - 1] : 0, tokens[0]);
        int i;

        printf("\n--- generation ---\n");
        printf("%s", prompt);
        for (i = 0; i < gen_len; i++) {
            int next = sample_next_char(&model, current_id, temperature, &seed);
            putchar((char)id_to_char[next]);
            current_id = next;
        }
        printf("\n");
    }

    charmlp_free(&model);
    free(tokens);
    free(text);
    return 0;
}
