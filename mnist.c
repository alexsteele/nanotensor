#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static float rand_uniform(unsigned int *seed) {
    *seed = (1103515245U * (*seed) + 12345U);
    return (float)((*seed) & 0x7fffffffU) / 2147483648.0f;
}

static uint32_t mnist_read_u32_be(FILE *f) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) {
        die("failed to read u32");
    }
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | (uint32_t)b[3];
}

MnistSet mnist_load(const char *images_path, const char *labels_path, int max_samples) {
    FILE *fi = fopen(images_path, "rb");
    FILE *fl = fopen(labels_path, "rb");
    MnistSet ds = {0};
    uint32_t img_magic;
    uint32_t lbl_magic;
    uint32_t n_images;
    uint32_t n_labels;
    uint32_t rows;
    uint32_t cols;
    unsigned char *buf;

    if (!fi || !fl) {
        die("failed to open MNIST files");
    }

    img_magic = mnist_read_u32_be(fi);
    n_images = mnist_read_u32_be(fi);
    rows = mnist_read_u32_be(fi);
    cols = mnist_read_u32_be(fi);

    lbl_magic = mnist_read_u32_be(fl);
    n_labels = mnist_read_u32_be(fl);

    if (img_magic != 2051U || lbl_magic != 2049U) {
        die("invalid MNIST magic (expecting raw IDX files, not .gz)");
    }
    if (rows != MNIST_ROWS || cols != MNIST_COLS) {
        die("unexpected MNIST image shape");
    }
    if (n_images != n_labels) {
        die("MNIST images/labels length mismatch");
    }

    ds.n = (int)n_images;
    if (max_samples > 0 && max_samples < ds.n) {
        ds.n = max_samples;
    }
    ds.rows = (int)rows;
    ds.cols = (int)cols;
    ds.images = (float *)malloc(sizeof(float) * (size_t)ds.n * MNIST_PIXELS);
    ds.labels = (unsigned char *)malloc((size_t)ds.n);
    buf = (unsigned char *)malloc((size_t)ds.n * MNIST_PIXELS);

    if (!ds.images || !ds.labels || !buf) {
        die("allocation failed loading MNIST");
    }

    if (fread(buf, 1, (size_t)ds.n * MNIST_PIXELS, fi) != (size_t)ds.n * MNIST_PIXELS) {
        die("failed to read MNIST images");
    }
    if (fread(ds.labels, 1, (size_t)ds.n, fl) != (size_t)ds.n) {
        die("failed to read MNIST labels");
    }

    for (int i = 0; i < ds.n * MNIST_PIXELS; i++) {
        ds.images[i] = (float)buf[i] / 255.0f;
    }

    free(buf);
    fclose(fi);
    fclose(fl);
    return ds;
}

void mnist_free(MnistSet *ds) {
    if (!ds) {
        return;
    }
    free(ds->images);
    free(ds->labels);
    ds->images = NULL;
    ds->labels = NULL;
    ds->n = 0;
}

void mnist_shuffle_indices(int *indices, int n, unsigned int *seed) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(rand_uniform(seed) * (float)(i + 1));
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

void mnist_gather_batch(const MnistSet *ds,
                        const int *indices,
                        int start,
                        int batch,
                        float *batch_images,
                        unsigned char *batch_labels) {
    for (int b = 0; b < batch; b++) {
        int idx = indices[start + b];
        memcpy(batch_images + (size_t)b * MNIST_PIXELS,
               ds->images + (size_t)idx * MNIST_PIXELS,
               sizeof(float) * MNIST_PIXELS);
        batch_labels[b] = ds->labels[idx];
    }
}

void mnist_gather_batch_images(const MnistSet *ds,
                               const int *indices,
                               int start,
                               int batch,
                               float *batch_images) {
    for (int b = 0; b < batch; b++) {
        int idx = indices[start + b];
        memcpy(batch_images + (size_t)b * MNIST_PIXELS,
               ds->images + (size_t)idx * MNIST_PIXELS,
               sizeof(float) * MNIST_PIXELS);
    }
}
