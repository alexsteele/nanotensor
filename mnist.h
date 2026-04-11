#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>

#define MNIST_ROWS 28
#define MNIST_COLS 28
#define MNIST_PIXELS (MNIST_ROWS * MNIST_COLS)

typedef struct {
    int n;
    int rows;
    int cols;
    float *images;      /* [n, rows*cols], normalized to [0, 1] */
    unsigned char *labels;
} MnistSet;

MnistSet mnist_load(const char *images_path, const char *labels_path, int max_samples);
void mnist_free(MnistSet *ds);
void mnist_shuffle_indices(int *indices, int n, unsigned int *seed);
void mnist_gather_batch(const MnistSet *ds,
                        const int *indices,
                        int start,
                        int batch,
                        float *batch_images,
                        unsigned char *batch_labels);
void mnist_gather_batch_images(const MnistSet *ds,
                               const int *indices,
                               int start,
                               int batch,
                               float *batch_images);

#endif
