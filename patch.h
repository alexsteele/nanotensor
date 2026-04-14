#ifndef PATCH_H
#define PATCH_H

#include "tensor.h"

/*
 * patch.h
 *
 * Shared helpers for patch-based image models built on the 2D tensor API.
 * - Describes sliding-window patch geometry with PatchLayout
 * - Owns reusable extraction buffers with PatchBatch
 * - Extracts image batches into patch-row matrices for matmul-based models
 * - Provides simple mean-pooling helpers to group patch rows back per image
 */

typedef struct {
    int image_h;
    int image_w;
    int kernel_h;
    int kernel_w;
    int out_h;
    int out_w;
    int patches_per_image;
    int patch_dim;
} PatchLayout;

typedef struct {
    PatchLayout layout;
    int batch;
    float *buffer;
} PatchBatch;

PatchLayout patch_layout_make(int image_h, int image_w, int kernel_h, int kernel_w);
int patch_layout_num_rows(const PatchLayout *layout, int batch);

PatchBatch patch_batch_create(PatchLayout layout, int batch);
void patch_batch_free(PatchBatch *pb);

void patch_extract_batch(const PatchLayout *layout,
                         const float *images,
                         int batch,
                         float *out_rows);

Tensor *patch_batch_to_tensor(const PatchBatch *pb);
Tensor *patch_make_mean_pool_tensor(int batch, const PatchLayout *layout);
/* Builds the internal pooling tensor and registers both it and the returned
 * pooled output in `temps`, so they stay alive for autograd.
 */
Tensor *patch_mean_pool_rows(TensorList *temps, Tensor *patch_rows, int batch, const PatchLayout *layout);

#endif
