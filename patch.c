#include "patch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

PatchLayout patch_layout_make(int image_h, int image_w, int kernel_h, int kernel_w) {
    PatchLayout layout;

    if (image_h <= 0 || image_w <= 0 || kernel_h <= 0 || kernel_w <= 0) {
        die("patch_layout_make: dimensions must be > 0");
    }
    if (kernel_h > image_h || kernel_w > image_w) {
        die("patch_layout_make: kernel must fit inside image");
    }

    layout.image_h = image_h;
    layout.image_w = image_w;
    layout.kernel_h = kernel_h;
    layout.kernel_w = kernel_w;
    layout.out_h = image_h - kernel_h + 1;
    layout.out_w = image_w - kernel_w + 1;
    layout.patches_per_image = layout.out_h * layout.out_w;
    layout.patch_dim = kernel_h * kernel_w;
    return layout;
}

int patch_layout_num_rows(const PatchLayout *layout, int batch) {
    if (!layout || batch <= 0) {
        die("patch_layout_num_rows: invalid layout or batch");
    }
    return batch * layout->patches_per_image;
}

PatchBatch patch_batch_create(PatchLayout layout, int batch) {
    PatchBatch pb;
    int rows = patch_layout_num_rows(&layout, batch);

    pb.layout = layout;
    pb.batch = batch;
    pb.buffer = (float *)malloc(sizeof(float) * (size_t)rows * (size_t)layout.patch_dim);
    if (!pb.buffer) {
        die("patch_batch_create: allocation failed");
    }
    return pb;
}

void patch_batch_free(PatchBatch *pb) {
    if (!pb) {
        return;
    }
    free(pb->buffer);
    pb->buffer = NULL;
    pb->batch = 0;
    memset(&pb->layout, 0, sizeof(pb->layout));
}

void patch_extract_batch(const PatchLayout *layout,
                         const float *images,
                         int batch,
                         float *out_rows) {
    if (!layout || !images || !out_rows || batch <= 0) {
        die("patch_extract_batch: invalid arguments");
    }

    for (int b = 0; b < batch; b++) {
        const float *img = images + (size_t)b * (size_t)layout->image_h * (size_t)layout->image_w;
        for (int y = 0; y < layout->out_h; y++) {
            for (int x = 0; x < layout->out_w; x++) {
                int row = (b * layout->out_h + y) * layout->out_w + x;
                float *dst = out_rows + (size_t)row * (size_t)layout->patch_dim;
                int k = 0;
                for (int ky = 0; ky < layout->kernel_h; ky++) {
                    for (int kx = 0; kx < layout->kernel_w; kx++) {
                        dst[k++] = img[(size_t)(y + ky) * (size_t)layout->image_w + (size_t)(x + kx)];
                    }
                }
            }
        }
    }
}

Tensor *patch_batch_to_tensor(const PatchBatch *pb) {
    if (!pb || !pb->buffer || pb->batch <= 0) {
        die("patch_batch_to_tensor: invalid patch batch");
    }
    return tensor_from_array(
        patch_layout_num_rows(&pb->layout, pb->batch), pb->layout.patch_dim, pb->buffer, 0);
}

Tensor *patch_make_mean_pool_tensor(int batch, const PatchLayout *layout) {
    Tensor *pool;
    float scale;

    if (!layout || batch <= 0) {
        die("patch_make_mean_pool_tensor: invalid layout or batch");
    }

    pool = tensor_create(batch, batch * layout->patches_per_image, 0);
    scale = 1.0f / (float)layout->patches_per_image;
    tensor_fill(pool, 0.0f);
    for (int b = 0; b < batch; b++) {
        int base = b * layout->patches_per_image;
        for (int p = 0; p < layout->patches_per_image; p++) {
            pool->data[b * pool->cols + base + p] = scale;
        }
    }
    return pool;
}

Tensor *patch_mean_pool_rows(Tensor *patch_rows, int batch, const PatchLayout *layout) {
    Tensor *pool;
    Tensor *pooled;
    int expected_rows;

    if (!patch_rows || !layout) {
        die("patch_mean_pool_rows: invalid arguments");
    }
    expected_rows = patch_layout_num_rows(layout, batch);
    if (patch_rows->rows != expected_rows) {
        die("patch_mean_pool_rows: patch row count mismatch");
    }

    /* TODO: if patch pooling shows up in profiling, hide a cached pool tensor
     * behind this helper instead of rebuilding it per call. */
    pool = patch_make_mean_pool_tensor(batch, layout);
    pooled = tensor_matmul(pool, patch_rows);
    tensor_free(pool);
    return pooled;
}
