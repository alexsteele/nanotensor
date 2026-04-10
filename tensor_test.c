#include "tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "ASSERT_TRUE failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return 1; \
        } \
    } while (0)

#define ASSERT_CLOSE(a, b, eps) \
    do { \
        float _da = (float)(a); \
        float _db = (float)(b); \
        if (fabsf(_da - _db) > (eps)) { \
            fprintf(stderr, "ASSERT_CLOSE failed at %s:%d: got %.7f expected %.7f (eps=%.7f)\n", \
                    __FILE__, __LINE__, _da, _db, (float)(eps)); \
            return 1; \
        } \
    } while (0)

static int test_basic_ops(void) {
    float av[] = {1, 2, 3, 4};
    float bv[] = {4, 3, 2, 1};
    Tensor *a = tensor_from_array(2, 2, av, 0);
    Tensor *b = tensor_from_array(2, 2, bv, 0);
    Tensor *add = tensor_add(a, b);
    Tensor *sub = tensor_sub(a, b);
    Tensor *mul = tensor_mul_elem(a, b);
    Tensor *sc = tensor_scalar_mul(a, 0.5f);

    ASSERT_CLOSE(add->data[0], 5.0f, 1e-6f);
    ASSERT_CLOSE(add->data[3], 5.0f, 1e-6f);
    ASSERT_CLOSE(sub->data[0], -3.0f, 1e-6f);
    ASSERT_CLOSE(sub->data[3], 3.0f, 1e-6f);
    ASSERT_CLOSE(mul->data[0], 4.0f, 1e-6f);
    ASSERT_CLOSE(mul->data[3], 4.0f, 1e-6f);
    ASSERT_CLOSE(sc->data[2], 1.5f, 1e-6f);

    tensor_free(a);
    tensor_free(b);
    tensor_free(add);
    tensor_free(sub);
    tensor_free(mul);
    tensor_free(sc);
    return 0;
}

static int test_tensor_copy(void) {
    float av[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor *a = tensor_from_array(2, 2, av, 1);
    Tensor *b = tensor_cpy(a);

    ASSERT_TRUE(b != NULL);
    ASSERT_TRUE(b != a);
    ASSERT_TRUE(b->rows == a->rows && b->cols == a->cols);
    ASSERT_TRUE(b->requires_grad == a->requires_grad);
    ASSERT_TRUE(tensor_equal(a, b));

    b->data[0] = 99.0f;
    ASSERT_CLOSE(a->data[0], 1.0f, 1e-6f);

    tensor_free(a);
    tensor_free(b);
    return 0;
}

static int test_broadcast_ops(void) {
    float av[] = {1, 2, 3, 4};
    float bv[] = {10, 20};
    float zv[] = {0, 0, 0, 0};
    Tensor *a = tensor_from_array(2, 2, av, 1);
    Tensor *b = tensor_from_array(1, 2, bv, 1);
    Tensor *z = tensor_from_array(2, 2, zv, 0);

    Tensor *add = tensor_add_broadcast(a, b);
    ASSERT_CLOSE(add->data[0], 11.0f, 1e-6f);
    ASSERT_CLOSE(add->data[1], 22.0f, 1e-6f);
    ASSERT_CLOSE(add->data[2], 13.0f, 1e-6f);
    ASSERT_CLOSE(add->data[3], 24.0f, 1e-6f);

    Tensor *add_loss = tensor_mse_loss(add, z);
    tensor_backward(add_loss);
    ASSERT_CLOSE(a->grad[0], 5.5f, 1e-6f);
    ASSERT_CLOSE(a->grad[1], 11.0f, 1e-6f);
    ASSERT_CLOSE(a->grad[2], 6.5f, 1e-6f);
    ASSERT_CLOSE(a->grad[3], 12.0f, 1e-6f);
    ASSERT_CLOSE(b->grad[0], 12.0f, 1e-6f);
    ASSERT_CLOSE(b->grad[1], 23.0f, 1e-6f);

    Tensor *mul = tensor_mul_broadcast(a, b);
    ASSERT_CLOSE(mul->data[0], 10.0f, 1e-6f);
    ASSERT_CLOSE(mul->data[1], 40.0f, 1e-6f);
    ASSERT_CLOSE(mul->data[2], 30.0f, 1e-6f);
    ASSERT_CLOSE(mul->data[3], 80.0f, 1e-6f);

    tensor_free(a);
    tensor_free(b);
    tensor_free(z);
    tensor_free(add);
    tensor_free(add_loss);
    tensor_free(mul);
    return 0;
}

static int test_axis_reductions(void) {
    float xv[] = {1, 2, 3, 4, 5, 6};
    float tv[] = {0, 0, 0};
    Tensor *x = tensor_from_array(2, 3, xv, 1);
    Tensor *sum0 = tensor_sum_axis(x, 0);
    Tensor *sum1 = tensor_sum_axis(x, 1);
    Tensor *mean0 = tensor_mean_axis(x, 0);
    Tensor *mean1 = tensor_mean_axis(x, 1);

    ASSERT_TRUE(sum0->rows == 1 && sum0->cols == 3);
    ASSERT_TRUE(sum1->rows == 2 && sum1->cols == 1);
    ASSERT_CLOSE(sum0->data[0], 5.0f, 1e-6f);
    ASSERT_CLOSE(sum0->data[1], 7.0f, 1e-6f);
    ASSERT_CLOSE(sum0->data[2], 9.0f, 1e-6f);
    ASSERT_CLOSE(sum1->data[0], 6.0f, 1e-6f);
    ASSERT_CLOSE(sum1->data[1], 15.0f, 1e-6f);
    ASSERT_CLOSE(mean0->data[0], 2.5f, 1e-6f);
    ASSERT_CLOSE(mean0->data[1], 3.5f, 1e-6f);
    ASSERT_CLOSE(mean0->data[2], 4.5f, 1e-6f);
    ASSERT_CLOSE(mean1->data[0], 2.0f, 1e-6f);
    ASSERT_CLOSE(mean1->data[1], 5.0f, 1e-6f);

    /* Gradient check through sum_axis(0). */
    Tensor *target = tensor_from_array(1, 3, tv, 0);
    Tensor *loss = tensor_mse_loss(sum0, target);
    tensor_backward(loss);
    ASSERT_CLOSE(x->grad[0], 10.0f / 3.0f, 1e-6f);
    ASSERT_CLOSE(x->grad[1], 14.0f / 3.0f, 1e-6f);
    ASSERT_CLOSE(x->grad[2], 6.0f, 1e-6f);
    ASSERT_CLOSE(x->grad[3], 10.0f / 3.0f, 1e-6f);
    ASSERT_CLOSE(x->grad[4], 14.0f / 3.0f, 1e-6f);
    ASSERT_CLOSE(x->grad[5], 6.0f, 1e-6f);

    tensor_free(x);
    tensor_free(sum0);
    tensor_free(sum1);
    tensor_free(mean0);
    tensor_free(mean1);
    tensor_free(target);
    tensor_free(loss);
    return 0;
}

static int test_matmul_and_bias(void) {
    float xv[] = {1, 2, 3, 4};
    float wv[] = {1, 0, 0, 1, 1, 1};
    float bv[] = {0.5f, -0.5f, 1.0f};
    Tensor *x = tensor_from_array(2, 2, xv, 0);
    Tensor *w = tensor_from_array(2, 3, wv, 0);
    Tensor *b = tensor_from_array(1, 3, bv, 0);

    Tensor *z = tensor_matmul(x, w);
    Tensor *y = tensor_add_bias(z, b);

    ASSERT_TRUE(z->rows == 2 && z->cols == 3);
    ASSERT_CLOSE(z->data[0], 3.0f, 1e-6f);
    ASSERT_CLOSE(z->data[1], 2.0f, 1e-6f);
    ASSERT_CLOSE(z->data[2], 2.0f, 1e-6f);
    ASSERT_CLOSE(y->data[0], 3.5f, 1e-6f);
    ASSERT_CLOSE(y->data[1], 1.5f, 1e-6f);
    ASSERT_CLOSE(y->data[2], 3.0f, 1e-6f);

    tensor_free(x);
    tensor_free(w);
    tensor_free(b);
    tensor_free(z);
    tensor_free(y);
    return 0;
}

static int test_activations(void) {
    float xv[] = {-1.0f, 0.0f, 2.0f};
    Tensor *x = tensor_from_array(1, 3, xv, 0);
    Tensor *r = tensor_relu(x);
    Tensor *s = tensor_sigmoid(x);
    Tensor *t = tensor_tanh(x);

    ASSERT_CLOSE(r->data[0], 0.0f, 1e-6f);
    ASSERT_CLOSE(r->data[2], 2.0f, 1e-6f);
    ASSERT_CLOSE(s->data[1], 0.5f, 1e-6f);
    ASSERT_CLOSE(t->data[1], 0.0f, 1e-6f);
    ASSERT_CLOSE(t->data[2], tanhf(2.0f), 1e-6f);

    tensor_free(x);
    tensor_free(r);
    tensor_free(s);
    tensor_free(t);
    return 0;
}

static int test_softmax_cross_entropy(void) {
    float lv[] = {1.0f, 2.0f, 0.0f};
    float tv[] = {0.0f, 1.0f, 0.0f};
    Tensor *logits = tensor_from_array(1, 3, lv, 1);
    Tensor *target = tensor_from_array(1, 3, tv, 0);
    Tensor *probs = tensor_softmax(logits);
    Tensor *loss = tensor_cross_entropy(probs, target);

    tensor_backward(loss);

    ASSERT_CLOSE(probs->data[0] + probs->data[1] + probs->data[2], 1.0f, 1e-6f);
    ASSERT_TRUE(probs->data[1] > probs->data[0]);
    ASSERT_TRUE(probs->data[0] > probs->data[2]);
    ASSERT_CLOSE(loss->data[0], -logf(probs->data[1]), 1e-6f);

    /* For softmax + CE with one-hot target and batch=1: dL/dlogits = probs - target. */
    ASSERT_CLOSE(logits->grad[0], probs->data[0] - 0.0f, 1e-5f);
    ASSERT_CLOSE(logits->grad[1], probs->data[1] - 1.0f, 1e-5f);
    ASSERT_CLOSE(logits->grad[2], probs->data[2] - 0.0f, 1e-5f);

    tensor_free(logits);
    tensor_free(target);
    tensor_free(probs);
    tensor_free(loss);
    return 0;
}

static int test_pow_sqrt(void) {
    float xv[] = {2.0f};
    float tv[] = {0.0f};
    Tensor *x = tensor_from_array(1, 1, xv, 1);
    Tensor *p = tensor_pow(x, 3.0f);
    Tensor *t = tensor_from_array(1, 1, tv, 0);
    Tensor *loss = tensor_mse_loss(p, t);

    tensor_backward(loss);
    ASSERT_CLOSE(p->data[0], 8.0f, 1e-6f);
    ASSERT_CLOSE(x->grad[0], 192.0f, 1e-4f);

    tensor_free(x);
    tensor_free(p);
    tensor_free(t);
    tensor_free(loss);

    xv[0] = 4.0f;
    x = tensor_from_array(1, 1, xv, 1);
    Tensor *s = tensor_sqrt(x);
    t = tensor_from_array(1, 1, tv, 0);
    loss = tensor_mse_loss(s, t);

    tensor_backward(loss);
    ASSERT_CLOSE(s->data[0], 2.0f, 1e-6f);
    ASSERT_CLOSE(x->grad[0], 1.0f, 1e-6f);

    tensor_free(x);
    tensor_free(s);
    tensor_free(t);
    tensor_free(loss);
    return 0;
}

static int test_layernorm(void) {
    float xv[] = {1.0f, 3.0f, 2.0f, 4.0f};
    float gv[] = {1.0f, 1.0f};
    float bv[] = {0.0f, 0.0f};
    Tensor *x = tensor_from_array(2, 2, xv, 1);
    Tensor *gamma = tensor_from_array(1, 2, gv, 1);
    Tensor *beta = tensor_from_array(1, 2, bv, 1);
    Tensor *y = tensor_layernorm(x, gamma, beta, 1e-5f);
    Tensor *sum0 = tensor_sum_axis(y, 0);
    Tensor *loss = tensor_mean_axis(sum0, 1);

    ASSERT_CLOSE(y->data[0], -1.0f, 2e-3f);
    ASSERT_CLOSE(y->data[1], 1.0f, 2e-3f);
    ASSERT_CLOSE(y->data[2], -1.0f, 2e-3f);
    ASSERT_CLOSE(y->data[3], 1.0f, 2e-3f);

    tensor_backward(loss);
    ASSERT_CLOSE(gamma->grad[0], -1.0f, 2e-3f);
    ASSERT_CLOSE(gamma->grad[1], 1.0f, 2e-3f);
    ASSERT_CLOSE(beta->grad[0], 1.0f, 1e-6f);
    ASSERT_CLOSE(beta->grad[1], 1.0f, 1e-6f);
    ASSERT_CLOSE(x->grad[0], 0.0f, 1e-3f);
    ASSERT_CLOSE(x->grad[1], 0.0f, 1e-3f);
    ASSERT_CLOSE(x->grad[2], 0.0f, 1e-3f);
    ASSERT_CLOSE(x->grad[3], 0.0f, 1e-3f);

    tensor_free(x);
    tensor_free(gamma);
    tensor_free(beta);
    tensor_free(y);
    tensor_free(sum0);
    tensor_free(loss);
    return 0;
}

static int test_cmp_ops(void) {
    float av[] = {1, 2, 3};
    float bv[] = {1, 1, 4};
    Tensor *a = tensor_from_array(1, 3, av, 0);
    Tensor *b = tensor_from_array(1, 3, bv, 0);
    Tensor *c = tensor_cmp(a, b);

    ASSERT_CLOSE(c->data[0], 0.0f, 1e-6f);
    ASSERT_CLOSE(c->data[1], 1.0f, 1e-6f);
    ASSERT_CLOSE(c->data[2], -1.0f, 1e-6f);
    ASSERT_TRUE(!tensor_equal(a, b));
    ASSERT_TRUE(tensor_allclose(a, b, 1.0f, 0.0f));

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    return 0;
}

static int test_backward_mse_matmul(void) {
    float xv[] = {2.0f, -1.0f};
    float wv[] = {0.5f, -0.25f};
    float yv[] = {1.0f};

    Tensor *x = tensor_from_array(1, 2, xv, 0);
    Tensor *w = tensor_from_array(2, 1, wv, 1);
    Tensor *y = tensor_from_array(1, 1, yv, 0);

    Tensor *pred = tensor_matmul(x, w);
    Tensor *loss = tensor_mse_loss(pred, y);
    tensor_backward(loss);

    ASSERT_CLOSE(pred->data[0], 1.25f, 1e-6f);
    ASSERT_CLOSE(loss->data[0], 0.0625f, 1e-6f);

    /* dloss/dpred = 2*(pred-y) = 0.5, so dloss/dw = x^T * 0.5 */
    ASSERT_CLOSE(w->grad[0], 1.0f, 1e-6f);
    ASSERT_CLOSE(w->grad[1], -0.5f, 1e-6f);

    tensor_free(x);
    tensor_free(w);
    tensor_free(y);
    tensor_free(pred);
    tensor_free(loss);
    return 0;
}

static int test_reshape_transpose_slice(void) {
    float av[] = {1, 2, 3, 4, 5, 6};
    Tensor *a = tensor_from_array(2, 3, av, 0);
    Tensor *r = tensor_reshape(a, 3, 2);
    Tensor *t = tensor_transpose(a);
    Tensor *s = tensor_slice(a, 0, 2, 1, 3);

    ASSERT_TRUE(r->rows == 3 && r->cols == 2);
    ASSERT_CLOSE(r->data[0], 1.0f, 1e-6f);
    ASSERT_CLOSE(r->data[5], 6.0f, 1e-6f);

    ASSERT_TRUE(t->rows == 3 && t->cols == 2);
    ASSERT_CLOSE(t->data[0], 1.0f, 1e-6f);
    ASSERT_CLOSE(t->data[1], 4.0f, 1e-6f);
    ASSERT_CLOSE(t->data[4], 3.0f, 1e-6f);
    ASSERT_CLOSE(t->data[5], 6.0f, 1e-6f);

    ASSERT_TRUE(s->rows == 2 && s->cols == 2);
    ASSERT_CLOSE(s->data[0], 2.0f, 1e-6f);
    ASSERT_CLOSE(s->data[1], 3.0f, 1e-6f);
    ASSERT_CLOSE(s->data[2], 5.0f, 1e-6f);
    ASSERT_CLOSE(s->data[3], 6.0f, 1e-6f);

    tensor_free(a);
    tensor_free(r);
    tensor_free(t);
    tensor_free(s);
    return 0;
}

static int test_backward_layout_ops(void) {
    float xv[] = {1, 2, 3, 4, 5, 6};
    float yv[] = {0, 0, 0, 0};
    Tensor *x = tensor_from_array(2, 3, xv, 1);
    Tensor *s = tensor_slice(x, 0, 2, 1, 3);
    Tensor *r = tensor_reshape(s, 1, 4);
    Tensor *t = tensor_transpose(r);
    Tensor *y = tensor_from_array(4, 1, yv, 0);
    Tensor *loss = tensor_mse_loss(t, y);

    tensor_backward(loss);

    ASSERT_CLOSE(loss->data[0], 18.5000f, 1e-6f);
    ASSERT_CLOSE(x->grad[0], 0.0f, 1e-6f);
    ASSERT_CLOSE(x->grad[1], 0.0f, 1e-6f);
    ASSERT_CLOSE(x->grad[2], 0.0f, 1e-6f);
    ASSERT_CLOSE(x->grad[3], 0.0f, 1e-6f);
    ASSERT_CLOSE(x->grad[4], 0.0f, 1e-6f);
    ASSERT_CLOSE(x->grad[5], 0.0f, 1e-6f);

    tensor_free(x);
    tensor_free(s);
    tensor_free(r);
    tensor_free(t);
    tensor_free(y);
    tensor_free(loss);
    return 0;
}

static int test_save_load(void) {
    const char *single_path = "tensor_test_single.bin";
    const char *snap_path = "tensor_test_snapshot.bin";

    float av[] = {1, 2, 3, 4};
    Tensor *a = tensor_from_array(2, 2, av, 1);

    ASSERT_TRUE(tensor_save(a, single_path) == 0);
    Tensor *b = tensor_load(single_path);
    ASSERT_TRUE(b != NULL);
    ASSERT_TRUE(b->requires_grad == 1);
    ASSERT_TRUE(tensor_equal(a, b));

    float p1v[] = {1.0f, 2.0f};
    float p2v[] = {3.0f, 4.0f};
    Tensor *p1 = tensor_from_array(1, 2, p1v, 1);
    Tensor *p2 = tensor_from_array(1, 2, p2v, 1);
    Tensor *params[] = {p1, p2};

    ASSERT_TRUE(tensor_snapshot_save(params, 2, snap_path) == 0);
    p1->data[0] = 99.0f;
    ASSERT_TRUE(tensor_snapshot_load(params, 2, snap_path) == 0);
    ASSERT_CLOSE(p1->data[0], 1.0f, 1e-6f);

    tensor_free(a);
    tensor_free(b);
    tensor_free(p1);
    tensor_free(p2);

    remove(single_path);
    remove(snap_path);
    return 0;
}

int main(void) {
    if (test_basic_ops() != 0) return 1;
    if (test_tensor_copy() != 0) return 1;
    if (test_broadcast_ops() != 0) return 1;
    if (test_axis_reductions() != 0) return 1;
    if (test_matmul_and_bias() != 0) return 1;
    if (test_activations() != 0) return 1;
    if (test_pow_sqrt() != 0) return 1;
    if (test_layernorm() != 0) return 1;
    if (test_softmax_cross_entropy() != 0) return 1;
    if (test_cmp_ops() != 0) return 1;
    if (test_backward_mse_matmul() != 0) return 1;
    if (test_reshape_transpose_slice() != 0) return 1;
    if (test_backward_layout_ops() != 0) return 1;
    if (test_save_load() != 0) return 1;

    printf("All tensor tests passed.\n");
    return 0;
}
