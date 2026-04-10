
# Issues

"Why is eval loss so poor?"

Probably a mix of “small model” and “hard-to-optimize setup,” not one single bug.

In this demo, the architecture in [mnist_conv_demo.c](/Users/alex/Code/nanotensor/mnist_conv_demo.c) is pretty limited:
- each `5x5` patch is projected independently
- patch logits are then averaged with a fixed mean pool
- that mean pool throws away spatial information and can wash out a few strong digit strokes with lots of background patches
- there’s only one hidden conv-like stage and no deeper feature hierarchy

A few other reasons accuracy can stay low after 5 epochs:
- optimization is very basic: plain SGD in [tensor.c](/Users/alex/Code/nanotensor/tensor.c) with no momentum or Adam
- there’s no data shuffling between epochs, so batches always arrive in the same order
- the default channel count is small (`8`), which limits capacity
- if you’re training on the subset defaults (`10000` train / `2000` test), that’s also a smaller signal than full MNIST
- 5 epochs is not much for a weak architecture with simple SGD

One architecture-specific issue matters a lot here: after the classifier head, the model does:

- per-patch class scores `[batch * patches, 10]`
- mean pool over all patches to `[batch, 10]`

That means every patch votes equally. For MNIST, many patches are mostly blank background, so equal averaging can drown out the informative patches.

If we want the quickest accuracy bump, I’d try these first:
- increase `channels` from `8` to `32` or `64`
- train longer, like `20-30` epochs
- shuffle training samples each epoch
- replace uniform mean pooling with something less lossy

