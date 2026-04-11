# Issues

"Why is eval loss so poor?"

Probably a mix of “small model” and “hard-to-optimize setup,” not one single bug.

In this demo, the architecture in [convnet.c](/Users/alex/Code/nanotensor/convnet.c)
is pretty limited:

- each `5x5` patch is projected independently
- patch logits are then averaged with a fixed mean pool
- that mean pool throws away spatial information and can wash out a few strong digit strokes with
  lots of background patches
- there’s only one hidden conv-like stage and no deeper feature hierarchy

A few other reasons accuracy can stay low after 5 epochs:

- optimization is very basic: plain SGD in [tensor.c](/Users/alex/Code/nanotensor/tensor.c) with no
  momentum or Adam
- there’s no data shuffling between epochs, so batches always arrive in the same order
- the default channel count is small (`8`), which limits capacity
- if you’re training on the subset defaults (`10000` train / `2000` test), that’s also a smaller
  signal than full MNIST
- 5 epochs is not much for a weak architecture with simple SGD

One architecture-specific issue matters a lot here: after the classifier head, the model does:

- per-patch class scores `[batch * patches, 10]`
- mean pool over all patches to `[batch, 10]`

That means every patch votes equally. For MNIST, many patches are mostly blank background, so equal
averaging can drown out the informative patches.

If we want the quickest accuracy bump, I’d try these first:

- increase `channels` from `8` to `32` or `64`
- train longer, like `20-30` epochs
- shuffle training samples each epoch
- replace uniform mean pooling with something less lossy

# Knobs

The current demo exposes a few simple CLI knobs:

- `--epochs=N`: more epochs means more passes over the data; this often helps because the model is
  small and plain SGD learns slowly
- `--lr=FLOAT`: learning rate controls step size; too small learns slowly, too large can bounce
  around or diverge
- `--batch=N`: batch size controls how many images contribute to each SGD update; smaller batches
  give noisier but more frequent updates
- `--channels=N`: channel count is the width of the patch feature extractor; larger values usually
  improve accuracy up to the point where pooling becomes the bottleneck
- `--momentum=FLOAT`: momentum smooths SGD updates with a velocity term; values like `0.9` often
  converge faster than plain SGD

The demo now shuffles the training order each epoch before batching. That is usually a better
default than reusing the same fixed sample order every epoch.

# Experiment Log

We should track one short entry per meaningful training version keyed by git commit.

Suggested fields:

- commit
- config
- final train loss
- final eval loss
- final test acc / test error
- brief notes

### `badaa98` plain SGD baseline

- config: `epochs=3 batch=32 channels=8 lr=0.03 momentum=0.0`
- final train loss: `2.298944`
- final test acc / error: `0.115927 / 0.884073`
- notes: shuffled training order enabled, but plain SGD barely moved off chance-level accuracy on
  this setup

### `badaa98` shuffled + momentum SGD

- config: `epochs=3 batch=32 channels=8 lr=0.03 momentum=0.9`
- final train loss: `2.119055`
- final test acc / error: `0.212702 / 0.787298`
- notes: same architecture and data subset as baseline; momentum produced a clear early improvement
  in both loss and test accuracy

### `badaa98` wider model + longer run

- config: `epochs=10 batch=32 channels=32 lr=0.03 momentum=0.9`
- final train loss: `1.469522`
- final test acc / error: `0.450101 / 0.549899`
- notes: much stronger than the 8-channel runs; accuracy climbed steadily through 10 epochs, though
  this architecture is still far below a standard MNIST convnet
