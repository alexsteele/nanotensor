# TODO

## Autoencoder

- compare a few latent sizes and hidden widths on MNIST reconstruction quality
- add a fixed qualitative reconstruction panel so runs are easy to compare
- try a denoising autoencoder variant after the plain baseline
- consider a convolutional autoencoder once the MLP baseline is well logged

## Seq2Seq

- compare plain RNN vs attention on the same fixed eval set with Adam
- add a cleaner qualitative eval panel with a fixed set of decode examples

## Optimizers

- compare SGD + momentum vs Adam on the autoencoder more systematically

## Language Demos

- compare skip-gram initialized embeddings against the current n-gram baseline
- improve sampling in `ngram.c` beyond greedy decoding

## Infra

- training helpers for running on different machines more cleanly than ad hoc
  scripts
- consider extending `tensor_bench.c` with a couple of end-to-end model-shaped
  microbenches once the primitive-op table settles
- consider caching the internal pool tensor inside `patch_mean_pool_rows(...)`
  if profiling shows the current rebuild-per-call approach matters
- continue replacing ad hoc temp-tensor ownership patterns with `TensorList`
  where it clearly simplifies demos without hiding lifetimes
