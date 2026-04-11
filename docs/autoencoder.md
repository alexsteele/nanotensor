# Autoencoder

`autoencoder.c` is a minimal MNIST reconstruction demo built on the core
tensor library.

## Architecture

The current model is a plain MLP autoencoder over flattened MNIST digits:

- input: `[batch, 784]`
- encoder hidden: `784 -> hidden` with `relu`
- latent bottleneck: `hidden -> latent` with `relu`
- decoder hidden: `latent -> hidden` with `relu`
- output head: `hidden -> 784` with `sigmoid`

The sigmoid output maps reconstructed pixels back into `[0, 1]`, matching the
normalized MNIST input range.

## Loss

Training uses mean-squared reconstruction loss between:

- the normalized input image
- the reconstructed output image

This is a simple baseline, not a variational autoencoder or denoising setup.

## CLI Knobs

The main options are:

- `--epochs=N`
- `--batch=N`
- `--hidden=N`
- `--latent=N`
- `--lr=FLOAT`
- `--log=PATH`

Defaults in the current demo are:

- `epochs=10`
- `batch=32`
- `hidden=128`
- `latent=32`
- `lr=0.01`
- `log=out/autoencoder_training_log.csv`

## Dataset

Like the existing MNIST classifier demo, this currently trains on the raw IDX
files under `data/mnist/` and uses the fixed subset defaults:

- train: `10000`
- eval: `2000`

## Notes

This first pass is intentionally small and readable:

- no convolutional encoder
- no tied weights
- no latent sampling
- no reconstruction image export yet

That makes it a good baseline for future comparison against deeper or more
structured autoencoder variants.
