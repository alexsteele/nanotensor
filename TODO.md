# TODO

## Autoencoder

- compare a few latent sizes and hidden widths on MNIST reconstruction quality
- add a fixed qualitative reconstruction panel so runs are easy to compare
- try a denoising autoencoder variant after the plain baseline
- consider a convolutional autoencoder once the MLP baseline is well logged

## Seq2Seq

- switch eval to a fixed held-out synthetic set so `eval_tok` and `eval_seq`
  are less noisy across checkpoints and runs
- compare plain RNN vs attention on the same fixed eval set
- try a learned attention scoring layer if dot attention plateaus

## MNIST Shared Infra

- consider pulling the IDX reader / MNIST dataset logic into a shared `mnist.h`
  + `mnist.c` loader used by classifier and autoencoder demos

## Optimizers

- add an Adam optimizer to the tensor library
- compare SGD + momentum vs Adam on MNIST, seq2seq, and autoencoder demos

## Language Demos

- compare skip-gram initialized embeddings against the current n-gram baseline
- improve sampling in `ngram.c` beyond greedy decoding

## Infra

- training helpers for running on different machines more cleanly than ad hoc
  scripts
