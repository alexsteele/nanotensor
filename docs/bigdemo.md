# Bigdemo

`scripts/run_bigdemo.sh` is a curated showcase runner for the strongest current
demo configurations in the repo.

It is meant to be heavier than the quick `examples` refresh:

- stronger training settings
- distinct `out/bigdemo_*` artifact names
- one end-of-run summary table with headline metrics

## What It Runs

The current script runs:

- `autoencoder`
- `seq2seq`
- `skipgram`
- `ngram`
- `mnist_conv_demo`
- `resnet_demo`

## Build And Run

Strong recommendation on macOS: use the Accelerate-backed BLAS build.

`bigdemo` is dominated by matmul-heavy training loops, so the BLAS backend can
make the run dramatically faster than the naive kernel.

Preferred command on macOS:

```bash
make BLAS=accelerate bigdemo
```

Plain portable build:

```bash
make bigdemo
```

That target builds the needed binaries and then runs:

- [`scripts/run_bigdemo.sh`](/Users/alex/Code/nanotensor/scripts/run_bigdemo.sh)

## Output Convention

Artifacts are written to `out/` with a `bigdemo_` prefix, for example:

- `out/bigdemo_autoencoder_recon.png`
- `out/bigdemo_seq2seq_report.txt`
- `out/bigdemo_skipgram_report.txt`
- `out/bigdemo_ngram_report.txt`
- `out/bigdemo_conv_loss.png`
- `out/bigdemo_resnet_loss.png`

The script finishes by printing a compact summary table that includes one
headline metric per model and the most relevant artifact path.
