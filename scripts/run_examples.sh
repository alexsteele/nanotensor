#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON:-.venv/bin/python3}"

mkdir -p out examples

./autoencoder --epochs=20 --batch=32 --hidden=256 --latent=64 --opt=adam --loss=bce --lr=0.001 \
  --log=out/autoencoder_adam_bce_log.csv --recon=out/autoencoder_adam_bce_recon.csv
"$PYTHON_BIN" scripts/render_autoencoder_recon.py \
  out/autoencoder_adam_bce_recon.csv out/autoencoder_adam_bce_recon.png
cp out/autoencoder_adam_bce_recon.png examples/autoencoder_recon.png
cp out/autoencoder_adam_bce_log.csv examples/autoencoder_training_log.csv

./seq2seq --steps=2000 --attention=1 --opt=adam --lr=0.003 --report=out/seq2seq_report.txt
cp out/seq2seq_report.txt examples/seq2seq_report.txt

./skipgram --steps=2000 --report=out/skipgram_neighbors.txt
cp out/skipgram_neighbors.txt examples/skipgram_neighbors.txt

./ngram --steps=2000 --report=out/ngram_report.txt
cp out/ngram_report.txt examples/ngram_report.txt

./mnist_conv_demo --epochs=5 --log=out/mnist_training_log.csv
"$PYTHON_BIN" scripts/plot_training_loss.py out/mnist_training_log.csv out/mnist_plot_smoke.png
cp out/mnist_plot_smoke.png examples/mnist_plot_smoke.png

./mnist_resnet_demo --epochs=5 --batch=256 --dim=16 --hidden=32 --opt=adam --lr=0.001 \
  --train-limit=512 --test-limit=256 --log=out/mnist_resnet_example_log.csv
"$PYTHON_BIN" scripts/plot_training_loss.py \
  out/mnist_resnet_example_log.csv out/mnist_resnet_example_loss.png
cp out/mnist_resnet_example_loss.png examples/mnist_resnet_example_loss.png
cp out/mnist_resnet_example_log.csv examples/mnist_resnet_example_log.csv
