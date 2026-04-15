#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON:-.venv/bin/python3}"

mkdir -p out

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

csv_last_field() {
  local path="$1"
  local field="$2"
  awk -F, -v field="$field" 'NF > 0 && $1 !~ /^#/ { last = $field } END { print last }' "$path"
}

ngram_eval_summary() {
  local path="$1"
  awk '
    /^eval_loss=/ {
      split($1, a, "=");
      split($2, b, "=");
      printf "eval_loss=%s ppl=%s", a[2], b[2];
      exit;
    }
  ' "$path"
}

skipgram_summary() {
  local path="$1"
  awk '
    /^king:/ {
      line = $0;
      sub(/^king: /, "", line);
      print "king -> " line;
      exit;
    }
  ' "$path"
}

print_summary_row() {
  local model="$1"
  local metric="$2"
  local artifact="$3"
  printf "%-12s  %-34s  %s\n" "$model" "$metric" "$artifact"
}

log "running autoencoder"
./autoencoder \
  --epochs=20 --batch=32 --hidden=256 --latent=64 --opt=adam --loss=bce --lr=0.001 \
  --log=out/bigdemo_autoencoder_log.csv \
  --recon=out/bigdemo_autoencoder_recon.csv
"$PYTHON_BIN" scripts/render_autoencoder_recon.py \
  out/bigdemo_autoencoder_recon.csv out/bigdemo_autoencoder_recon.png

log "running seq2seq"
./seq2seq \
  --steps=3000 --batch=32 --embed=16 --hidden=32 --min-len=3 --max-len=8 \
  --attention=1 --opt=adam --lr=0.003 \
  --log=out/bigdemo_seq2seq_log.csv \
  --report=out/bigdemo_seq2seq_report.txt

log "running skipgram"
./skipgram \
  --steps=3000 --batch=64 --window=2 --lr=0.05 --embed=32 --vocab=800 \
  --snapshot=out/bigdemo_skipgram_snapshot.bin \
  --vocab-out=out/bigdemo_skipgram_vocab.txt \
  --report=out/bigdemo_skipgram_report.txt

log "running ngram"
./ngram \
  --steps=3000 --batch=64 --context=3 --lr=0.03 --embed=32 --hidden=64 --vocab=1000 \
  --snapshot=out/bigdemo_ngram_snapshot.bin \
  --vocab-out=out/bigdemo_ngram_vocab.txt \
  --report=out/bigdemo_ngram_report.txt

log "running mnist conv"
./mnist_conv_demo \
  --epochs=10 --batch=32 --channels=32 --opt=adam --lr=0.001 \
  --log=out/bigdemo_conv_log.csv \
  --snapshot=out/bigdemo_conv_snapshot.bin
"$PYTHON_BIN" scripts/plot_training_loss.py \
  out/bigdemo_conv_log.csv out/bigdemo_conv_loss.png

log "running resnet"
./resnet_demo \
  --epochs=10 --batch=128 --dim=32 --hidden=64 --opt=adam --lr=0.001 \
  --train-limit=2000 --test-limit=1000 \
  --log=out/bigdemo_resnet_log.csv \
  --snapshot=out/bigdemo_resnet_snapshot.bin
"$PYTHON_BIN" scripts/plot_training_loss.py \
  out/bigdemo_resnet_log.csv out/bigdemo_resnet_loss.png

auto_eval_loss="$(csv_last_field out/bigdemo_autoencoder_log.csv 3)"
seq_eval_tok="$(csv_last_field out/bigdemo_seq2seq_log.csv 6)"
seq_eval_seq="$(csv_last_field out/bigdemo_seq2seq_log.csv 7)"
conv_test_acc="$(csv_last_field out/bigdemo_conv_log.csv 5)"
resnet_test_acc="$(csv_last_field out/bigdemo_resnet_log.csv 5)"
ngram_metric="$(ngram_eval_summary out/bigdemo_ngram_report.txt)"
skipgram_metric="$(skipgram_summary out/bigdemo_skipgram_report.txt)"

printf '\n%s\n' "bigdemo summary"
printf "%-12s  %-34s  %s\n" "model" "headline" "artifact"
printf "%-12s  %-34s  %s\n" "------------" "----------------------------------" "------------------------------"
print_summary_row "autoencoder" "eval_loss=${auto_eval_loss}" "out/bigdemo_autoencoder_recon.png"
print_summary_row "seq2seq" "eval_tok=${seq_eval_tok} eval_seq=${seq_eval_seq}" "out/bigdemo_seq2seq_report.txt"
print_summary_row "skipgram" "${skipgram_metric}" "out/bigdemo_skipgram_report.txt"
print_summary_row "ngram" "${ngram_metric}" "out/bigdemo_ngram_report.txt"
print_summary_row "mnist_conv" "test_acc=${conv_test_acc}" "out/bigdemo_conv_loss.png"
print_summary_row "resnet" "test_acc=${resnet_test_acc}" "out/bigdemo_resnet_loss.png"
