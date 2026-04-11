CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -pedantic
AR ?= ar
ARFLAGS ?= rcs

MNIST_DIR ?= data/mnist
PYTHON ?= .venv/bin/python3
LOG ?= out/mnist_training_log.csv
OUT ?= out/mnist_training_loss.png
RECON ?= out/autoencoder_recon.csv
RECON_OUT ?= out/autoencoder_recon.png

LIB := libtensor.a
OBJ := tensor.o

all: demo

static: $(LIB)

$(OBJ): tensor.c tensor.h
	$(CC) $(CFLAGS) -c tensor.c -o $(OBJ)

$(LIB): $(OBJ)
	$(AR) $(ARFLAGS) $(LIB) $(OBJ)

demo: demo.c tensor.h $(LIB)
	$(CC) $(CFLAGS) demo.c $(LIB) -lm -o demo

llm: llm.c tensor.h $(LIB)
	$(CC) $(CFLAGS) llm.c $(LIB) -lm -o llm

skipgram: skipgram.c vocab.c vocab.h tensor.h $(LIB)
	$(CC) $(CFLAGS) skipgram.c vocab.c $(LIB) -lm -o skipgram

ngram: ngram.c vocab.c vocab.h tensor.h $(LIB)
	$(CC) $(CFLAGS) ngram.c vocab.c $(LIB) -lm -o ngram

seq2seq: seq2seq.c tensor.h $(LIB)
	$(CC) $(CFLAGS) seq2seq.c $(LIB) -lm -o seq2seq

autoencoder: autoencoder.c tensor.h $(LIB)
	$(CC) $(CFLAGS) autoencoder.c $(LIB) -lm -o autoencoder

mnist-conv: mnist.c tensor.h $(LIB)
	$(CC) $(CFLAGS) mnist.c $(LIB) -lm -o mnist_conv_demo

tensor_test: tensor_test.c tensor.h $(LIB)
	$(CC) $(CFLAGS) tensor_test.c $(LIB) -lm -o tensor_test

test: tensor_test
	./tensor_test

run: demo
	./demo

run-llm: llm
	./llm

run-skipgram: skipgram
	./skipgram

run-ngram: ngram
	./ngram

run-seq2seq: seq2seq
	./seq2seq

run-autoencoder: autoencoder
	./autoencoder

run-mnist-conv: mnist-conv
	./mnist_conv_demo

mnist-data:
	./scripts/setup_mnist.sh $(MNIST_DIR)

plot-loss:
	$(PYTHON) scripts/plot_training_loss.py $(LOG) $(OUT)

plot-autoencoder:
	$(PYTHON) scripts/render_autoencoder_recon.py $(RECON) $(RECON_OUT)

rebuild:
	$(MAKE) clean
	$(MAKE) all llm skipgram ngram seq2seq autoencoder mnist-conv tensor_test

run-shakespeare:
	./scripts/run_shakespeare_llm.sh

clean:
	rm -f demo llm skipgram ngram seq2seq autoencoder mnist_conv_demo tensor_test $(OBJ) $(LIB) tensor_test_single.bin tensor_test_snapshot.bin

.PHONY: all static test run run-llm run-skipgram run-ngram run-seq2seq run-autoencoder run-mnist-conv mnist-data plot-loss plot-autoencoder rebuild run-shakespeare clean
