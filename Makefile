CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -pedantic
AR ?= ar
ARFLAGS ?= rcs

MNIST_DIR ?= data/mnist
PYTHON ?= .venv/bin/python3
LOG ?= out/mnist_training_log.csv
OUT ?= out/mnist_training_loss.png

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

run-mnist-conv: mnist-conv
	./mnist_conv_demo

mnist-data:
	./scripts/setup_mnist.sh $(MNIST_DIR)

plot-loss:
	$(PYTHON) scripts/plot_training_loss.py $(LOG) $(OUT)

rebuild:
	$(MAKE) clean
	$(MAKE) all llm skipgram ngram mnist-conv tensor_test

run-shakespeare:
	./scripts/run_shakespeare_llm.sh

clean:
	rm -f demo llm skipgram ngram mnist_conv_demo tensor_test $(OBJ) $(LIB) tensor_test_single.bin tensor_test_snapshot.bin

.PHONY: all static test run run-llm run-skipgram run-ngram run-mnist-conv mnist-data plot-loss rebuild run-shakespeare clean
