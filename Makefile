CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -pedantic
AR ?= ar
ARFLAGS ?= rcs
BLAS ?= none

BLAS_CFLAGS :=
BLAS_LDFLAGS :=

ifeq ($(BLAS),accelerate)
BLAS_CFLAGS += -DTENSOR_USE_BLAS -DTENSOR_BLAS_ACCELERATE -DACCELERATE_NEW_LAPACK
BLAS_LDFLAGS += -framework Accelerate
else ifeq ($(BLAS),openblas)
BLAS_CFLAGS += -DTENSOR_USE_BLAS -DTENSOR_BLAS_OPENBLAS
BLAS_LDFLAGS += -lopenblas
else ifeq ($(BLAS),none)
else
$(error Unsupported BLAS backend '$(BLAS)'; use BLAS=none, accelerate, or openblas)
endif

MNIST_DIR ?= data/mnist
PYTHON ?= .venv/bin/python3
LOG ?= out/mnist_training_log.csv
OUT ?= out/mnist_training_loss.png
RECON ?= out/autoencoder_recon.csv
RECON_OUT ?= out/autoencoder_recon.png

LIB := libtensor.a
OBJ := tensor.o
BLAS_MARKER := .blas_backend_$(BLAS)
BLAS_MARKERS := .blas_backend_none .blas_backend_accelerate .blas_backend_openblas

all: demo

static: $(LIB)

$(BLAS_MARKER):
	rm -f $(BLAS_MARKERS)
	printf "%s\n" "$(BLAS)" > $(BLAS_MARKER)

$(OBJ): tensor.c tensor.h Makefile $(BLAS_MARKER)
	$(CC) $(CFLAGS) $(BLAS_CFLAGS) -c tensor.c -o $(OBJ)

$(LIB): $(OBJ)
	$(AR) $(ARFLAGS) $(LIB) $(OBJ)

demo: demo.c tensor.h $(LIB)
	$(CC) $(CFLAGS) demo.c $(LIB) $(BLAS_LDFLAGS) -lm -o demo

llm: llm.c tensor.h $(LIB)
	$(CC) $(CFLAGS) llm.c $(LIB) $(BLAS_LDFLAGS) -lm -o llm

gpt-char: gpt_char.c tensor.h $(LIB)
	$(CC) $(CFLAGS) gpt_char.c $(LIB) $(BLAS_LDFLAGS) -lm -o gpt_char

bench: tensor_bench.c tensor.h $(LIB)
	$(CC) $(CFLAGS) tensor_bench.c $(LIB) $(BLAS_LDFLAGS) -lm -o tensor_bench

skipgram: skipgram.c vocab.c vocab.h tensor.h $(LIB)
	$(CC) $(CFLAGS) skipgram.c vocab.c $(LIB) $(BLAS_LDFLAGS) -lm -o skipgram

ngram: ngram.c vocab.c vocab.h tensor.h $(LIB)
	$(CC) $(CFLAGS) ngram.c vocab.c $(LIB) $(BLAS_LDFLAGS) -lm -o ngram

seq2seq: seq2seq.c tensor.h $(LIB)
	$(CC) $(CFLAGS) seq2seq.c $(LIB) $(BLAS_LDFLAGS) -lm -o seq2seq

autoencoder: autoencoder.c mnist.c mnist.h tensor.h $(LIB)
	$(CC) $(CFLAGS) autoencoder.c mnist.c $(LIB) $(BLAS_LDFLAGS) -lm -o autoencoder

mnist-conv: convnet.c mnist.c mnist.h patch.c patch.h tensor.h $(LIB)
	$(CC) $(CFLAGS) convnet.c mnist.c patch.c $(LIB) $(BLAS_LDFLAGS) -lm -o mnist_conv_demo

mnist-resnet: mnist_resnet.c mnist.c mnist.h patch.c patch.h tensor.h $(LIB)
	$(CC) $(CFLAGS) mnist_resnet.c mnist.c patch.c $(LIB) $(BLAS_LDFLAGS) -lm -o mnist_resnet_demo

tensor_test: tensor_test.c tensor.h $(LIB)
	$(CC) $(CFLAGS) tensor_test.c $(LIB) $(BLAS_LDFLAGS) -lm -o tensor_test

test: tensor_test
	./tensor_test

run: demo
	./demo

run-llm: llm
	./llm

run-gpt-char: gpt-char
	./gpt_char

run-bench: bench
	./tensor_bench

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

run-mnist-resnet: mnist-resnet
	./mnist_resnet_demo

mnist-data:
	./scripts/setup_mnist.sh $(MNIST_DIR)

plot-loss:
	$(PYTHON) scripts/plot_training_loss.py $(LOG) $(OUT)

plot-autoencoder:
	$(PYTHON) scripts/render_autoencoder_recon.py $(RECON) $(RECON_OUT)

rebuild:
	$(MAKE) clean
	$(MAKE) all llm gpt-char bench skipgram ngram seq2seq autoencoder mnist-conv mnist-resnet tensor_test

run-shakespeare:
	./scripts/run_shakespeare_llm.sh

examples: autoencoder seq2seq skipgram ngram mnist-conv mnist-resnet
	PYTHON=$(PYTHON) ./scripts/run_examples.sh

clean:
	rm -f demo llm gpt_char tensor_bench skipgram ngram seq2seq autoencoder mnist_conv_demo mnist_resnet_demo tensor_test $(OBJ) $(LIB) tensor_test_single.bin tensor_test_snapshot.bin

.PHONY: all static test run run-llm run-gpt-char run-bench run-skipgram run-ngram run-seq2seq run-autoencoder run-mnist-conv run-mnist-resnet mnist-data plot-loss plot-autoencoder rebuild run-shakespeare examples clean
