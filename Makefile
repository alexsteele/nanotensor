CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -pedantic
AR ?= ar
ARFLAGS ?= rcs

LIB := libtensor.a
OBJ := tensor.o

all: demo

static: $(LIB)

$(OBJ): tensor.c tensor.h
	$(CC) $(CFLAGS) -c tensor.c -o $(OBJ)

$(LIB): $(OBJ)
	$(AR) $(ARFLAGS) $(LIB) $(OBJ)

demo: main.c tensor.h $(LIB)
	$(CC) $(CFLAGS) main.c $(LIB) -lm -o demo

llm: llm.c tensor.h $(LIB)
	$(CC) $(CFLAGS) llm.c $(LIB) -lm -o llm

tensor_test: tensor_test.c tensor.h $(LIB)
	$(CC) $(CFLAGS) tensor_test.c $(LIB) -lm -o tensor_test

test: tensor_test
	./tensor_test

run: demo
	./demo

run-llm: llm
	./llm

run-shakespeare:
	./scripts/run_shakespeare_llm.sh

clean:
	rm -f demo llm tensor_test $(OBJ) $(LIB) tensor_test_single.bin tensor_test_snapshot.bin

.PHONY: all static test run run-llm run-shakespeare clean
