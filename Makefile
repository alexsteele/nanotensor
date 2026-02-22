CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -pedantic

all: demo

demo: tensor.c tensor.h main.c
	$(CC) $(CFLAGS) tensor.c main.c -lm -o demo

tensor_test: tensor.c tensor.h tensor_test.c
	$(CC) $(CFLAGS) tensor.c tensor_test.c -lm -o tensor_test

test: tensor_test
	./tensor_test

run: demo
	./demo

clean:
	rm -f demo tensor_test tensor_test_single.bin tensor_test_snapshot.bin

.PHONY: all test run clean
