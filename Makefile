CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -pedantic

all: demo

demo: tensor.c tensor.h main.c
	$(CC) $(CFLAGS) tensor.c main.c -lm -o demo

run: demo
	./demo

clean:
	rm -f demo

.PHONY: all run clean
