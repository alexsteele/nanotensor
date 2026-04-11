# LLM Demo

`llm.c` is a minimal character-level language model demo built on nanotensor.

It trains a small MLP to predict the next character from a text corpus and then
generates text from a prompt.

Build and run:

```bash
make llm
./llm shakespeare.txt 3000 "To be" 64 300 0.2 0.9 96
```

To download and prepare the Shakespeare corpus automatically and run the demo:

```bash
make run-shakespeare
```

Or run the helper script directly:

```bash
./scripts/run_shakespeare_llm.sh [steps] [prompt] [batch] [gen_len] [lr] [temperature] [hidden]
```
