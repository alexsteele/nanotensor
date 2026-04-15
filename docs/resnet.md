# ResNet

`resnet.c` is a small residual patch-network demo for MNIST.

It is not a full spatial CNN ResNet. Instead, it keeps the current 2D tensor
setup and applies residual MLP blocks over extracted image patches.

## Architecture

The model currently does:

- input images: `[batch, 28 * 28]`
- extract `5x5` sliding patches with the shared `patch.*` helpers
- patch stem: `25 -> dim` with `relu`
- two residual blocks:
  `x + linear(relu(linear(layernorm(x))))`
- mean-pool patch features back to one row per image
- classifier head: `dim -> 10`

That makes it a useful bridge between the existing patch-conv demo and a more
recognizable ResNet-style residual architecture.

## CLI Knobs

The main options are:

- `--epochs=N`
- `--batch=N`
- `--dim=N`
- `--hidden=N`
- `--opt=sgd|momentum|adam`
- `--lr=FLOAT`
- `--momentum=FLOAT`
- `--train-limit=N`
- `--test-limit=N`
- `--log=PATH`

Defaults in the current demo are:

- `epochs=5`
- `batch=32`
- `dim=32`
- `hidden=64`
- `opt=adam`
- `lr=0.001`
- `train-limit=2000`
- `test-limit=1000`

## Example Run

Build and run:

```bash
make resnet
./resnet_demo \
  --epochs=5 \
  --batch=256 \
  --dim=16 \
  --hidden=32 \
  --opt=adam \
  --lr=0.001 \
  --train-limit=512 \
  --test-limit=256 \
  --log=out/resnet_example_log.csv
```

Plot the log:

```bash
./.venv/bin/python3 scripts/plot_training_loss.py \
  out/resnet_example_log.csv \
  out/resnet_example_loss.png
```

## Artifact

- plot:
  [examples/resnet_example_loss.png](/Users/alex/Code/nanotensor/examples/resnet_example_loss.png)
- metrics:
  [examples/resnet_example_log.csv](/Users/alex/Code/nanotensor/examples/resnet_example_log.csv)

## Experiment Log

### first residual patch baseline

- config:
  `epochs=5 batch=256 dim=16 hidden=32 opt=adam lr=0.001 train_limit=512 test_limit=256`
- final result:
  `train_loss=2.294517 train_acc=0.181641 test_acc=0.156250`
- notes:
  this confirms the residual patch stack trains end to end and performs
  slightly above chance on a very small subset, but it is still a weak baseline.
  The next likely gains are a wider patch stem, a larger training subset, or a
  better pooling/classification strategy.
