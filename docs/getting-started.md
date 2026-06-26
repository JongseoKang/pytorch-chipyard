# Getting Started

This page describes the shortest path from a repository checkout to a generated
PyTorch-Chipyard artifact directory.

## Repository Layout

The repository is organized around a top-level PyTorch model path and a
lower-level Triton-Chipyard backend path:

```text
pytorch-chipyard/
  examples/             # PyTorch model examples
  triton_chipyard/      # Triton-Chipyard backend, tests, and scripts
  pytorch/              # matching PyTorch checkout
  triton/               # matching Triton checkout
  llvm-project/         # matching LLVM checkout
  buddy-mlir/           # matching Buddy-MLIR checkout
```

The documentation site itself lives under `docs/`.

## Checkout

Clone the repository and initialize the submodules used by the compiler stack:

```sh
git clone --recurse-submodules <pytorch-chipyard-repo-url> pytorch-chipyard
cd pytorch-chipyard
git submodule update --init --recursive
```

If the submodules are already present, confirm that the local branches match the
PyTorch-Chipyard branches expected by this repository.

## Environment

The backend setup is currently controlled by `triton_chipyard/env.sh`. Source it
before running examples:

```sh
cd pytorch-chipyard
source triton_chipyard/env.sh
```

Before public release, check this file for local machine paths. In particular,
`LLAPI_ROOT`, `BUDDY_BINARY_DIR`, `CHIPYARD_ENV_PATH`,
`TRITON_CHIPYARD_OPT_PATH`, and `LLVM_PROJECT_PATH` must match the checkout and
build layout on the machine running the examples.

The important runtime variables are:

- `PYTORCH_CHIPYARD_DUMP_PATH`: output directory for model-level artifacts.
- `TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER`: enables generated runner artifacts.
- `TRITON_CHIPYARD_DUMP_PATH`: optional lower-level Triton-Chipyard IR dump path.
- `TRITON_CACHE_DIR`: Triton compilation cache location.

## Install Triton With the Backend

Install the matching Triton checkout with the out-of-tree backend visible to
Triton:

```sh
cd pytorch-chipyard
export TRITON_PLUGIN_DIRS=$PWD/triton_chipyard
cd triton
pip install -e .
```

Then return to the repository root:

```sh
cd ..
```

## First Example

The smallest model-level path is to compile one vision example:

```sh
cd pytorch-chipyard
source triton_chipyard/env.sh
python examples/AlexNet.py --compile
```

By default, the example writes artifacts under `IR/alexnet`. To choose an
explicit location:

```sh
export PYTORCH_CHIPYARD_DUMP_PATH=$PWD/IR/alexnet
python examples/AlexNet.py --compile
```

A successful artifact directory should contain files such as:

```text
runner.cpp
model_spec.json
weights.bin
weights.manifest.json
input.bin
output.bin
build.sh
util.py
```

Some runs may also produce lower-level inspection files such as `tt.mlir`,
`ttshared.mlir`, `temp.mlir`, `gemmini-lowering-info.json`, and lowering logs.

## Validation

After an external Chipyard or target execution path writes `output.bin`, the
corresponding PyTorch example can compare it against eager PyTorch:

```sh
python examples/AlexNet.py --validate
```

Validation expects the artifact directory and generated `util.py` from the
matching compile step.
