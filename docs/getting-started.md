# Getting Started

This page describes the basic path from a repository checkout to generated
PyTorch-Chipyard artifacts. Full paper replication follows the artifact
replication flow in the README. This page focuses on installing the compiler
stack and compiling a first model.

## Repository Layout

```text
pytorch-chipyard/
  examples/             # PyTorch model entry points
  scripts/              # install, environment, artifact generation scripts
  triton_chipyard/      # out-of-tree Triton backend
  pytorch/              # custom PyTorch checkout
  triton/               # custom Triton checkout
  llvm-project/         # matching LLVM/MLIR checkout
  buddy-mlir/           # matching Buddy-MLIR checkout
  chipyard/             # Chipyard, FireMarshal, FireSim tree
  docs/                 # Sphinx documentation
```

This project is version-sensitive. The documented path assumes the custom
versions of `pytorch`, `triton`, `triton_chipyard`, `llvm-project`,
`buddy-mlir`, and `chipyard` pinned by this repository. Other commits may work,
but they are outside the documented configuration.

## Installation

Follow the installation sequence from the README:

```bash
# pre-requisite: conda-24.11.3
git clone https://github.com/JongseoKang/pytorch-chipyard

cd pytorch-chipyard
git submodule update --init pytorch triton triton_chipyard llvm-project buddy-mlir chipyard

cd chipyard
./build-setup.sh riscv-tools
cd ..

bash scripts/install.sh
```

`scripts/install.sh` performs the following work:

- Creates a `pytorch-chipyard` conda environment and installs Python 3.12 based
  dependencies.
- Installs PyTorch 2.8.0, torchvision 0.23.0, Triton build dependencies, and
  Buddy-MLIR dependencies.
- Builds `llvm-project` with MLIR, LLVM, LLD, and the RISC-V target enabled.
- Installs the custom Triton checkout in editable mode with
  `TRITON_PLUGIN_DIRS=$WORKSPACE/triton_chipyard`.
- Builds `buddy-mlir` against the custom LLVM/MLIR build.
- Replaces the installed PyTorch package's `torch/_inductor` directory with the
  custom `pytorch/torch/_inductor` directory from this repository.

This repository includes the compiler stack and the Chipyard tree, but it does
not automatically configure the FPGA host system. FireSim/FPGA replication
requires the host to provide:

- An AMD/Xilinx Alveo U250-class host. Some experiments may run on a U280 with a
  matching bitstream, but the full paper configuration assumes U250.
- An Ubuntu 20.04-class host environment.
- XRT, the XDMA kernel module, and Vivado/Vitis-compatible tooling.
- FireSim database files and access to the target FPGA device.
- Paper-replication prebuilt bitstreams, or bitstreams built from the matching
  Chipyard/FireSim hardware configuration.

## Environment

Most commands start by sourcing `scripts/env.sh` from the repository root:

```bash
cd pytorch-chipyard
source scripts/env.sh
```

`scripts/env.sh` defines these path variables:

| Variable | Meaning |
| --- | --- |
| `WORKSPACE` | Repository root. |
| `BUDDY_BINARY_DIR` | Buddy-MLIR build binary directory. The default is `$WORKSPACE/buddy-mlir/build/bin`. |
| `CHIPYARD_ENV_PATH` | Chipyard environment script. The default is `$WORKSPACE/chipyard/env.sh`. |
| `TRITON_CHIPYARD_OPT_PATH` | Triton-Chipyard MLIR pass driver binary. |
| `LLVM_PROJECT_PATH` | Custom LLVM checkout path. |
| `CHIPYARD_SIM_VERILATOR_PATH` | Simulator binary path for running a standalone Triton-Chipyard kernel through Verilator. The default is empty. When it is empty, direct standalone kernel launches compile but skip simulator execution. |

TorchInductor and artifact-generation variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TORCHINDUCTOR_FORCE_DISABLE_CACHES` | `1` | Disables Inductor cache reuse so artifacts are regenerated for the current input and environment. |
| `TORCHINDUCTOR_MAX_AUTOTUNE` | `1` | Enables Inductor template autotune candidate generation. |
| `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` | `TRITON` | Restricts GEMM autotune backends to Triton. |
| `TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS` | `TRITON` | Restricts convolution autotune backends to Triton. |
| `TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER` | `1` | Enables generation of `runner.cpp`, `model_spec.json`, `weights.bin`, and `util.py`. |
| `TORCHINDUCTOR_STAGE_CHIPYARD_KERNEL_ARTIFACTS` | Set to `1` by scripts | Stages kernel objects and metadata into the generated runner artifact. |
| `TORCHINDUCTOR_COMPILE_CHIPYARD_MODEL_RUNNER` | Direct examples use `0` or leave it unset | If set to `1`, compilation also attempts to build `model.elf`. The documented direct flow keeps this disabled and calls the generated artifact `build.sh` explicitly after compilation. |
| `TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE` | `0` | If set to `1`, generates a larger Gemmini tiling search space. This can make simulation much slower. |
| `TORCHINDUCTOR_IM2COL_MM` | unset | Set to `1` by `scripts/run-im2col.sh` for convolution-to-im2col+matmul lowering experiments. |

Gemmini target variables:

| Variable | Default |
| --- | --- |
| `TRITON_CHIPYARD_USE_GEMMINI` | `1` |
| `TRITON_CHIPYARD_GEMMINI_ADDR_LEN` | `32` |
| `TRITON_CHIPYARD_GEMMINI_DIM` | `8` |
| `TRITON_CHIPYARD_GEMMINI_BANK_ROWS` | `2048` |
| `TRITON_CHIPYARD_GEMMINI_ACC_ROWS` | `2048` |
| `TRITON_CHIPYARD_GEMMINI_ELEM_T` | `f32` |
| `TRITON_CHIPYARD_GEMMINI_ACC_T` | `f32` |
| `TRITON_CHIPYARD_RISCV_MARCH` | `rv64imafdc` |
| `TRITON_CHIPYARD_RISCV_MABI` | `lp64d` |

To use the RVV target, disable Gemmini and set the RVV variables below. Stage
scripts such as `scripts/run-cnn.sh` set these automatically when
`--backend=rvv` is selected.

| Variable | RVV value |
| --- | --- |
| `TRITON_CHIPYARD_USE_GEMMINI` | `0` |
| `TRITON_CHIPYARD_USE_RVV` | `1` |
| `TRITON_CHIPYARD_RISCV_MARCH` | `rv64imafdcv_zicsr_zifencei_zvl128b` |
| `TRITON_CHIPYARD_RISCV_MABI` | `lp64d` |
| `TRITON_CHIPYARD_RISCV_VARCH` | `vlen:128,elen:64` |
| `TRITON_CHIPYARD_SPIKE_ISA` | `rv64gcv` |
| `TRITON_CHIPYARD_SPIKE_VARCH` | `vlen:128,elen:64` |

Common per-run variables:

| Variable | Meaning |
| --- | --- |
| `PYTORCH_CHIPYARD_DUMP_PATH` | Model-level artifact output directory. |
| `TRITON_CHIPYARD_DUMP_PATH` | Lower-level Triton/Triton-Chipyard IR dump directory. This is usually set to the artifact directory. |
| `TRITON_CACHE_DIR` | Triton compile cache directory. The scripts use `/tmp/triton-chipyard-cache/<key>`. |
| `TRITON_ALWAYS_COMPILE` | Set to `1` by scripts to prefer recompilation over cached launcher reuse. |
| `CHIPYARD_OMP_NUM_THREADS` | Core/thread label used when generated `build.sh` builds the runner. |
| `LLM_TOKEN_LENGTH` | Sequence length for LLM examples. |
| `HF_TOKEN` | Token used for gated Hugging Face models when required. |
| `TRITON_CHIPYARD_PERF_OPS` | If it includes `matmul`, enables the matmul performance-counter path. |
| `TRITON_CHIPYARD_PACKET_TMPDIR` | Temporary directory for standalone kernel launch packets. The preferred default is `/dev/shm`. |

`scripts/artifact-stage1-common.sh` also reads these convenience variables:

| Variable | Meaning |
| --- | --- |
| `CONDA_ENV_NAME` or `CONDA_ENV` | Conda environment activated by stage scripts. The fallback default is `llapi`. |
| `PYTORCH_CHIPYARD_SKIP_CONDA` | If set to `1`, skips conda activation in stage scripts. |
| `CNN_BATCH_SIZE`, `CNN_SEED` | Default batch size and seed for CNN scripts. |
| `LLM_BATCH_SIZE`, `LLM_SEED` | Default batch size and seed for LLM scripts. |

## First Artifact

The direct PyTorch-Chipyard path is an environment-variable setup followed by a
normal Python example entry point. Do not use the stage wrapper scripts for this
minimal flow.

```bash
cd pytorch-chipyard
source scripts/env.sh

export TRITON_ALWAYS_COMPILE=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON
export TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS=TRITON
export TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER=1
export TORCHINDUCTOR_STAGE_CHIPYARD_KERNEL_ARTIFACTS=1
export TORCHINDUCTOR_COMPILE_CHIPYARD_MODEL_RUNNER=0

export TRITON_CHIPYARD_USE_GEMMINI=1
export TRITON_CHIPYARD_USE_RVV=0
export TRITON_CHIPYARD_GEMMINI_ADDR_LEN=32
export TRITON_CHIPYARD_GEMMINI_DIM=8
export TRITON_CHIPYARD_GEMMINI_BANK_ROWS=2048
export TRITON_CHIPYARD_GEMMINI_ACC_ROWS=2048
export TRITON_CHIPYARD_GEMMINI_ELEM_T=f32
export TRITON_CHIPYARD_GEMMINI_ACC_T=f32
export TRITON_CHIPYARD_RISCV_MARCH=rv64imafdc
export TRITON_CHIPYARD_RISCV_MABI=lp64d

export PYTORCH_CHIPYARD_DUMP_PATH=$PWD/examples/resnet50/gemmini
export TRITON_CHIPYARD_DUMP_PATH=$PYTORCH_CHIPYARD_DUMP_PATH
export TRITON_CACHE_DIR=/tmp/triton-chipyard-cache/resnet50-gemmini

python examples/ResNet50.py --compile --batch-size 1 --seed 0
```

Compilation writes the artifact files, but it does not build `model.elf` in this
direct flow. Build the runner explicitly from the generated artifact directory:

```bash
cd "$PYTORCH_CHIPYARD_DUMP_PATH"
CHIPYARD_OMP_NUM_THREADS=4 bash ./build.sh
cp -f model.elf model-4core.elf
```

The output directory is:

```text
examples/resnet50/gemmini/
```

A successful artifact directory usually contains:

```text
runner.cpp
model_spec.json
weights.bin
weights.manifest.json
input.bin
output.bin
build.sh
util.py
model.elf
model-4core.elf
```

When `TRITON_CHIPYARD_DUMP_PATH` is set, debug files such as `tt.mlir`,
`ttshared.mlir`, `temp.mlir`, `gemmini-lowering-info.json`, and lowering logs may
also appear in the same directory.

The runner reads `input.bin` and `weights.bin`, then writes `output.bin`,
`model.log`, and `autotune.log` in its current working directory. FPGA setup and
execution are outside the coverage of this work: prepare and run the FPGA system
yourself, then copy the resulting files back into the artifact directory for
validation. After target execution fills `output.bin`, compare it with eager
PyTorch using the matching example entry point:

```bash
PYTORCH_CHIPYARD_DUMP_PATH=$PWD/examples/resnet50/gemmini \
  python examples/ResNet50.py --validate
```
