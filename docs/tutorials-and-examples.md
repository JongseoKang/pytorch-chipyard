# Section 2: Tutorials

This section contains two tutorials:

- Standalone Triton-Chipyard matmul.
- PyTorch-Chipyard ResNet50 compilation and artifact execution flow.

The examples call the Python entry points directly where that is the clearest
path. The later packaging and FireSim steps use the repository scripts that
generate FireMarshal and FireSim workload files.

## 2.1 Triton-Chipyard Matmul

The main goal of
[pytorch-chipyard](https://github.com/JongseoKang/pytorch-chipyard) is PyTorch
model execution, but Triton-Chipyard can also be used as a standalone Triton
backend. The full framework installation installs Triton with the out-of-tree
`triton_chipyard` backend, so the standalone example is available after the
Section 1 installation flow.

To run a standalone Triton kernel, set the Verilator simulator path and run
`triton_chipyard/example/test_matmul.py`. This example includes both the Triton
matmul kernel and the Chipyard driver activation:

```{literalinclude} ../triton_chipyard/example/test_matmul.py
:language: python
:linenos:
:caption: triton_chipyard/example/test_matmul.py
:emphasize-lines: 6,110-112
```

The important Chipyard-specific lines are:

- `from triton.backends.triton_chipyard.driver import ChipyardDriver`
- `triton.runtime.driver.set_active(ChipyardDriver())`

The first line imports the out-of-tree backend driver. The second line makes
Triton route kernel compilation and launch through Triton-Chipyard instead of a
normal GPU backend.

Run the example with:

```bash
cd pytorch-chipyard

# Edit scripts/env.sh if your local checkout or build paths differ.
source scripts/env.sh

export CHIPYARD_SIM_VERILATOR_PATH=/path/to/chipyard/verilator/simulator
export TRITON_CHIPYARD_DUMP_PATH=$PWD/IR/triton-matmul
export TRITON_CACHE_DIR=/tmp/triton-chipyard-cache/triton-matmul

python triton_chipyard/example/test_matmul.py \
  --block-size-m 128 \
  --block-size-n 128 \
  --block-size-k 128
```

`CHIPYARD_SIM_VERILATOR_PATH` is required for this standalone example to execute
the kernel through the Verilator simulator. If the variable is empty,
Triton-Chipyard still compiles the kernel path, but the standalone simulator
launch is skipped.

### Triton-Chipyard Environment Variables

`scripts/env.sh` defines the default environment for both standalone
Triton-Chipyard examples and PyTorch-Chipyard model compilation. Update local
paths in that file if your checkout, LLVM build, Triton build, Buddy-MLIR build,
or Chipyard tree differs from the default repository layout.

Core path variables:

| Variable | Purpose |
| --- | --- |
| `WORKSPACE` | Repository root. Computed from `scripts/env.sh`. |
| `BUDDY_BINARY_DIR` | Buddy-MLIR build binary directory. |
| `TRITON_CHIPYARD_OPT_PATH` | `triton-chipyard-opt` pass driver built with Triton. |
| `LLVM_PROJECT_PATH` | LLVM/MLIR source tree used for runtime headers and support sources. |
| `CHIPYARD_DIR` | Chipyard checkout. |
| `CHIPYARD_ENV_PATH` | Chipyard environment script sourced by generated build paths. |
| `CHIPYARD_SIM_VERILATOR_PATH` | Verilator simulator binary used by standalone kernel execution. |
| `TRITON_CHIPYARD_DUMP_PATH` | Optional dump directory for `tt.mlir`, `ttshared.mlir`, and related lowering files. |
| `TRITON_CACHE_DIR` | Triton compilation cache directory. |

TorchInductor/PyTorch-Chipyard variables:

| Variable | Purpose |
| --- | --- |
| `TORCHINDUCTOR_FORCE_DISABLE_CACHES` | Forces Inductor to regenerate artifacts instead of reusing caches. |
| `TORCHINDUCTOR_MAX_AUTOTUNE` | Enables Inductor template candidate generation. |
| `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` | Restricts GEMM autotune backends. The documented value is `TRITON`. |
| `TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS` | Restricts convolution autotune backends. The documented value is `TRITON`. |
| `TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER` | Enables model artifact generation, including `runner.cpp` and `model_spec.json`. |
| `TORCHINDUCTOR_STAGE_CHIPYARD_KERNEL_ARTIFACTS` | Stages generated kernel objects into the model artifact directory. |
| `TORCHINDUCTOR_COMPILE_CHIPYARD_MODEL_RUNNER` | If set to `1`, Inductor tries to call the generated `build.sh`. The direct tutorial leaves it at `0` and calls `build.sh` explicitly. |
| `TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE` | Enables a larger Gemmini tiling search space. This can make simulation much slower. |

Gemmini target variables:

| Variable | Purpose |
| --- | --- |
| `TRITON_CHIPYARD_USE_GEMMINI` | Enables Gemmini lowering when set to `1`. |
| `TRITON_CHIPYARD_USE_RVV` | Should be `0` for the Gemmini target. |
| `TRITON_CHIPYARD_GEMMINI_ADDR_LEN` | Gemmini address length. The default FP32 configuration uses `32`. |
| `TRITON_CHIPYARD_GEMMINI_DIM` | Gemmini systolic array dimension. The default FP32 configuration uses `8`. |
| `TRITON_CHIPYARD_GEMMINI_BANK_ROWS` | Scratchpad bank rows. The default FP32 configuration uses `2048`. |
| `TRITON_CHIPYARD_GEMMINI_ACC_ROWS` | Accumulator rows. The default FP32 configuration uses `2048`. |
| `TRITON_CHIPYARD_GEMMINI_ELEM_T` | Gemmini element type. The documented default is `f32`. |
| `TRITON_CHIPYARD_GEMMINI_ACC_T` | Gemmini accumulator type. The documented default is `f32`. |
| `TRITON_CHIPYARD_RISCV_MARCH` | RISC-V ISA string used by generated builds. Gemmini examples use `rv64imafdc`. |
| `TRITON_CHIPYARD_RISCV_MABI` | RISC-V ABI string. Gemmini examples use `lp64d`. |

Gemmini example:

```bash
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
```

RVV target variables:

| Variable | Purpose |
| --- | --- |
| `TRITON_CHIPYARD_USE_GEMMINI` | Should be `0` for RVV. |
| `TRITON_CHIPYARD_USE_RVV` | Enables RVV lowering when set to `1`. |
| `TRITON_CHIPYARD_RISCV_MARCH` | RVV examples use `rv64imafdcv_zicsr_zifencei_zvl128b`. |
| `TRITON_CHIPYARD_RISCV_MABI` | RVV examples use `lp64d`. |
| `TRITON_CHIPYARD_RISCV_VARCH` | RVV vector architecture string consumed by the backend. The documented value is `vlen:128,elen:64`. |

RVV example:

```bash
export TRITON_CHIPYARD_USE_GEMMINI=0
export TRITON_CHIPYARD_USE_RVV=1
export TRITON_CHIPYARD_RISCV_MARCH=rv64imafdcv_zicsr_zifencei_zvl128b
export TRITON_CHIPYARD_RISCV_MABI=lp64d
export TRITON_CHIPYARD_RISCV_VARCH=vlen:128,elen:64
```

`TRITON_CHIPYARD_RISCV_VARCH` is still used by
`triton_chipyard/backend/compiler.py`.

FireMarshal and FireSim path variables are described in the ResNet50 flow below.

## 2.2 PyTorch-Chipyard ResNet50

The PyTorch-Chipyard path starts from a regular PyTorch model example under
`examples/`. The ResNet50 example configures Triton-Chipyard as the Inductor CPU
backend, runs `torch.compile`, and writes model-level artifacts.

The complete example is included below:

```{literalinclude} ../examples/ResNet50.py
:language: python
:linenos:
:caption: examples/ResNet50.py
```

Important parts of the example:

- `artifact_dir()` reads `PYTORCH_CHIPYARD_DUMP_PATH`, which controls where the
  generated model artifacts are written.
- `configure_triton_chipyard()` sets `ChipyardDriver()` as the active Triton
  driver and sets `inductor_config.cpu_backend = "triton_chipyard"`.
- `configure_artifact_env()` enables Chipyard runner generation.
- `run_compile()` builds the torchvision ResNet50 model, calls `torch.compile`,
  runs the compiled model once, imports the generated `util.py`, and writes
  `input.bin`.
- `run_validate()` reads the generated `input.bin` and target-produced
  `output.bin`, then compares the result against eager PyTorch.

Compile ResNet50 for the default FP32 Gemmini configuration:

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
export TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE=0

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

# Keep non-Gemmini target settings out of this build if the shell was previously
# used for an RVV run.
unset TRITON_CHIPYARD_RISCV_VARCH

export PYTORCH_CHIPYARD_DUMP_PATH=$PWD/examples/resnet50/gemmini
export TRITON_CHIPYARD_DUMP_PATH=$PYTORCH_CHIPYARD_DUMP_PATH
export TRITON_CACHE_DIR=/tmp/triton-chipyard-cache/resnet50-gemmini

python examples/ResNet50.py --compile --batch-size 1 --seed 0
```

The compilation step writes `runner.cpp`, `model_spec.json`, `weights.bin`,
`weights.manifest.json`, `input.bin`, `build.sh`, `util.py`, and staged kernel
objects into `PYTORCH_CHIPYARD_DUMP_PATH`.

`build.sh` is generated but is not called automatically in the command sequence
above. Build the RISC-V executable explicitly:

```bash
cd "$PYTORCH_CHIPYARD_DUMP_PATH"
CHIPYARD_OMP_NUM_THREADS=4 bash ./build.sh
cp -f model.elf model-4core.elf
```

The generated executable expects input, weight, and output file paths:

```bash
./model.elf input.bin weights.bin output.bin
```

### FireMarshal Packaging and FireSim Execution

The repository's `scripts/` directory contains the Stage 2 workflow that packages
generated artifacts into FireMarshal workloads, builds FireMarshal images, runs
FireSim workloads, and collects result files.

The relevant environment variables are set by `scripts/env.sh`:

| Variable | Purpose |
| --- | --- |
| `FIREMARSHAL_DIR` | FireMarshal checkout under `chipyard/software/firemarshal`. |
| `FIREMARSHAL_IMAGE_DIR` | FireMarshal output image directory. |
| `FIRESIM_DIR` | FireSim checkout under `chipyard/sims/firesim`. |
| `FIRESIM_DEPLOY_DIR` | FireSim deploy directory. |
| `FIRESIM_HWDB_PATH` | FireSim hardware database YAML. |
| `FIRESIM_BUILD_RECIPES_PATH` | FireSim build recipes YAML. |
| `FIRESIM_WORKLOAD_DIR` | FireSim deploy workload JSON directory. |
| `PYTORCH_CHIPYARD_WORKLOAD_DIR` | Generated PyTorch-Chipyard FireMarshal workload directory. |
| `PYTORCH_CHIPYARD_RESULTS_WORKLOAD_DIR` | FireSim `results-workload` directory. |
| `PYTORCH_CHIPYARD_FPGA_DB` | FPGA database path. Default: `/opt/firesim-db.json`. |
| `PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR` | Directory for generated FireSim runtime YAML files. |
| `PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR` | Timestamp-free collected result directory used by figure scripts. |

Package the generated `model-<N>core.elf` files into FireMarshal and FireSim
workload descriptions:

```bash
cd pytorch-chipyard
source scripts/env.sh

bash scripts/package-firemarshal-workload.sh --artifact-root=$PWD/examples
```

For the ResNet50 Gemmini artifact above, this discovers
`examples/resnet50/gemmini/model-4core.elf` and writes:

```text
$PYTORCH_CHIPYARD_WORKLOAD_DIR/resnet50-gemmini-4core.json
$PYTORCH_CHIPYARD_WORKLOAD_DIR/overlay-resnet50-gemmini-4core/
$FIRESIM_WORKLOAD_DIR/resnet50-gemmini-4core.json
```

The generated overlay contains the model ELF, `input.bin`, `weights.bin`, and a
guest runner script. The guest runner executes:

```bash
./model-4core.elf input.bin weights.bin output.bin
```

It also configures OpenMP variables such as `OMP_NUM_THREADS`,
`OMP_THREAD_LIMIT`, `OMP_PLACES`, and `GOMP_CPU_AFFINITY` based on the core count
encoded in the ELF filename.

Build and install the FireMarshal images:

```bash
bash scripts/build-firemarshal-images.sh
```

This runs `./marshal build` and `./marshal install` for the generated workload
JSON files. FireMarshal image construction may require `sudo` because overlays
are applied through mounted root filesystems.

The expected image outputs are:

```text
$FIREMARSHAL_IMAGE_DIR/resnet50-gemmini-4core/resnet50-gemmini-4core-bin
$FIREMARSHAL_IMAGE_DIR/resnet50-gemmini-4core/resnet50-gemmini-4core.img
$FIRESIM_WORKLOAD_DIR/resnet50-gemmini-4core.json
```

Run the workload through FireSim:

```bash
bash scripts/run-firesim-workloads.sh --workload=resnet50-gemmini-4core
```

`scripts/run-firesim-workloads.sh` generates a per-workload runtime YAML under
`$PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR`, infers the hardware config from the
workload name, and runs:

```bash
firesim launchrunfarm
firesim infrasetup
firesim runworkload
firesim terminaterunfarm
```

If the inferred hardware config is not the correct bitstream entry for your
host, override it explicitly:

```bash
PYTORCH_CHIPYARD_FIRESIM_HW_CONFIG_RESNET50_GEMMINI_4CORE=alveo_u250_firesim_fp8x8_gemmini_rocket_4core_no_nic \
  bash scripts/run-firesim-workloads.sh --workload=resnet50-gemmini-4core
```

FPGA host setup, bitstream installation, XRT/XDMA configuration, and the FireSim
hardware database are not created by these scripts. They must already be set up
on the FPGA host.

After the run, the script collects the latest result files into:

```text
$PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR/resnet50-gemmini-4core/model.log
$PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR/resnet50-gemmini-4core/autotune.log
$PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR/resnet50-gemmini-4core/output.bin
```

To collect from an already completed FireSim run without launching a new run:

```bash
bash scripts/run-firesim-workloads.sh --workload=resnet50-gemmini-4core --collect-only
```

Copy `output.bin` back to the original artifact directory before validation:

```bash
cp "$PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR/resnet50-gemmini-4core/output.bin" \
   "$PYTORCH_CHIPYARD_DUMP_PATH/output.bin"
```

Then validate the target output against eager PyTorch:

```bash
cd pytorch-chipyard
PYTORCH_CHIPYARD_DUMP_PATH=$PWD/examples/resnet50/gemmini \
  python examples/ResNet50.py --validate --batch-size 1 --seed 0
```
