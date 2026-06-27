# Tutorials and Examples

This page covers two usage patterns:

- Use Triton-Chipyard as a standalone backend and run a small Triton kernel
  through Verilator.
- Use PyTorch-Chipyard by setting environment variables and directly calling a
  real `examples/*.py` entry point.

The PyTorch-Chipyard examples below intentionally do not use wrapper scripts
such as `scripts/run-cnn.sh`. Wrapper scripts are useful for batch experiments,
but the clearest documentation path is the direct environment plus Python entry
point flow.

## Triton-Chipyard Matmul

The paper focuses on PyTorch model execution, but after installation the
Triton-Chipyard backend can also be used independently. This path lowers a
hand-written Triton kernel, such as `triton_chipyard/example/test_matmul.py`,
and runs it through the Verilator simulator when `CHIPYARD_SIM_VERILATOR_PATH`
is set.

The core example kernel below is copied from
`triton_chipyard/example/test_matmul.py`:

```python
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_mask = k_start + offs_k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_mask[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_mask[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)
```

Run it with:

```bash
cd pytorch-chipyard
source scripts/env.sh

# Required when a standalone Triton kernel should actually run in the simulator.
export CHIPYARD_SIM_VERILATOR_PATH=/path/to/chipyard/verilator/simulator

# Optional: IR dump and compile cache locations.
export TRITON_CHIPYARD_DUMP_PATH=$PWD/IR/triton-matmul
export TRITON_CACHE_DIR=/tmp/triton-chipyard-cache/triton-matmul

python triton_chipyard/example/test_matmul.py \
  --block-size-m 16 \
  --block-size-n 16 \
  --block-size-k 16
```

If `CHIPYARD_SIM_VERILATOR_PATH` is empty, the Triton-Chipyard driver skips the
single-kernel simulator launch so it does not interfere with the PyTorch model
artifact-generation path. To make `test_matmul.py` complete the eager PyTorch
comparison, set this variable to a valid Verilator simulator binary.

## PyTorch-Chipyard ResNet50

The PyTorch model path uses entry points under `examples/`. The ResNet50 example
runs `torchvision.models.resnet50` on CPU tensors and sends the Inductor/Triton
kernels produced by `torch.compile(..., backend="inductor")` to the
Triton-Chipyard backend.

The real setup code in `examples/ResNet50.py` contains these pieces:

```python
def artifact_dir() -> Path:
    return Path(os.environ.get("PYTORCH_CHIPYARD_DUMP_PATH", DEFAULT_ARTIFACT_DIR)).resolve()


def configure_triton_chipyard(task_name: str) -> None:
    import triton
    from triton.backends.triton_chipyard.driver import ChipyardDriver

    cache_dir = Path(os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton-chipyard-cache/{task_name}"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    triton.runtime.driver.set_active(ChipyardDriver())
    inductor_config.cpu_backend = "triton_chipyard"


def configure_artifact_env(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.environ["PYTORCH_CHIPYARD_DUMP_PATH"] = str(path)
    os.environ["TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER"] = "1"
```

The compile path in the same file is:

```python
def run_compile(args: argparse.Namespace) -> None:
    path = artifact_dir()
    configure_artifact_env(path)
    configure_triton_chipyard(TASK_NAME)
    model = build_model(args.seed)
    inputs = make_image_input(args.batch_size)
    print_config(path, tuple(inputs.shape), IMAGE_PATH)

    started_at = time.perf_counter()
    compiled_model = torch.compile(model, backend="inductor")
    with torch.inference_mode():
        _ = compiled_model(inputs)
    compile_time_s = time.perf_counter() - started_at

    util = import_artifact_util(path)
    input_path = util.write_inputs_bin(inputs)
    print(f"[compile] seconds={compile_time_s:.3f}")
    print(f"[artifact] input_bin={input_path}")
```

Use the direct command sequence below to compile ResNet50 for the 8x8 FP32
Gemmini configuration. This is the same information the wrapper scripts set,
written out explicitly:

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
unset TRITON_CHIPYARD_RISCV_VARCH
unset TRITON_CHIPYARD_SPIKE_ISA
unset TRITON_CHIPYARD_SPIKE_VARCH

export PYTORCH_CHIPYARD_DUMP_PATH=$PWD/examples/resnet50/gemmini
export TRITON_CHIPYARD_DUMP_PATH=$PYTORCH_CHIPYARD_DUMP_PATH
export TRITON_CACHE_DIR=/tmp/triton-chipyard-cache/resnet50-gemmini

python examples/ResNet50.py --compile --batch-size 1 --seed 0
```

The compile step generates the C++ runner and binary input data, but with
`TORCHINDUCTOR_COMPILE_CHIPYARD_MODEL_RUNNER=0` it does not call the generated
artifact `build.sh`. Build the target ELF explicitly:

```bash
cd "$PYTORCH_CHIPYARD_DUMP_PATH"
CHIPYARD_OMP_NUM_THREADS=4 bash ./build.sh
cp -f model.elf model-4core.elf
```

The generated runner expects positional input, weight, and output paths:

```bash
./model.elf input.bin weights.bin output.bin
```

On target execution, the same directory should receive:

```text
output.bin
model.log
autotune.log
```

FPGA setup and execution are outside the coverage of this work. You must set up
the FPGA host, bitstream, FireSim runtime configuration, and result collection
for your own machine. After `output.bin` is copied back to the artifact
directory, validate it with:

```bash
cd pytorch-chipyard
PYTORCH_CHIPYARD_DUMP_PATH=$PWD/examples/resnet50/gemmini \
  python examples/ResNet50.py --validate --batch-size 1 --seed 0
```

## Other Direct Example Entry Points

The same direct environment pattern applies to the other model entry points.
Change `PYTORCH_CHIPYARD_DUMP_PATH`, `TRITON_CACHE_DIR`, and the Python script:

| Workload | Direct Python entry point |
| --- | --- |
| AlexNet | `python examples/AlexNet.py --compile --batch-size 1 --seed 0` |
| MobileNetV2 | `python examples/MobileNetV2.py --compile --batch-size 1 --seed 0` |
| SqueezeNet | `python examples/SqueezeNet.py --compile --batch-size 1 --seed 0` |
| GPT-2 | `LLM_TOKEN_LENGTH=256 python examples/GPT2.py --compile --batch-size 1 --seed 0` |
| GPT-Neo 125M | `LLM_TOKEN_LENGTH=256 python examples/GPT-Neo-125m.py --compile --batch-size 1 --seed 0` |
| OPT 125M SDPA | `LLM_TOKEN_LENGTH=256 python examples/Opt-125m.py --compile --batch-size 1 --seed 0 --attn sdpa` |
| Pythia 160M window attention | `LLM_TOKEN_LENGTH=1024 python examples/Pythia-160m.py --compile --batch-size 1 --seed 0 --attn window` |
| Gemmini max autotune | `TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE=1 python examples/gemmini-max-autotune.py --compile` |
