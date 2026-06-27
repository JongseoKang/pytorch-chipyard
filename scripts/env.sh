#!/usr/bin/env bash
# Root workspace
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
export WORKSPACE="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
# Buddy-MLIR
export BUDDY_BINARY_DIR="$WORKSPACE/buddy-mlir/build/bin"
# Chipyard
export CHIPYARD_ENV_PATH="$WORKSPACE/chipyard/env.sh"
# triton-chipyard-opt path
export TRITON_CHIPYARD_OPT_PATH="$WORKSPACE/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_chipyard/tools/triton-chipyard-opt/triton-chipyard-opt"
# LLVM project
export LLVM_PROJECT_PATH="$WORKSPACE/llvm-project"
# Verilator: triton-chipyard passes simulation path, only does compilation
export CHIPYARD_SIM_VERILATOR_PATH="" 

# TorchInductor settings
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_MAX_AUTOTUNE=1 
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON 
export TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS=TRITON 
export TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER=1

# Gemmini Usage env vars
export TRITON_CHIPYARD_USE_GEMMINI=1
export TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE=0 # if set, it tries multiple gemmini's tilings, which consumes very long simulation time
# FP32 Gemmini Configs
export TRITON_CHIPYARD_GEMMINI_ADDR_LEN="32"
export TRITON_CHIPYARD_GEMMINI_DIM="8"
export TRITON_CHIPYARD_GEMMINI_BANK_ROWS="2048"
export TRITON_CHIPYARD_GEMMINI_ACC_ROWS="2048"
export TRITON_CHIPYARD_GEMMINI_ELEM_T="f32"
export TRITON_CHIPYARD_GEMMINI_ACC_T="f32"
export TRITON_CHIPYARD_RISCV_MARCH="rv64imafdc"
export TRITON_CHIPYARD_RISCV_MABI="lp64d"

# RVV Usage env vars: please uncomment below lines, comment above gemmini settings, re 'source env.sh'
# export TRITON_CHIPYARD_USE_RVV=1
# export TRITON_CHIPYARD_RISCV_MARCH=rv64imafdcv_zicsr_zifencei_zvl128b
# export TRITON_CHIPYARD_RISCV_MABI=lp64d
# export TRITON_CHIPYARD_RISCV_VARCH=vlen:128,elen:64
