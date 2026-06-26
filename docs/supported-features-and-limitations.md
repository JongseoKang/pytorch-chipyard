# Supported Features and Limitations

This page combines user-visible support status, target assumptions, and current
limitations. It also carries configuration-like information that affects whether
examples run.

## Supported Features

The strongest supported path is model-level PyTorch compilation through
TorchInductor and Triton-Chipyard into Chipyard-oriented runner artifacts.

Supported or actively documented areas:

- PyTorch model examples compiled through TorchInductor.
- Triton-Chipyard as the backend path for generated kernels.
- Generation of Chipyard/Gemmini-oriented runner artifacts.
- Vision model examples including AlexNet, SqueezeNet, ResNet50, and MobileNetV2.
- LLM examples including GPT-2, GPT-Neo, OPT, and Pythia variants.
- Artifact files such as `runner.cpp`, `model_spec.json`, `weights.bin`,
  `weights.manifest.json`, `input.bin`, `output.bin`, `build.sh`, and `util.py`.
- Lower-level Triton examples for matmul, softmax, convolution, layernorm, and
  runner testing.

## Artifact Contract

The generated artifact directory is the interface between PyTorch compilation and
the Chipyard execution path.

Important files:

- `runner.cpp`: generated model runner with kernel launches and buffer planning.
- `model_spec.json`: model input, output, constant, buffer, kernel, size, stride,
  and layout contract.
- `weights.bin`: packed static model parameters and buffers.
- `weights.manifest.json`: metadata for packed weights.
- `input.bin`: packed runtime inputs.
- `output.bin`: expected target output location for validation.
- `build.sh`: helper script for building the generated runner.
- `util.py`: Python helpers for writing inputs and reading outputs.

When debugging an artifact mismatch, `model_spec.json` should be treated as the
source of truth for the runner contract.

## Target Assumptions

The current setup assumes a matching local compiler stack:

- PyTorch branch with PyTorch-Chipyard integration.
- Triton branch with the out-of-tree Triton-Chipyard backend enabled.
- Triton-Chipyard backend checkout.
- LLVM and Buddy-MLIR checkouts/builds.
- Chipyard/Gemmini environment for target execution.

The checked-in `triton_chipyard/env.sh` may contain machine-specific paths.
Update it before running examples on a new machine.

## Current Limitations

Current known limitations:

- The strongest path is model-level PyTorch Inductor compilation into Chipyard
  runner artifacts.
- Matmul support is centered on 2D dot/matmul patterns. Batched matmul support
  should be described carefully when it is represented through launch dimensions
  rather than rank-3 batch-matmul lowering.
- Convolution and pooling support should be described in terms of the currently
  validated Inductor/Triton paths.
- Flex attention and SDPA lowering are experimental unless explicitly validated
  for a given model and shape.
- Some target-friendly math rewrites may still require additional numerical
  validation.
- The current documentation does not include a separate troubleshooting page
  because the project has not yet collected public user issues.
