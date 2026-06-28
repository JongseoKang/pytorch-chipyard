# Section 3: PyTorch-Chipyard

This section is a structural placeholder for the PyTorch-Chipyard design
description. The figures are included first; detailed text can be expanded from
these diagrams.

## 3.1 Overview

![PyTorch-Chipyard system overview](figures/system-overview.png)

The full system connects PyTorch model compilation, Triton-Chipyard lowering,
and Chipyard/Gemmini execution artifacts.

## 3.2 PyTorch-Side Modifications

![Chipyard artifact generation](figures/chipyard-artifact-generation.png)

This figure summarizes the PyTorch/TorchInductor changes that generate model
artifacts such as `runner.cpp`, `model_spec.json`, `weights.bin`, and
`input.bin`.

## 3.3 Triton-Chipyard

![Triton-Chipyard compilation flow](figures/compilation-flow.png)

This figure summarizes the Triton-Chipyard lowering path from Triton IR through
MLIR/Buddy-oriented lowering toward Chipyard execution artifacts.
