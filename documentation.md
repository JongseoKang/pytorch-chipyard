# PyTorch-Chipyard Website Documentation Plan

This document records the proposed website documentation structure for
PyTorch-Chipyard. The goal is to satisfy the CGO tool-paper documentation
requirement with a focused website that explains the tool, shows how to get
started, and points readers to runnable tutorials and examples.

The documentation should stay compact at first. It should avoid creating empty
or redundant pages before the project is ready for public users.

## Documentation Goals

- Present PyTorch-Chipyard as a reusable compiler-stack foundation for ML
  research on Chipyard hardware.
- Help a new reader understand the project scope quickly.
- Provide a clear path from checkout/setup to the first runnable example.
- Keep PyTorch-Chipyard examples and Triton-Chipyard examples visibly separate.
- Document supported functionality and current limitations honestly.
- Leave citation and license information as placeholders until the final wording
  is confirmed with the advisor.

## Proposed Website Pages

The initial website should use the following top-level pages:

```text
Overview
Getting Started
Tutorials and Examples
Supported Features and Limitations
Citation and License
```

This structure intentionally omits separate pages for troubleshooting,
configuration, paper-result reproduction, and the compiler pipeline. Those topics
are either premature or overlap with the pages above.

## Page Details

### Overview

The overview page should be the landing page for the documentation site. It
should explain what PyTorch-Chipyard is, who it is for, and how the main
components fit together.

Suggested content:

- One-paragraph project summary.
- The relationship between PyTorch-Chipyard and Triton-Chipyard.
- A short end-to-end flow:

```text
PyTorch model
  -> TorchInductor / Triton
  -> Triton-Chipyard lowering
  -> MLIR / Buddy / Gemmini-oriented artifacts
  -> Chipyard execution path
```

- A short note that the project is research infrastructure for compiling ML
  workloads toward Chipyard/Gemmini-style hardware targets.
- Links to Getting Started and Tutorials and Examples.

There should not be a separate "Compiler Pipeline" page for now. The overview
page can carry the high-level pipeline explanation.

### Getting Started

The getting-started page should be the first hands-on path through the project.
It should contain the smallest reliable sequence that lets a reader set up the
repository and run one example.

Suggested content:

- Repository layout.
- Required dependencies and expected checkout structure.
- Environment setup.
- How to enable the Triton-Chipyard backend.
- How to run the smallest recommended example.
- Where generated artifacts are written.
- What output files or logs indicate that the run succeeded.

This page can include setup/configuration details, but there should not be a
separate "Configuration" page yet. Any target assumptions, environment variables,
or hardware-related settings that matter to users can be documented here or in
Supported Features and Limitations.

### Tutorials and Examples

Tutorials and examples should be merged into one section. For this project, the
distinction between a tutorial and an example is not useful enough to justify
separate pages at the start. A tutorial can simply be a more detailed example.

This section should be divided into two groups.

#### PyTorch-Chipyard Examples

These examples should cover model-level compilation from PyTorch through the
PyTorch-Chipyard path.

Current candidate examples:

- `examples/AlexNet.py`
- `examples/SqueezeNet.py`
- `examples/ResNet50.py`
- `examples/MobileNetV2.py`
- `examples/GPT2.py`
- `examples/GPT-Neo-125m.py`
- `examples/Opt-125m.py`
- `examples/Pythia-160m.py`
- `examples/gemmini-max-autotune.py`

Each example entry should eventually include:

- What the example demonstrates.
- The command to run it.
- Required model/data inputs.
- Expected artifact directory.
- Expected generated files.
- Any known constraints.

#### Triton-Chipyard Examples

These examples should cover lower-level Triton-Chipyard behavior and backend
validation. They are useful for readers who want to inspect compiler behavior
below the model level.

Current candidate examples:

- `triton_chipyard/example/test_matmul.py`
- `triton_chipyard/example/test_softmax.py`
- `triton_chipyard/example/test_conv.py`
- `triton_chipyard/example/test_layernorm.py`
- `triton_chipyard/example/test_runner.py`

Current candidate scripts:

- `triton_chipyard/scripts/run-cnn.sh`
- `triton_chipyard/scripts/run-llm.sh`
- `triton_chipyard/scripts/run_vision_artifacts.sh`
- `triton_chipyard/scripts/run_llm_prefill_artifacts.sh`
- `triton_chipyard/scripts/run_llm_flex_window_prefill_artifacts.sh`
- `triton_chipyard/scripts/verify-cnn-output.py`
- `triton_chipyard/scripts/verify-llm-output.py`
- `triton_chipyard/scripts/pack-llm-input.py`

The paper-result reproduction material can be folded into this section instead
of becoming a separate "Reproducing Paper Results" page. If an example
corresponds to a paper figure, table, or artifact, tag it explicitly in the
example description.

### Supported Features and Limitations

This page should combine user-visible support status, target assumptions, and
current limitations. It replaces a separate configuration page.

Suggested supported-feature topics:

- Model-level PyTorch compilation through TorchInductor.
- Triton-Chipyard as the backend path for generated kernels.
- Generation of Chipyard/Gemmini-oriented runner artifacts.
- Vision-model examples such as AlexNet, SqueezeNet, ResNet50, and MobileNetV2.
- LLM examples such as GPT-2, GPT-Neo, OPT, and Pythia variants.
- Artifact files such as `runner.cpp`, `model_spec.json`, `weights.bin`,
  `input.bin`, `output.bin`, and generated build scripts.
- Lower-level Triton examples for matmul, softmax, convolution, layernorm, and
  runner testing.

Suggested limitation topics:

- The strongest path is currently model-level PyTorch Inductor compilation into
  Chipyard runner artifacts.
- Matmul support is centered on 2D dot/matmul patterns. Batched matmul support
  should be described carefully if it is represented through launch dimensions
  rather than rank-3 batch matmul lowering.
- Convolution and pooling support should be described in terms of the currently
  validated Inductor/Triton paths.
- Flex attention and SDPA support should be marked experimental if they are not
  fully validated.
- Some target-friendly math rewrites may still require more numerical validation.
- Exact environment and checkout-layout assumptions should be documented here
  when they affect whether examples run.

### Citation and License

This page should exist, but its content can remain as a placeholder until the
advisor confirms the final wording.

Suggested placeholder:

```markdown
# Citation and License

## Citation

TODO: Add the final paper citation and BibTeX entry after the submission metadata
is confirmed.

## License

TODO: Confirm the repository license and third-party license notes before public
release.
```

## Pages Not Included Initially

The following pages should not be created as standalone pages in the initial
website.

### Troubleshooting

Do not add this page yet. The project has not been publicly distributed, so
there are no real user-facing troubleshooting cases to document. It can be added
later after common setup or runtime issues are known.

### Configuration

Do not add a separate configuration page yet. Configuration-like information
should be included in Getting Started when it is required to run the first
example, or in Supported Features and Limitations when it describes target
assumptions.

### Reproducing Paper Results

Do not add a separate reproduction page yet. The content would likely be large
and would overlap heavily with tutorials and examples. Paper-related examples can
be tagged inside Tutorials and Examples.

### Compiler Pipeline

Do not add a separate compiler-pipeline page yet. The overview page should carry
the high-level pipeline summary.

## Suggested Initial File Layout

If the website uses Markdown pages, the documentation can start with this layout:

```text
docs/
  index.md
  getting-started.md
  tutorials-and-examples.md
  supported-features-and-limitations.md
  citation-and-license.md
```

If the documentation remains a single page at first, this `documentation.md`
file can be used as the source outline and expanded into the website pages later.
