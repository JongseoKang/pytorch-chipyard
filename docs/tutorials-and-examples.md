# Tutorials and Examples

Tutorials and examples are grouped together. In this project, a tutorial is a
more detailed example, so the initial documentation does not split them into
separate sections.

There are two example groups:

- PyTorch-Chipyard examples compile model-level PyTorch workloads.
- Triton-Chipyard examples exercise lower-level backend behavior.

## PyTorch-Chipyard Examples

The PyTorch-Chipyard examples live in `examples/`. They compile PyTorch models
through TorchInductor and the Triton-Chipyard backend into Chipyard-oriented
runner artifacts.

Common compile form:

```sh
cd pytorch-chipyard
source triton_chipyard/env.sh
python examples/<ExampleName>.py --compile
```

Common validation form:

```sh
python examples/<ExampleName>.py --validate
```

Current model examples:

| Example | Path | Notes |
| --- | --- | --- |
| AlexNet | `examples/AlexNet.py` | Vision model artifact generation and validation. |
| SqueezeNet | `examples/SqueezeNet.py` | Vision model artifact generation and validation. |
| ResNet50 | `examples/ResNet50.py` | Larger torchvision CNN path. |
| MobileNetV2 | `examples/MobileNetV2.py` | Mobile vision model path. |
| GPT-2 | `examples/GPT2.py` | LLM artifact generation path. |
| GPT-Neo 125M | `examples/GPT-Neo-125m.py` | LLM artifact generation path. |
| OPT 125M | `examples/Opt-125m.py` | LLM path with selectable attention mode. |
| Pythia 160M | `examples/Pythia-160m.py` | LLM path with selectable attention mode. |
| Gemmini max autotune | `examples/gemmini-max-autotune.py` | Autotuning-oriented experiment. |

Example output is controlled by `PYTORCH_CHIPYARD_DUMP_PATH`. If it is unset,
each example chooses a default directory under `IR/`.

## Vision Artifact Scripts

The vision scripts in `triton_chipyard/scripts/` automate groups of CNN runs and
target-specific artifact handling:

```sh
cd pytorch-chipyard/triton_chipyard
./scripts/run-cnn.sh --compile-only alexnet
./scripts/run-cnn.sh --backend gemmini mobilenetv2
./scripts/run-cnn.sh --validate --backend rvv mobilenetv2
```

Useful script:

- `triton_chipyard/scripts/run-cnn.sh`
- `triton_chipyard/scripts/run_vision_artifacts.sh`
- `triton_chipyard/scripts/verify-cnn-output.py`

## LLM Artifact Scripts

The LLM scripts automate OPT, Pythia, GPT-2, and GPT-Neo artifact generation and
verification flows:

```sh
cd pytorch-chipyard/triton_chipyard
./scripts/run-llm.sh --backend gemmini opt
LLM_TOKEN_LENGTHS="256 512" ./scripts/run-llm.sh --backend rocket opt pythia
./scripts/run-llm.sh --verify opt
```

Useful script:

- `triton_chipyard/scripts/run-llm.sh`
- `triton_chipyard/scripts/run_llm_prefill_artifacts.sh`
- `triton_chipyard/scripts/run_llm_flex_window_prefill_artifacts.sh`
- `triton_chipyard/scripts/pack-llm-input.py`
- `triton_chipyard/scripts/verify-llm-output.py`

## Triton-Chipyard Examples

The Triton-Chipyard examples live in `triton_chipyard/example/`. They are useful
when inspecting the backend below the PyTorch model level.

Current backend examples:

| Example | Path | Notes |
| --- | --- | --- |
| Matmul | `triton_chipyard/example/test_matmul.py` | Triton matmul lowering and paper-section shape presets. |
| Softmax | `triton_chipyard/example/test_softmax.py` | Softmax lowering path. |
| Convolution | `triton_chipyard/example/test_conv.py` | Convolution lowering path. |
| LayerNorm | `triton_chipyard/example/test_layernorm.py` | LayerNorm and reduction-oriented lowering path. |
| Runner | `triton_chipyard/example/test_runner.py` | Runner-oriented backend test. |

For example, the matmul test can list available shape presets:

```sh
cd pytorch-chipyard
source triton_chipyard/env.sh
python triton_chipyard/example/test_matmul.py --list-shapes
```

The paper-result reproduction material should be tagged inside this section
rather than split into a separate page. If an example corresponds to a paper
figure, table, or artifact, document that relationship in the example entry.
