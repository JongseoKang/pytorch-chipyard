# Artifact Evaluation 계획

이 문서는 PyTorch-Chipyard 논문 artifact를 재현하기 위해 현재 필요한 작업을
정리한 초안이다. 최종 artifact 사용 설명서가 아니라, 어떤 스크립트와 실행
경로가 필요한지 결정하기 위한 계획 문서로 둔다.

## 목표

논문 실험은 이 저장소 하나를 기준으로 재현 가능해야 한다. 기존처럼 PyTorch
모델 컴파일 서버와 FPGA 시뮬레이션 서버를 분리해서 쓰는 방식은 artifact
committee에 제공하기 어렵다.

현재 workflow는 다음처럼 나뉘어 있다.

- artifact 생성은 주로 `triton_chipyard/scripts/run-cnn.sh`,
  `triton_chipyard/scripts/run-llm.sh` 주변에 있다.
- FireMarshal/FireSim 실행은 `~/jseo/run-cnn.sh`, `~/jseo/run-llm.sh`
  같은 외부 스크립트에 의존한다.
- 실행 스크립트는 `/home/hongjun/jseo`,
  `/home/hongjun/hk_chipyard/chipyard` 같은 개인 경로를 가정한다.
- 일부 결과 검증은 `scp`로 원격 서버에 업로드하는 방식에 의존한다.

artifact 평가용 workflow는 다음 조건을 만족하는 쪽이 적절하다.

- PyTorch 모델을 Chipyard용 artifact로 컴파일한다.
- 생성된 artifact를 FireMarshal workload로 패키징한다.
- FPGA가 있는 로컬 호스트에서 FireSim을 실행한다.
- 로그와 결과물을 저장소 내부의 결과 디렉터리에 모은다.
- 개인 홈 디렉터리, 별도 checkout, 원격 검증 서버 업로드를 기본 경로에서
  제거한다.

## 재현 대상 실험

논문 Evaluation 섹션의 실험은 크게 다음 그룹으로 나눌 수 있다.

### Table 2: 정확도 및 검증

대상 workload:

- CNN: `SqueezeNet`, `AlexNet`, `MobileNetV2`, `ResNet50`
- LLM: `GPT2-124M`, `GPT-Neo-125M`, `OPT-125M`, `Pythia-160M`

필요한 실행 흐름:

- `torch.compile` 기반으로 artifact를 생성한다.
- `model.elf`, `input.bin`, `weights.bin`을 만든다.
- 각 workload를 FireMarshal workload로 패키징한다.
- 선택한 Gemmini target에서 실행한다.
- `output.bin`을 eager PyTorch 결과와 비교한다.
- compile time, simulation time, numerical error를 정리한다.

### Figure 5, 6, 7: End-to-end 성능 및 scaling

하드웨어 target:

- Rocket CPU: 4, 8, 16 cores
- Saturn/RVV: 2, 4 cores, VLEN 128, ELEN 64
- Gemmini: 2, 4 instances

대상 workload:

- Table 2의 CNN 모델들
- sequence length 256 기준 LLM prefill

필요한 결과:

- end-to-end cycle count
- scaling ratio
- `model.log` 기반 kernel breakdown

### Figure 8: FX graph lowering 비교

목적:

- direct convolution lowering과 im2col+matmul lowering을 비교한다.

대상 workload:

- `ResNet50`
- `AlexNet`
- `MobileNetV2`
- `SqueezeNet`

Target:

- 4-core Gemmini target

필요한 결과:

- direct convolution 대비 im2col+matmul의 normalized performance
- ResNet50, SqueezeNet 등에서 category별 cycle 차이

현재 이슈:

- 저장소 내부 generation 경로에서 direct lowering과 im2col lowering을 명확히
  선택하는 옵션이 아직 정리되어 있지 않다.
- 기존 외부 스크립트에는 im2col workload 이름이 보이지만, 이 repo 안에서
  바로 재현 가능한 통합 경로는 아직 불명확하다.

### Figure 9 및 Table 3: Gemmini autotuning

목적:

- 큰 GEMM 하나에 대해 Triton blocking과 Buddy/Gemmini tiling 선택지를 sweep한다.

Workload:

- matrix multiplication: 1024x1024 by 1024x4096

Target:

- 4 Rocket cores
- 각 core에 8x8 FP32 Gemmini instance 연결
- 256 KiB scratchpad, 64 KiB accumulator

필요한 결과:

- `autotune.log`
- best/worst configuration
- blocking level별 normalized performance
- 선택된 configuration들의 tile별 DMA behavior

현재 이슈:

- `examples/gemmini-max-autotune.py`는 현재 forward path 근처에 syntax error가
  있어, 이 실험을 실행하기 전에 수정이 필요하다.

### Figure 11 및 Table 4: FlexAttention

목적:

- SDPA, FlashAttention-style prefill, windowed FlexAttention prefill을 비교한다.

대상 workload:

- `OPT-125M`
- `Pythia-160M`

Sequence length:

- 256
- 512
- 768
- 1024

Target:

- 주 비교: Rocket+Gemmini
- window-vs-flash ratio 비교: BOOM+Gemmini

필요한 결과:

- prefill cycle count
- Window/Flash 성능 ratio
- full tile, partial tile, total tile 기준 attention score tile count

## 현재 저장소 상태와 gap

현재 checkout 기준으로 확인된 문제는 다음과 같다.

- `pytorch-chipyard/scripts/`에는 현재 설치/환경 스크립트만 있고, 논문 실험
  재현용 스크립트는 아직 없다.
- `triton_chipyard/scripts/` 아래의 기존 artifact generation 스크립트는
  `triton_chipyard/example/...` 같은 예전 경로를 참조한다.
- 실제 모델 entry point는 현재 `examples/` 아래에 있다.
- `pytorch-chipyard/chipyard/sims/firesim`은 현재 checkout에서 완전히
  초기화되어 있지 않다.
- 따라서 `deploy/config_hwdb.yaml`, `deploy/config_build_recipes.yaml` 같은
  FireSim deploy 설정 파일이 없다.
- 기존 FireSim 실행 스크립트는 별도 checkout인 `hk_chipyard/chipyard`를
  전제로 한다.
- 기존 실행 스크립트는 생성 artifact가 `~/jseo/...` 아래에 있다고 가정한다.
- 기존 실행 스크립트는 `output.bin`을 원격 검증 서버로 업로드한다.
  artifact 평가 기본 경로에서는 repo-local 결과 보관이 맞고, 업로드는 명시적
  옵션으로만 두는 편이 낫다.
- FPGA 실행 관련 가정이 hardcoded되어 있다.
  예: `/home/hongjun/FIRESIM_RUNS_DIR`, `/opt/firesim-db0.json`, Alveo U250 bus,
  XRT/XDMA setup.

## 제안하는 스크립트 구조

`scripts/` 아래에 재사용 가능한 작은 스크립트들을 두고, 최상위 driver 하나가
이들을 조합하는 구조가 적절하다.

### `scripts/artifact-env.sh`

역할:

- 저장소 root를 계산한다.
- PyTorch-Chipyard 환경을 source한다.
- conda/tool path를 검증한다.
- `chipyard/env.sh` 존재 여부를 검증한다.
- FireSim 및 FireMarshal 경로를 검증한다.
- FireSim 실행 시 FPGA-dependent 파일을 검증한다.
- 개인 경로 hardcoding을 제거한다.

### `scripts/artifact-common.sh`

역할:

- model/backend/spec parsing 공통 로직을 제공한다.
- backend별 환경변수 설정을 공통화한다.
- output directory layout을 공통화한다.
- log helper를 제공한다.

### `scripts/generate-artifacts.sh`

역할:

- `examples/` 아래 model entry point를 실행한다.
- `model.elf`, `input.bin`, `weights.bin`, `model_spec.json`, 생성된 build script를
  만든다.
- CNN 모델, LLM 모델, Gemmini autotuning microbenchmark를 지원한다.
- backend 선택을 지원한다: `rocket`, `rvv`, `gemmini`.
- core/thread label을 지원한다.
- attention mode를 지원한다: `sdpa`, `flash`, `window`.
- LLM sequence length를 지원한다.

### `scripts/package-firemarshal-workload.sh`

역할:

- 생성된 artifact를 FireMarshal workload overlay로 변환한다.
- workload JSON을 생성한다.
- FireSim deploy workload JSON을 생성한다.
- `~/jseo`를 사용하지 않는다.
- 생성된 workload material을 repo-local 디렉터리 또는 Chipyard FireMarshal
  디렉터리 아래의 명확한 위치에 기록한다.

### `scripts/run-firesim-workload.sh`

역할:

- workload 하나에 대한 runtime YAML을 생성한다.
- run farm을 띄운다.
- `infrasetup`을 실행한다.
- 가능하면 loaded target metadata를 확인한다.
- workload를 실행한다.
- 사용자가 유지하라고 요청하지 않은 경우 run farm을 종료한다.
- `output.bin`, `run.log`, `model.log`, `autotune.log`, `uartlog`를 수집한다.

### `scripts/validate-results.sh`

역할:

- 생성된 `output.bin`을 eager PyTorch 결과와 비교한다.
- Table 2용 correctness summary를 만든다.
- 예상 결과 파일이나 log가 없으면 명확한 에러로 실패한다.

### `scripts/summarize-results.py`

역할:

- `model.log`를 파싱한다.
- `autotune.log`를 파싱한다.
- FireSim result directory를 파싱한다.
- 논문 figure/table 재생성에 필요한 CSV/JSON summary를 생성한다.

### `scripts/replicate-paper.sh`

역할:

- 최상위 artifact driver 역할을 한다.
- preset을 제공한다.
  - `--preset smoke`
  - `--preset table2`
  - `--preset fig5`
  - `--preset fig6`
  - `--preset fig8`
  - `--preset fig9`
  - `--preset fig11`
  - `--preset full`
- stage별 실행을 지원한다.
  - `--generate-only`
  - `--package-only`
  - `--run-only`
  - `--validate-only`
  - `--summarize-only`
- 기본 동작에서는 원격 업로드를 하지 않는다.

## Dockerize 관련 메모

컴파일 단계는 FireSim 실행 단계보다 Dockerize하기 쉽다. FireSim/FPGA 실행은
호스트 FPGA 장치, XRT/XDMA, FireSim database, bitstream 설정에 강하게 묶인다.

Compile/container 쪽에 필요한 항목:

- Python 3.12 환경
- PyTorch 2.8.0 호환 setup
- patched TorchInductor
- Triton 및 Triton-Chipyard
- Buddy-MLIR
- LLVM build product
- Chipyard RISC-V toolchain path

FireSim/FPGA 쪽에 필요한 항목:

- host FPGA access
- XRT/XDMA 및 Alveo U250 device visibility
- 필요 시 privileged container mode 또는 명시적인 device mount
- `/opt/firesim-db0.json` 같은 FireSim database
- hardware target과 맞는 `config_hwdb.yaml`, `config_build_recipes.yaml`
- 모든 평가 target에 대한 prebuilt bitstream
  - artifact에서 bitstream build까지 지원할 것이 아니라면 prebuilt bitstream을
    제공하거나 받는 경로가 필요하다.

권장 artifact mode:

- `compile-only`: FPGA 없이 model artifact만 생성한다.
- `package-only`: FireSim 실행 없이 FireMarshal workload까지만 만든다.
- `smoke`: 사용 가능한 FPGA에서 작은 workload 하나만 실행한다.
- `full`: 논문 전체 matrix를 실행한다. 상당한 FPGA 시간이 필요하다고 명시해야 한다.

## 바로 정리할 TODO

- repo-local output layout을 결정한다. 예: `artifact-results/`.
- `examples/gemmini-max-autotune.py`를 수정하거나 대체한다.
- `triton_chipyard/scripts/run-cnn.sh`, `triton_chipyard/scripts/run-llm.sh`의
  예전 경로 가정을 제거하거나 새 스크립트로 대체한다.
- `~/jseo/run-cnn.sh`, `~/jseo/run-llm.sh`에서 FireMarshal/FireSim workload 생성에
  필요한 부분만 이 저장소로 port한다.
- default workflow에서 원격 업로드를 제거한다.
- FireSim deploy config 및 FPGA database 누락 여부를 명확히 검사한다.
- 긴 실험 전에 artifact reviewer가 먼저 돌릴 수 있는 작은 smoke test를 추가한다.
