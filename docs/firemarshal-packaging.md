# FireMarshal Packaging

PyTorch-Chipyard compilation produces a host-side artifact directory. To boot the
runner inside a Chipyard/FireSim Linux image, package the generated files into a
FireMarshal workload. This page covers image packaging only. FPGA setup,
bitstream management, FireSim runtime configuration, running the simulation, and
collecting results are not covered by this work; set up and run the FPGA system
for your environment separately.

The example files in this page are also checked in under
`docs/examples/firemarshal/resnet50-gemmini/`.

## Artifact Directory

Start from a compiled and built artifact directory:

```text
examples/resnet50/gemmini/
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

The compilation step writes `build.sh`, but it does not automatically call it in
the documented direct flow. Build the ELF before packaging:

```bash
cd pytorch-chipyard
source scripts/env.sh

cd examples/resnet50/gemmini
CHIPYARD_OMP_NUM_THREADS=4 bash ./build.sh
cp -f model.elf model-4core.elf
```

## Workload Directory

The FireMarshal workload directory for this example has this structure:

```text
docs/examples/firemarshal/resnet50-gemmini/
  resnet50-gemmini.yaml
  build.sh
  run.sh
  overlay/
    root/
      pytorch-chipyard-resnet50/
```

`build.sh` is a FireMarshal `host-init` script. It copies a generated
PyTorch-Chipyard artifact into the workload overlay. This script is distinct
from the generated artifact `build.sh`.

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
: "${PYTORCH_CHIPYARD_ARTIFACT_DIR:?set this to examples/resnet50/gemmini or another artifact directory}"

GUEST_APP_DIR="${SCRIPT_DIR}/overlay/root/pytorch-chipyard-resnet50"
rm -rf "${GUEST_APP_DIR}"
mkdir -p "${GUEST_APP_DIR}"

cp "${PYTORCH_CHIPYARD_ARTIFACT_DIR}/model.elf" "${GUEST_APP_DIR}/model.elf"
cp "${PYTORCH_CHIPYARD_ARTIFACT_DIR}/input.bin" "${GUEST_APP_DIR}/input.bin"
cp "${PYTORCH_CHIPYARD_ARTIFACT_DIR}/weights.bin" "${GUEST_APP_DIR}/weights.bin"
cp "${PYTORCH_CHIPYARD_ARTIFACT_DIR}/model_spec.json" "${GUEST_APP_DIR}/model_spec.json"
cp "${PYTORCH_CHIPYARD_ARTIFACT_DIR}/weights.manifest.json" "${GUEST_APP_DIR}/weights.manifest.json"
cp "${SCRIPT_DIR}/run.sh" "${GUEST_APP_DIR}/run.sh"

chmod +x "${GUEST_APP_DIR}/model.elf" "${GUEST_APP_DIR}/run.sh"
```

`run.sh` is the command executed inside the guest image:

```sh
#!/bin/sh
set -eu

APP_DIR=/root/pytorch-chipyard-resnet50
cd "$APP_DIR"

rc=0
./model.elf input.bin weights.bin output.bin > stdout.log 2> stderr.log || rc=$?
sync
exit "$rc"
```

The FireMarshal workload config follows the same style as Chipyard's tutorial
configs such as `chipyard/software/tutorial/marshal-configs/resnet50-linux.yaml`:

```yaml
{
  "name" : "pytorch-chipyard-resnet50-gemmini",
  "base" : "br-base.json",
  "workdir" : ".",
  "host-init" : "build.sh",
  "overlay" : "overlay",
  "command" : "/root/pytorch-chipyard-resnet50/run.sh",
  "spike-args" : "--extension=gemmini"
}
```

Build the FireMarshal image with:

```bash
cd pytorch-chipyard
export PYTORCH_CHIPYARD_ARTIFACT_DIR=$PWD/examples/resnet50/gemmini

cd chipyard/software/firemarshal
./marshal build ../../../docs/examples/firemarshal/resnet50-gemmini/resnet50-gemmini.yaml
```

This creates the FireMarshal workload image from the overlay. It does not set up
or run the FPGA. After you run the image through your own FPGA/FireSim setup,
collect these files from the guest application directory:

```text
/root/pytorch-chipyard-resnet50/output.bin
/root/pytorch-chipyard-resnet50/model.log
/root/pytorch-chipyard-resnet50/autotune.log
/root/pytorch-chipyard-resnet50/stdout.log
/root/pytorch-chipyard-resnet50/stderr.log
```

Copy `output.bin` back into the matching artifact directory before validation.
Keep `model.log` and `autotune.log` with the result set; the figure scripts parse
these logs for cycle summaries and autotune choices.

## Result File Examples

`model.log` is written by the generated runner. Its format begins with a model
summary and then lists hot kernels:

```text
Model Launch Log

Avg Model cycle: 12345678.00
Max Model cycle: 12390000
Min Model cycle: 12300000
Model samples: 5

Top 10 Hot kernels

1. Large conv/mm
Launch count: 16
Samples: 80
Total launch cycle: 9876543 (62.50%)
Avg launch cycle: 123456.79
Min launch cycle: 120000
Max launch cycle: 130000
Semantic source:
- convolution
Normalized kernel names:
- triton_mm
Autotune sites:
- site_0
```

`autotune.log` records candidate measurements and the chosen kernel for each
deferred autotune site:

```text
Autotune Log

Autotune Candidate
Site: 0
Site name: convolution
Site id: site_0
Candidate: 0
Kernel: triton_mm_0
Raw kernel: triton_mm_0
Normalized kernel: triton_mm
Grid: 196,1,1
Cycles: 456789
Source: measured
Config count: 6
  BLOCK_M=16
  BLOCK_N=16
  BLOCK_K=16

Autotune Site Result
Site: 0
Site name: convolution
Site id: site_0
Dispatch supported: 1
Best candidate: 0
Best kernel: triton_mm_0
Best raw kernel: triton_mm_0
Best cycles: 456789
```

The exact numbers depend on the target, bitstream, core count, and workload.
The field names and section layout are generated by `runner.cpp` and are stable
enough for the repository's parsing scripts.
