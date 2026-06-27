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
