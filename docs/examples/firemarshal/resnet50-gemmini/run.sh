#!/bin/sh
set -eu

APP_DIR=/root/pytorch-chipyard-resnet50
cd "$APP_DIR"

rc=0
./model.elf input.bin weights.bin output.bin > stdout.log 2> stderr.log || rc=$?
sync
exit "$rc"
