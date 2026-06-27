# pytorch-chipyard
A Reusable Compiler-Stack Foundation for ML Research on Chipyard Hardware.

## Installation

```
# pre-requisite: conda-24.11.3
git clone https://github.com/JongseoKang/pytorch-chipyard

cd ./pytorch-chipyard
git submodule update --init pytorch triton triton_chipyard llvm-project buddy-mlir chipyard

# install chipyard
cd chipyard
./build-setup.sh riscv-tools
cd ..

# install otherwise
bash scripts/install.sh
```
