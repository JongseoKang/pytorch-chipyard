PyTorch-Chipyard Documentation
==============================

PyTorch-Chipyard is a reusable compiler-stack foundation for ML research on
Chipyard hardware. It connects PyTorch model compilation through
TorchInductor/Triton with Triton-Chipyard lowering and Chipyard/Gemmini-oriented
runner artifacts.

The main flow is:

.. code-block:: text

   PyTorch model
     -> TorchInductor / Triton
     -> Triton-Chipyard lowering
     -> MLIR / Buddy / Gemmini-oriented artifacts
     -> Chipyard execution path

This documentation is intentionally compact. It focuses on the information
needed to understand the tool, set up a local checkout, run examples, and inspect
the current support boundary.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting-started
   tutorials-and-examples
   supported-features-and-limitations
   citation-and-license
