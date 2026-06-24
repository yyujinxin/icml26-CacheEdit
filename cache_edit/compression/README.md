# Compression Module

Activation compression is documented in the repository-level
`README_COMPRESSION.md`.

The main implementation files are:

- `activation_compressor.py`: tensor quantization, tiling, NVENC/NVDEC codec orchestration.
- `pipeline/quantization.py`: channel-wise and group-wise quantization steps.
- `pipeline/nvenc.py`: Python pipeline wrappers for the native codec extension.
- `csrc/`: native NVENC/NVDEC bindings.
