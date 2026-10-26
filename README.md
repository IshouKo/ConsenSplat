# ConsenSplat

ConsenSplat is a research implementation of 3D Gaussian Splatting for multi-view image restoration. It combines a shared Gaussian scene with illumination, medium, and particle decomposition, and includes a multi-view Gaussian observation bank for consensus supervision.

## Deploy on a GPU machine

### Prerequisites

- A Linux machine with an NVIDIA CUDA-capable GPU and a working NVIDIA driver.
- Python 3.10 or newer, Git, and a C++ build toolchain.
- A CUDA toolkit with `nvcc`, compatible with the CUDA version used by your PyTorch installation.
- Network access for dependency installation and the initial download of LPIPS/VGG evaluation weights.

The training entry points require CUDA. CPU-only and macOS/MPS training are not supported by this implementation. This is a command-line research project; deployment means setting up a GPU environment, not starting a web service.

### Installation

```bash
git clone https://github.com/IshouKo/ConsenSplat.git
cd ConsenSplat
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja
```

Install a matching CUDA-enabled `torch` and `torchvision` pair for your GPU environment before building the extensions. Then install the remaining dependencies and the two CUDA extensions specified by this project:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -r requirements-cuda.txt
```

`requirements-cuda.txt` installs `diff-gaussian-rasterization` from the `dr_aa` branch and `simple-knn` from its upstream repository. These commands install the extensions directly; a submodule checkout is not required. Dependencies are not pinned, so record the versions used for each experiment.

Verify the environment from the repository root:

```bash
nvidia-smi
nvcc --version
python -c "import torch; assert torch.cuda.is_available(); import diff_gaussian_rasterization; import simple_knn._C; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
python train_fused.py --help
```
