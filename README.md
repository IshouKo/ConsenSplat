# ConsenSplat

ConsenSplat is a research implementation of physics-aware 3D Gaussian Splatting for multi-view image restoration. It combines a shared Gaussian scene with illumination, medium, and particle decomposition, and includes a multi-view Gaussian observation bank for consensus supervision.

The code includes the components used by the canonical A4 MV3D-Teacher experiments: analytic initialization, leave-one-view-out observations, brightness calibration, confidence-aware supervision of Gaussian SH-DC colors, and held-out physical re-degradation checks. Clean reference images are evaluation-only in the degraded-only training workflow.

## Repository scope

This repository contains the training code, renderer integration, physics modules, and tests. The following local directories are intentionally excluded from version control:

- `outputs/`: trained models, checkpoints, and generated artifacts.
- `prepared/`: prepared datasets and derived inputs.
- `scripts/`: local preprocessing and campaign automation.
- `configs/`: local experiment and campaign configurations.

You must supply your own dataset. The commands below provide a basic deployment and training workflow; they do not reproduce the full 28-scene A4 campaign, whose preprocessing scripts and tuned configurations are not included.

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

Run training commands from the repository root so that the local renderer, scene, and utility modules can be imported.

## Prepare a scene

Provide undistorted images and a matching COLMAP reconstruction. A typical scene has this layout:

```text
prepared/my_scene/
├── images/                 # Degraded input images
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

The loader also accepts the corresponding COLMAP text files. Camera models must be `PINHOLE` or `SIMPLE_PINHOLE`. Image names must match the reconstruction. The loader generates `points3D.ply` when needed, so the scene directory must be writable.

For fused training, `--multi_condition_dirs` identifies the original degraded scene root, which must contain an `images/` directory. For multiple conditions, pass semicolon-separated roots in quotes; their images must align with the same scene and camera names.

## Train

Start a basic single-condition run without any external configuration file:

```bash
CUDA_VISIBLE_DEVICES=0 python train_fused.py \
  --source_path prepared/my_scene \
  --model_path outputs/my_scene \
  --multi_condition_dirs prepared/my_scene \
  --fused_training \
  --degraded_only \
  --eval \
  --iterations 30000 \
  --test_iterations 15000 20000 25000 30000 \
  --save_iterations 15000 20000 25000 30000 \
  --checkpoint_iterations 15000 20000 25000 30000
```

This uses the parser defaults for the physics schedule. It does not enable the complete canonical A4 teacher configuration. Inspect `arguments/__init__.py` and `python train_fused.py --help` for the available settings. Custom YAML files can be loaded with `--config /path/to/experiment.yaml`; keep local experiment files under the ignored `configs/` directory.

`--eval` enables a held-out camera split. To compute reference-based metrics, add `--clean_dir /path/to/clean/images`, using filenames that correspond to the degraded images. Without matching clean references, the validation routine returns no metrics. LPIPS/VGG weights may still be downloaded when validation is invoked.

### Resume a checkpoint

Repeat the original training arguments and add:

```bash
--start_checkpoint outputs/my_scene/chkpnt15000.pth
```

Keep the dataset, model settings, and schedule consistent with the original run. `--iterations` is the total target iteration count, not the number of additional iterations.

### Evaluate a checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python train_fused.py \
  --source_path prepared/my_scene \
  --model_path outputs/my_scene \
  --multi_condition_dirs prepared/my_scene \
  --fused_training \
  --degraded_only \
  --eval \
  --clean_dir /path/to/clean/images \
  --start_checkpoint outputs/my_scene/chkpnt30000.pth \
  --evaluate_checkpoint_only \
  --evaluation_iteration 30000
```

If training used custom settings, supply those same settings for evaluation.

## Generated artifacts

Artifacts are written under `--model_path`, including:

- `point_cloud/iteration_<N>/point_cloud.ply`: saved Gaussian scenes.
- `physics/iteration_<N>.pt`: saved physics state.
- `chkpnt<N>.pth`: checkpoints at requested checkpoint iterations.
- `metrics/base/test_iteration_<N>.json`: per-image and average metrics when clean references are available.

Keep model paths under `outputs/` to exclude generated files from Git.

## Troubleshooting

- **CUDA unavailable:** check the driver and ensure PyTorch was installed with CUDA support.
- **Extension build fails:** check `nvcc`, the C++ compiler, and CUDA/PyTorch compatibility; rebuild the extensions after changing PyTorch.
- **Scene type not recognized:** check `--source_path` and the `sparse/0/` reconstruction layout.
- **Missing condition image:** check that every condition root contains matching files under `images/`.
- **GPU out of memory:** reduce input resolution using `--resolution 2` or `--resolution 4`, then adjust Gaussian growth limits if necessary.

## License

See [LICENSE.md](LICENSE.md) and the copyright notices in the source files. The inherited Gaussian Splatting license restricts use to non-commercial research and evaluation; review the terms before using or redistributing the code.
