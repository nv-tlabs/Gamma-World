# Setup Guide

<!--TOC-->

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Downloading Checkpoints](#downloading-checkpoints)
- [Troubleshooting](#troubleshooting)

<!--TOC-->

## System Requirements

* NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer
* NVIDIA driver >=570.124.06 compatible with [CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions)
* Linux x86-64
* glibc>=2.35 (e.g Ubuntu >=22.04)

## Installation

Install [git lfs](https://git-lfs.com/):

```bash
sudo apt install git-lfs
git lfs install
```

Clone the repository:

```bash
git clone git@github.com:nv-tlabs/Gamma-World.git
cd Gamma-World
git lfs pull
```

Install one of the following environments:

<details id="virtual-environment"><summary><b>Virtual Environment</b></summary>

Install system dependencies:

```shell
sudo apt update && sudo apt -y install curl ffmpeg libx11-dev tree wget
```

* [uv](https://docs.astral.sh/uv/getting-started/installation/)

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Install the package into a new environment:

```shell
uv python install
uv sync --extra=cu128
source .venv/bin/activate
```

Or, install the package into the active environment (e.g. conda):

```shell
uv sync --extra=cu128 --active --inexact
```

CUDA Variants:

| CUDA Version | Arguments | Notes |
| --- | --- | --- |
| CUDA 12.8 | `--extra cu128` | [NVIDIA Driver](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |
| CUDA 13.0 | `--extra cu130` | [NVIDIA Driver](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |

For DGX Spark and Jetson AGX, you must use CUDA 13.0.
</details>

<details id="docker-container"><summary><b>Docker Container</b></summary>

Please make sure you have access to Docker on your machine and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed.

Build the container:

```bash
# Ampere - Hopper
image_tag=$(docker build -f Dockerfile -q .)
# Blackwell
image_tag=$(docker build -f docker/nightly.Dockerfile -q .)
```

Run the container:

```bash
docker run -it --runtime=nvidia --ipc=host --rm -v .:/workspace -v /workspace/.venv -v /root/.cache:/root/.cache -e HF_TOKEN="$HF_TOKEN" $image_tag
```

Optional arguments:

* `--ipc=host`: Use host system's shared memory, since parallel torchrun consumes a large amount of shared memory. If not allowed by security policy, increase `--shm-size` ([documentation](https://docs.docker.com/engine/containers/run/#runtime-constraints-on-resources)).
* `-v /root/.cache:/root/.cache`: Mount host cache to avoid re-downloading cache entries.
* `-e HF_TOKEN="$HF_TOKEN"`: Set Hugging Face token to avoid re-authenticating.

If you get `docker: Error response from daemon: unknown or invalid runtime name: nvidia`, you need to [configure docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker):

```shell
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

</details>

## Downloading Checkpoints

Gamma-World pulls three sets of weights, all hosted on Hugging Face:

| Component | Hugging Face repo | Contents |
| --- | --- | --- |
| World-model networks | [`chijw/Gamma-World`](https://huggingface.co/chijw/Gamma-World) | `bidirectional/`, `causal/`, `causal-few-step/` network `model.safetensors` |
| VAE tokenizer | [`chijw/Gamma-World`](https://huggingface.co/chijw/Gamma-World) | `tokenizer.pth` |
| Text encoder (VLM) | [`nvidia/Cosmos-Reason1-7B`](https://huggingface.co/nvidia/Cosmos-Reason1-7B) | encodes the text prompt |

1. Get a [Hugging Face Access Token](https://huggingface.co/settings/tokens) with `Read` permission.
2. Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli): `uv tool install -U "huggingface_hub[cli]"`.
3. Login: `hf auth login`.
4. Accept the [NVIDIA Open Model License](https://huggingface.co/nvidia/Cosmos-Reason1-7B) on the Cosmos-Reason1-7B page.

The VAE and the text encoder download automatically on first run — `--vae` and `--text-encoder` default to their `hf://` URIs. The network is selected with `--checkpoint hf://chijw/Gamma-World/<mode>/model.safetensors`. Set [`HF_HOME`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome) to change where they are cached.

For an offline node, pre-download everything and pass local paths instead:

```bash
hf download chijw/Gamma-World        --local-dir ./checkpoints/Gamma-World
hf download nvidia/Cosmos-Reason1-7B --local-dir ./checkpoints/Cosmos-Reason1-7B
```

Then run with `--checkpoint ./checkpoints/Gamma-World/causal-few-step/model.safetensors --vae ./checkpoints/Gamma-World/tokenizer.pth --text-encoder ./checkpoints/Cosmos-Reason1-7B` (see [Inference](inference.md)).

## Troubleshooting

These errors surface when running `python scripts/check_environment.py` on a host without CUDA installed system-wide (i.e. relying on the pip-installed `nvidia-*-cu12` wheels that PyTorch pulls in).

### `ldconfig -p | grep 'libnvrtc'` returns non-zero

```
File ".../transformer_engine/common/__init__.py", line 111, in _load_nvrtc
    libs = subprocess.check_output("ldconfig -p | grep 'libnvrtc'", shell=True)
subprocess.CalledProcessError: Command 'ldconfig -p | grep 'libnvrtc'' returned non-zero exit status 1.
```

`transformer_engine._load_nvrtc()` first globs `$CUDA_HOME/**/libnvrtc.so*` and only falls back to `ldconfig` if `CUDA_HOME` is unset. Point `CUDA_HOME` at the venv's bundled CUDA libs:

```bash
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia"
```

### `libcublas.so.12: cannot open shared object file`

```
OSError: libcublas.so.12: cannot open shared object file: No such file or directory
```

The `transformer_engine` native `.so` depends on `libcublas`, `libcudnn`, `libnvJitLink`, etc. at `dlopen()` time. Add every `nvidia/*/lib` directory in the venv to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$(find $VIRTUAL_ENV/lib/python3.10/site-packages/nvidia -maxdepth 3 -type d -name lib | paste -sd:)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

### `GLIBC_2.34' not found` from `~/.triton/cache/.../cuda_utils.so`

```
ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found
  (required by /home/<user>/.triton/cache/.../cuda_utils.so)
```

`~/.triton/cache` contains a `cuda_utils.so` compiled against a newer glibc (e.g. from a previous run inside a container with Ubuntu 22+). On a host with older glibc the cached object is ABI-incompatible. Triton regenerates the cache on demand, so the fix is to drop it:

```bash
rm -rf ~/.triton
# Optional: redirect future Triton compilation cache off $HOME
export TRITON_CACHE_DIR=/path/to/triton_cache
```

Note: this host must still meet the glibc requirement in [System Requirements](#system-requirements) for the recompiled cache to load.
