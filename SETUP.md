# GOPT Setup Guide (Python 3.9 + `venv` + CPU smoke run)

This guide shows the exact process to get this repo working without Conda, using a standard Python virtual environment.

It targets:

- Python `3.9`
- CPU only
- no rendering
- a short smoke-training run

It does **not** cover:

- CUDA / GPU setup
- `--render` / VTK
- full training with the default long config

## 1. Prerequisites

You need:

- Python `3.9.x`
- `pip`
- a shell

Do **not** use Python `3.14` for this repo with the current dependency versions. The pinned package stack here was built for an older Python version and is likely to fail on 3.14.

## 2. Go to the repo

```bash
cd /path/to/GOPT
```

## 3. Create a virtual environment

```bash
python3.9 -m venv venv
```

## 4. Activate it

On macOS / Linux:

```bash
source venv/bin/activate
```

After activation, verify the Python version:

```bash
python -V
```

It should print something like:

```bash
Python 3.9.x
```

If it does not show Python 3.9, stop here and fix that first.

## 5. Upgrade packaging tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 6. Install PyTorch

### CPU

PyTorch is required by the code, but it is **not** listed in `requirements.txt`, so install it first:

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

### CUDA

If you want to use CUDA instead, do **not** use the CPU command above. On a Linux or Windows machine with an NVIDIA GPU, install the CUDA 12.1 build instead:

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

Notes:

- after installing the CUDA build, run training and testing without `--no-cuda`

## 7. Install the remaining dependencies needed for the smoke run

This smoke run does **not** need rendering, so we intentionally skip `vtk`.

The codebase has been adjusted so that `vtk` is only required when you actually use rendering. In other words, plain training and evaluation on CPU now work without installing `vtk`.

```bash
pip install scipy==1.11.3 matplotlib==3.8.0 gymnasium==0.29.1 "tensorboard>=2.5.0" "numba>=0.51.0" "h5py>=2.10.0" tqdm omegaconf
```

## 8. Verify imports

Run this import smoke test:

```bash
python - <<'PY'
import torch
import gymnasium
import scipy
import matplotlib
import numba
import h5py
import omegaconf
import tianshou
print("torch", torch.__version__)
print("imports ok")
PY
```

Expected result:

- no import errors
- `torch 2.1.0`
- `imports ok`

## 9. Create the smoke config

This repo does not include a smoke config by default. Use the file `cfg/smoke.yaml` with the contents shown below.

The purpose of this config is to make the first run short and lightweight:

- `num_processes: 1` avoids spawning many subprocesses
- `epoch: 1` keeps training short
- `step_per_epoch: 8` keeps the run tiny
- `batch_size: 8` fits the smoke run

## 10. Run an evaluation-path sanity check

Before training, make sure the code can parse config, register the environment, and build the model:

```bash
python ts_test.py --config cfg/smoke.yaml --no-cuda --ckp /tmp/does-not-exist.pth
```

Expected result:

- the script starts normally
- it ends by printing:

```bash
No model found
```

That is expected here. It means setup, imports, config loading, and model/env creation are working.

If you skipped `vtk`, this command should still work because non-render execution no longer imports the renderer eagerly.

## 11. Run the smoke training job

```bash
python ts_train.py --config cfg/smoke.yaml --no-cuda
```

Expected result:

- training starts on CPU
- a new directory is created under `logs/`
- the short run completes
- a final checkpoint file is saved

During the smoke run you may also see warnings like these:

- `PyparsingDeprecationWarning` from `matplotlib`
- a Gymnasium warning about `Env.reset` accepting `options`

These warnings are non-fatal for the smoke run. The training job can still complete successfully.

## 12. Verify the outputs

After the smoke run, check the `logs/` directory.

You should see a newly created run directory containing files such as:

- `policy_step_final.pth`
- a copy of the config
- a copy of `model.py`
- a copy of `arguments.py`

You should also see training output ending with a successful evaluation summary, for example:

```bash
Finished training!
Setup test envs ...
Testing agent ...
The result (over 1000 episodes): ...
```

## 13. Next step

Once the smoke run works, you can try the full config:

```bash
python ts_train.py --config cfg/config.yaml --no-cuda
```

If you later want rendering, that is a separate setup step because it requires `vtk`.

## 13.5 Optional: running with CUDA

If you are on Linux or Windows with an NVIDIA GPU and installed the CUDA PyTorch build from step 6, you can run the same commands without `--no-cuda`.

Examples:

```bash
python ts_train.py --config cfg/local.yaml --device 0
```

```bash
python ts_test.py --config cfg/local.yaml --device 0 --ckp /path/to/policy_step_final.pth --test-episode 20
```

To verify that PyTorch can see your GPU:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY
```

Expected result on a working CUDA machine:

- `cuda available: True`
- at least one visible CUDA device

## 14. Optional: rendering a saved policy

Rendering is only used in evaluation, not in training. To render a packing episode, you need to install the optional `vtk` dependency first.

Install it with:

```bash
pip install vtk==9.0.2
```

Then run evaluation with `--render` and a trained checkpoint:

```bash
python ts_test.py --config cfg/local.yaml --no-cuda --ckp /path/to/policy_step_final.pth --render
```

You can also use a different config file if needed, for example `cfg/smoke.yaml`.

Important behavior:

- `--render` is for `ts_test.py`, not `ts_train.py`
- when `--render` is enabled, the code forces the evaluation to exactly `1` episode
- this means `--test-episode` is ignored when `--render` is used

Without `--render`, you can choose the number of evaluation episodes manually, for example:

```bash
python ts_test.py --config cfg/local.yaml --no-cuda --ckp /path/to/policy_step_final.pth --test-episode 20
```

Notes:

- rendering is much slower than non-render evaluation
- the window is created by VTK, so behavior can depend on your machine and windowing system
- if `vtk` is not installed, rendering will fail with a missing-module error
