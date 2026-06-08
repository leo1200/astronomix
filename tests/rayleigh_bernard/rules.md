Your are only allowed to use free GPUs and CPU usage should be minimal.

To only use one GPU, you might use

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

before you import JAX. JAX will otherwise grab all GPUs no
matter if they are free or not.