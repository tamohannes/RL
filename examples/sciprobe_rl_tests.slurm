#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Account comes from SBATCH_ACCOUNT in the submit environment.
#SBATCH --job-name=sciprobe_rl_tests
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#
# In-container pytest gate for the SciProbe NeMo-RL work.
#
# The sciprobe tests import nemo_rl and omegaconf, so they cannot run on a laptop
# without a full project environment. This runs them inside the same image the
# canaries train in, against a staged checkout.
#
# Usage:
#   CODE_DIR=<staged checkout on lustre> \
#   RUN_ID=r1 \
#   sbatch examples/sciprobe_rl_tests.slurm
#
set -euo pipefail

PERSISTENT_ROOT="${PERSISTENT_ROOT:?set PERSISTENT_ROOT to the sciprobe lustre root}"
CODE_DIR="${CODE_DIR:?set CODE_DIR to the staged NeMo-RL checkout}"
RUN_ID="${RUN_ID:-r1}"
CONTAINER="${CONTAINER:-${PERSISTENT_ROOT}/nemo-rl/images/nemo-rl-main-a9529052-lightning-ns-tools-sciprobe-r2.sqsh}"

test -d "${CODE_DIR}" || { echo "CODE_DIR not found: ${CODE_DIR}" >&2; exit 1; }
test -f "${CONTAINER}" || { echo "CONTAINER not found: ${CONTAINER}" >&2; exit 1; }

srun \
  --ntasks=1 \
  --cpus-per-task=32 \
  --exact \
  --kill-on-bad-exit=1 \
  --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts="${CODE_DIR}:/workspace/RL" \
  bash -lc '
    set -euo pipefail
    export PATH=/opt/uv/bin:/opt/nemo_rl_venv/bin:${PATH}
    export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
    export UV_PYTHON_INSTALL_DIR=/opt/uv-python
    export NEMO_RL_VENV_DIR=/opt/ray_venvs
    gym_actor_python="${NEMO_RL_VENV_DIR}/nemo_rl.environments.nemo_gym.NemoGym/bin/python"
    test_layer=/tmp/sciprobe-rl-pytest
    test -x "${gym_actor_python}"
    uv pip install \
      --python "${gym_actor_python}" \
      --target "${test_layer}" \
      orjson \
      pytest==9.0.3 \
      pytest-asyncio \
      pytest-cov \
      pytest-shard \
      pytest-testmon \
      pytest-timeout
    pythonpath="${test_layer}:/workspace/RL:/workspace/RL/examples/nemo_gym_extensions:/workspace/RL/3rdparty/Gym-workspace/Gym:${PYTHONPATH:-}"
    cd /workspace/RL
    "${gym_actor_python}" --version
    env PYTHONPATH="${pythonpath}" "${gym_actor_python}" -c "import nemo_gym, openai; from nemo_gym.server_utils import ServerClient"
    echo "[SCIPROBE_RL_TESTS_START]"
    env PYTHONPATH="${pythonpath}" "${gym_actor_python}" -m pytest -q --tb=short \
      -k "not test_vllm_policy_generation" \
      tests/test_nemo_gym_identity.py \
      tests/test_nemo_gym_loopback.py \
      tests/unit/environments/test_nemo_gym.py \
      tests/unit/experience/test_rollout_manager.py \
      tests/unit/models/generation/test_generation_router.py \
      tests/unit/models/generation/test_vllm_generation.py \
      tests/test_sciprobe_gym_prefetch.py \
      tests/test_sciprobe_network_proc_listeners.py \
      tests/test_sciprobe_ray_loopback.py \
      tests/test_sciprobe_sandbox_loopback.py \
      tests/test_sciprobe_sandbox_seccomp.py \
      tests/test_sciprobe_secret_logging.py \
      tests/test_sciprobe_signal_capability_config.py \
      examples/nemo_gym_extensions/tests/test_sciprobe_capability.py
    echo "[SCIPROBE_RL_TESTS_OK]"
  '
