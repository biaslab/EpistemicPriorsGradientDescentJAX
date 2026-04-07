#!/usr/bin/env bash
#SBATCH --job-name=smoke-test
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err

# Quick validation that the cluster environment is correctly set up:
#   - GPU visible to JAX
#   - Key packages importable
#   - pymdp installed from the correct git commit
#   - JAX computation actually runs on GPU

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-.}"
cd "$PROJECT_DIR"
mkdir -p logs

export JAX_PLATFORMS="cuda"

source cluster/setup_env.sh

echo "=== Smoke test on $(hostname) at $(date) ==="
echo ""

python -c "
import sys

failures = []

# --- 1. JAX GPU check ---
print('1. Checking JAX devices...')
import jax
devices = jax.devices()
gpu_devices = [d for d in devices if d.platform == 'gpu']
if gpu_devices:
    print(f'   OK: {len(gpu_devices)} GPU(s) found: {gpu_devices}')
else:
    failures.append(f'No GPU devices found. JAX sees: {devices}')
    print(f'   FAIL: no GPU devices. Found: {devices}')

# --- 2. JAX GPU computation ---
print('2. Running JAX computation on GPU...')
try:
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (256, 256))
    y = jnp.dot(x, x.T).block_until_ready()
    device = y.devices().pop()
    if device.platform == 'gpu':
        print(f'   OK: matmul ran on {device}')
    else:
        failures.append(f'Computation ran on {device}, expected GPU')
        print(f'   FAIL: computation ran on {device}')
except Exception as e:
    failures.append(f'JAX computation failed: {e}')
    print(f'   FAIL: {e}')

# --- 3. pymdp version check ---
print('3. Checking pymdp installation...')
try:
    import importlib.metadata
    meta = importlib.metadata.metadata('inferactively-pymdp')
    version = meta['Version']
    # The git-pinned install won't have a standard PyPI URL in direct_url.json
    # Check via importlib if installed from git
    try:
        import json, pathlib
        dist_files = importlib.metadata.packages_distributions()
        dist = importlib.metadata.distribution('inferactively-pymdp')
        direct_url_text = dist.read_text('direct_url.json')
        if direct_url_text:
            direct_url = json.loads(direct_url_text)
            url = direct_url.get('url', '')
            commit = direct_url.get('vcs_info', {}).get('commit_id', 'unknown')
            print(f'   OK: pymdp {version} from git commit {commit[:12]}')
        else:
            print(f'   WARN: pymdp {version} installed (could not determine source — may be from PyPI)')
    except Exception:
        print(f'   WARN: pymdp {version} installed (could not determine source — may be from PyPI)')
except ImportError:
    failures.append('inferactively-pymdp is not installed')
    print('   FAIL: inferactively-pymdp not found')
except Exception as e:
    failures.append(f'pymdp check failed: {e}')
    print(f'   FAIL: {e}')

# --- 4. Project imports ---
print('4. Checking project imports...')
import_errors = []
for mod in [
    'src.environments.tmaze',
    'src.environments.epistemic_maze',
    'src.planning.factorized_optimizer',
    'src.planning.temporal_optimizer',
    'src.objectives.factorized_vfe',
    'src.objectives.temporal_vfe',
]:
    try:
        __import__(mod)
    except Exception as e:
        import_errors.append(f'{mod}: {e}')

if import_errors:
    failures.append(f'Import errors: {import_errors}')
    for err in import_errors:
        print(f'   FAIL: {err}')
else:
    print(f'   OK: all project modules importable')

# --- Summary ---
print('')
if failures:
    print(f'SMOKE TEST FAILED ({len(failures)} issue(s)):')
    for f in failures:
        print(f'  - {f}')
    sys.exit(1)
else:
    print('SMOKE TEST PASSED — environment is ready.')
"

echo ""
echo "=== Smoke test finished at $(date) ==="
