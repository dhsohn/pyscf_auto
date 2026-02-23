#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/pyscf_runs}"
echo "=== Preflight Check ==="

# 1. Running processes (exclude this script itself and parent shell)
echo "[1/4] Checking running processes..."
RUNNING=$(pgrep -af 'pyscf_auto|python.*cli_new|run_opt.py' 2>/dev/null | grep -v "preflight_check" | grep -v "$$" || true)
if [ -n "$RUNNING" ]; then
  echo "  FAIL: pyscf_auto-related processes are running. Abort cutover."
  echo "$RUNNING"
  exit 1
fi
echo "  OK: No running processes."

# 2. Stale locks
echo "[2/4] Checking stale locks..."
LOCKS=$(find "$ROOT" -name 'run.lock' 2>/dev/null || true)
if [ -n "$LOCKS" ]; then
  echo "  WARN: Found lock files:"
  echo "$LOCKS"
  echo "  Review and remove stale locks before proceeding."
else
  echo "  OK: No lock files."
fi

# 3. In-progress states
echo "[3/4] Checking in-progress states..."
IN_PROGRESS=$(find "$ROOT" -name 'run_state.json' -exec grep -l '"status": "r' {} \; 2>/dev/null || true)
if [ -n "$IN_PROGRESS" ]; then
  echo "  WARN: Found running/retrying states:"
  echo "$IN_PROGRESS"
  echo "  Complete or stop these runs before cutover."
else
  echo "  OK: No running/retrying states."
fi

# 4. Disk space
echo "[4/4] Disk space check..."
df -h "$HOME"
echo "=== Preflight Complete ==="
