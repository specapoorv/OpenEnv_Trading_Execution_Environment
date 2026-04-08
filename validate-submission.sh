#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACE_URL="${1:-${SPACE_URL:-}}"
REPO_DIR="${2:-$ROOT}"

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: repo_dir '%s' not found\n" "${2:-$ROOT}" >&2
  exit 1
fi

if [ -z "$SPACE_URL" ]; then
  printf "Usage: %s <space_url> [repo_dir]\n" "$0" >&2
  printf "Example: %s https://specapoorv-trading-execution-environment.hf.space\n" "$0" >&2
  exit 1
fi

SPACE_URL="${SPACE_URL%/}"

run_python_validator() {
  if command -v uv >/dev/null 2>&1; then
    SPACE_URL="$SPACE_URL" RUN_DOCKER_CHECK=0 uv run python server/core/validate_submission.py
    return
  fi
  SPACE_URL="$SPACE_URL" RUN_DOCKER_CHECK=0 python3 server/core/validate_submission.py
}

run_openenv_validate() {
  if command -v uv >/dev/null 2>&1; then
    uv run openenv validate
    return
  fi
  openenv validate
}

check_docker_access() {
  if ! command -v docker >/dev/null 2>&1; then
    printf "docker is not installed or not on PATH\n" >&2
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    printf "docker is installed but the daemon is not accessible\n" >&2
    printf "Start Docker or run with permission to access /var/run/docker.sock\n" >&2
    exit 1
  fi
}

cd "$REPO_DIR"

printf "== validate-submission ==\n"
printf "repo: %s\n" "$REPO_DIR"
printf "space: %s\n" "$SPACE_URL"

printf "\n[1/3] Checking app contract and live Space endpoints...\n"
run_python_validator

printf "\n[2/3] Running docker build...\n"
check_docker_access
docker build -t execution-desk-check .

printf "\n[3/3] Running openenv validate...\n"
run_openenv_validate

printf "\nAll checks passed.\n"
