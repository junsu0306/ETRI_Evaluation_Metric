#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

if [[ ! -x .venv/bin/python ]]; then
  if ! python3 -m venv .venv; then
    if [[ ! -x .venv/bin/python ]]; then
      printf '%s\n' 'Failed to create .venv and no Python executable was produced.' >&2
      exit 1
    fi
  fi
fi

if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  bootstrap="$(mktemp)"
  trap 'rm -f "${bootstrap}"' EXIT
  curl --fail --location https://bootstrap.pypa.io/get-pip.py --output "${bootstrap}"
  .venv/bin/python "${bootstrap}"
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install --requirement requirements.txt
.venv/bin/python -m pip install --editable .

printf '%s\n' 'Environment ready. Activate it with: source .venv/bin/activate'
