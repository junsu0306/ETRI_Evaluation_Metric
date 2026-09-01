#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

if [[ ! -x .venv/bin/python ]]; then
  if ! python3 -m venv .venv; then
    # Debian may create a partial environment and then fail because ensurepip
    # is unavailable.  Complete the standard venv scripts without pip; pip is
    # bootstrapped explicitly below.
    python3 -m venv --without-pip --upgrade .venv
  fi
fi

if [[ ! -f .venv/bin/activate ]]; then
  venv_stdlib="$(python3 -c 'import sysconfig; print(sysconfig.get_path("stdlib"))')"
  venv_activate_template="${venv_stdlib}/venv/scripts/common/activate"
  if [[ ! -f "${venv_activate_template}" ]]; then
    printf 'Missing venv activation template: %s\n' "${venv_activate_template}" >&2
    exit 1
  fi
  sed \
    -e "s|__VENV_DIR__|${project_root}/.venv|g" \
    -e 's|__VENV_BIN_NAME__|bin|g' \
    -e "s|__VENV_PROMPT__|'(etri-robot-compression) '|g" \
    "${venv_activate_template}" > .venv/bin/activate
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
