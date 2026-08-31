#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="${PITTSBURGH_ROOT:-${project_root}/data/datasets/pittsburgh}"
download_root="${PITTSBURGH_DOWNLOAD_ROOT:-${project_root}/data/downloads/pittsburgh}"
base_url="https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Pittsburgh250k"

mkdir -p "${dataset_root}/images" "${download_root}"

image_files=(000.tar 001.tar 002.tar 003.tar 004.tar 005.tar 006.tar)
if [[ "${PITTSBURGH_FULL:-0}" == "1" ]]; then
  image_files+=(007.tar 008.tar 009.tar 010.tar)
fi
files=("${image_files[@]}" queries_real.tar netvlad_v100_datasets.tar.gz md5sums.txt)

for name in "${files[@]}"; do
  printf 'Downloading %s\n' "${name}"
  curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
    --output "${download_root}/${name}" "${base_url}/${name}"
done

printf '%s\n' 'Verifying available checksums'
(
  cd "${download_root}"
  md5sum --ignore-missing --check md5sums.txt
)

for name in "${image_files[@]}"; do
  marker="${dataset_root}/.extracted-${name}"
  if [[ ! -f "${marker}" ]]; then
    printf 'Extracting %s\n' "${name}"
    tar -xf "${download_root}/${name}" -C "${dataset_root}/images"
    touch "${marker}"
  fi
done

if [[ ! -f "${dataset_root}/.extracted-queries_real.tar" ]]; then
  printf '%s\n' 'Extracting queries_real.tar'
  tar -xf "${download_root}/queries_real.tar" -C "${dataset_root}"
  touch "${dataset_root}/.extracted-queries_real.tar"
fi

if [[ ! -f "${dataset_root}/.extracted-netvlad_v100_datasets.tar.gz" ]]; then
  printf '%s\n' 'Extracting dataset specifications'
  tar -xzf "${download_root}/netvlad_v100_datasets.tar.gz" -C "${dataset_root}"
  touch "${dataset_root}/.extracted-netvlad_v100_datasets.tar.gz"
fi

printf 'Pittsburgh dataset ready at %s\n' "${dataset_root}"
