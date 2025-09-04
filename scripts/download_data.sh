#!/usr/bin/env bash
set -euo pipefail

QSCALED_ZIP_PATH="${HOME}/.value-scaling2/zip"
BASE_URL="https://value-scaling.github.io"  # no trailing slash
mkdir -p "${QSCALED_ZIP_PATH}"

filenames=(
  "compute_optimal.zip"
  "dmc_bro_ablations.zip"
  "extrapolated_bs.zip"
  "fitted_bs.zip"
  "interpolated_bs.zip"
  "model_scaling_const_lr.zip"
  "n_scaling_bl.zip"
  "overfitting.zip"
  "side_critic.zip"
  "simbav2_model_scaling_linear20_with_base.zip"
  "utd_scaling_bl.zip"
)

for filename in "${filenames[@]}"; do
  url="${BASE_URL}/data/${filename}"
  out="${QSCALED_ZIP_PATH}/${filename}"

  if wget --no-verbose --show-progress --progress=bar:force:noscroll -O "${out}.partial" "${url}"; then
    mv "${out}.partial" "${out}"
    echo "saved  ${filename}"
  else
    rm -f "${out}.partial"
    echo "failed ${filename}" >&2
  fi
done
