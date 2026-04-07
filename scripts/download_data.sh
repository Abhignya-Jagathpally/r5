#!/usr/bin/env bash
# Download real public Multiple Myeloma datasets used by this pipeline.
# Data is intentionally NOT committed to git (see .gitignore).
#
# Usage:
#   bash scripts/download_data.sh           # download all available open datasets
#   bash scripts/download_data.sh --tcia    # also print TCIA/dbGaP retrieval steps
#
# Datasets handled here:
#   - LabIA-UFBA Multiple-Myeloma-Dataset (open, GitHub, direct clone)
#   - MiMM_SBILab  (TCIA, open access, NBIA Data Retriever)
#   - SN-AM        (TCIA, open access, NBIA Data Retriever)
#   - CMB-MML      (TCIA + dbGaP, restricted — manual step required)
#
# See docs/dataset_catalog.md for licenses, citations, and access notes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/raw"
mkdir -p "${RAW_DIR}"

echo "[download_data] target directory: ${RAW_DIR}"

# ---- 1. LabIA-UFBA Multiple Myeloma Dataset (direct, open) -------------------
LABIA_DIR="${RAW_DIR}/labia_ufba_mm"
if [ ! -d "${LABIA_DIR}/.git" ]; then
  echo "[download_data] cloning LabIA-UFBA Multiple-Myeloma-Dataset..."
  git clone --depth 1 https://github.com/LabIA-UFBA/Multiple-Myeloma-Dataset.git "${LABIA_DIR}"
else
  echo "[download_data] LabIA-UFBA dataset already present, pulling latest..."
  git -C "${LABIA_DIR}" pull --ff-only || true
fi

# ---- 2. TCIA open-access collections (MiMM_SBILab, SN-AM) --------------------
# TCIA distributes imaging via NBIA Data Retriever using .tcia manifest files.
# We fetch manifests programmatically; the user runs NBIA Data Retriever (or
# the `nbia-data-retriever` CLI) to materialize the actual DICOM/SVS files.
TCIA_DIR="${RAW_DIR}/tcia_manifests"
mkdir -p "${TCIA_DIR}"

fetch_manifest () {
  local name="$1"
  local url="$2"
  local out="${TCIA_DIR}/${name}.tcia"
  if [ ! -f "${out}" ]; then
    echo "[download_data] fetching TCIA manifest: ${name}"
    curl -fsSL "${url}" -o "${out}" || {
      echo "  (manifest fetch failed — visit the TCIA collection page and download the .tcia file manually into ${TCIA_DIR}/)"
      rm -f "${out}"
    }
  fi
}

# Public TCIA collection landing pages (manifest URLs change; users may need
# to grab the latest manifest from the collection page if these 404):
fetch_manifest "MiMM_SBILab" "https://www.cancerimagingarchive.net/wp-content/uploads/MiMM_SBILab.tcia" || true
fetch_manifest "SN-AM"       "https://www.cancerimagingarchive.net/wp-content/uploads/SN-AM.tcia"       || true

cat <<EOF

[download_data] Next steps for TCIA imaging:
  1. Install NBIA Data Retriever: https://wiki.cancerimagingarchive.net/x/2QKPBQ
  2. Open the .tcia manifests in ${TCIA_DIR}/ with NBIA Data Retriever
  3. Choose ${RAW_DIR} as the destination directory
  4. For CMB-MML (restricted): apply via dbGaP study phs002192, then use the
     authorized .tcia manifest from the CMB-MML collection page.

All raw files land under data/raw/ which is gitignored. The pipeline reads
from data/raw/ — once downloaded you can run:

  snakemake --cores 8 --configfile configs/pipeline.yaml

EOF

echo "[download_data] done."
