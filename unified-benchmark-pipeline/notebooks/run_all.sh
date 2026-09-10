#!/usr/bin/env bash
# Headless-execute all walkthrough notebooks (light steps + cached renders only;
# SAYF_NB_REGEN=0, so no GPU/API). Fails if any cell errors.
#
# Run on a compute node (the login node blocks installs / heavy I/O), e.g.:
#   srun -p cpu-all --mem 8G --time 00:30:00 --pty bash -c \
#     'cd unified-benchmark-pipeline/notebooks && ./run_all.sh'
set -uo pipefail

cd "$(dirname "$0")"
export SAYF_NB_REGEN="${SAYF_NB_REGEN:-0}"
export MPLBACKEND=Agg
PY="${PY:-python}"
OUTDIR="${OUTDIR:-.}"   # execute in place: the notebooks/ files ARE the single executed set

"$PY" build.py   # regenerate from build.py (the cell source of truth), then execute in place
rc=0
for nb in 0[0-6]_*.ipynb; do
    echo "== executing $nb =="
    "$PY" -m jupyter nbconvert --to notebook --execute \
        --output-dir "$OUTDIR" --output "$nb" \
        --ExecutePreprocessor.timeout=1800 "$nb" || rc=1
done
echo "run_all: $([ $rc = 0 ] && echo PASS || echo FAIL rc=$rc)   (notebooks executed in place)"
exit $rc
