#!/usr/bin/env bash
# Run every registered model over every month-end origin of the FULL panel,
# then the combinations (which read the other models' stored forecasts).
# Results go to results/full. Usage: scripts/run_all.sh [jobs]
# The interpreter is $ASSETFORECAST_PYTHON if set (the pipeline passes its own,
# so every step imports the same copy of the package), else python3.
set -uo pipefail
cd "$(dirname "$0")/.."
unset ASSETFORECAST_MAX_DATE ASSETFORECAST_DROP_TIERS
export OMP_NUM_THREADS=1 ASSETFORECAST_RESULTS="$PWD/results/full"
JOBS="${1:-4}"
PY="${ASSETFORECAST_PYTHON:-python3}"
"$PY" -m assetforecast.eval.run targets
for kind in vol corr cov; do
  base=$("$PY" -c "from assetforecast.models import list_models; print(' '.join(m for m in list_models('$kind') if not m.startswith('combo_')))")
  "$PY" -m assetforecast.eval.run forecast --kind "$kind" --model $base --jobs "$JOBS" --force
done
for kind in vol corr cov; do
  combo=$("$PY" -c "from assetforecast.models import list_models; print(' '.join(m for m in list_models('$kind') if m.startswith('combo_')))")
  "$PY" -m assetforecast.eval.run forecast --kind "$kind" --model $combo --jobs "$JOBS" --force
done
