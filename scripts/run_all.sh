#!/usr/bin/env bash
# Run every registered model over every month-end origin of the FULL panel,
# then the combinations (which read the other models' stored forecasts).
# Results go to results/full. Usage: scripts/run_all.sh [jobs]
set -uo pipefail
cd "$(dirname "$0")/.."
unset VOLMODEL_MAX_DATE VOLMODEL_DROP_TIERS
export OMP_NUM_THREADS=1 VOLMODEL_RESULTS="$PWD/results/full"
JOBS="${1:-4}"
python3 -m volmodel.eval.run targets
for kind in vol corr cov; do
  base=$(python3 -c "from volmodel.models import list_models; print(' '.join(m for m in list_models('$kind') if not m.startswith('combo_')))")
  python3 -m volmodel.eval.run forecast --kind "$kind" --model $base --jobs "$JOBS" --force
done
for kind in vol corr cov; do
  combo=$(python3 -c "from volmodel.models import list_models; print(' '.join(m for m in list_models('$kind') if m.startswith('combo_')))")
  python3 -m volmodel.eval.run forecast --kind "$kind" --model $combo --jobs "$JOBS" --force
done
