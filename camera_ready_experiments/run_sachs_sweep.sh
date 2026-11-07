#!/bin/zsh
# Fig. 4 sweep for JCI and Psi-FCI on Sachs (alpha=0.05 as IGSP in Fig. 4, 30 draws as in the caption).
# One process per (pool, size) so the Psi-FCI runs proceed in parallel.
cd "$(dirname "$0")"
mkdir -p logs results/sachs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SQ=(cd3cd28 cd3cd28+u0126 cd3cd28+psitect cd3cd28+ly cd3cd28+aktinhib cd3cd28+g0076)
for size in 200 500 1000 2000 4000 all; do
  ${PYTHON:-python} run_sachs.py --data-dir data/sachs_fig4 --reference cd3cd28 --conditions $SQ \
     --methods jci psifci --alpha 0.05 --sizes $size --n-boot 30 \
     --out results/sachs/squires_n$size.csv > logs/sachs_squires_n$size.log 2>&1 &
  ${PYTHON:-python} run_sachs.py --data-dir data/sachs_fig4 --reference cd3cd28 \
     --methods jci psifci --alpha 0.05 --sizes $size --n-boot 30 \
     --out results/sachs/full9_n$size.csv > logs/sachs_full9_n$size.log 2>&1 &
done
# Known-target baselines (GIES, IGSP) on the 5-condition subset, same draws as above
${PYTHON:-python} run_sachs.py --data-dir data/sachs_fig4 --reference cd3cd28 --conditions $SQ \
   --methods gies igsp --alpha 0.05 --sizes 200 500 1000 2000 4000 all --n-boot 30 \
   --out results/sachs/squires_labelaware.csv > logs/sachs_squires_labelaware.log 2>&1 &
${PYTHON:-python} run_sachs.py --data-dir data/sachs_fig4 --methods ica_lingam_castle \
   --sizes 200 500 1000 2000 4000 all --n-boot 30 --out results/sachs/full9_icalingam.csv > logs/sachs_icalingam.log 2>&1 &
wait
