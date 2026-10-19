#!/bin/bash
# Full reproduction of every table and figure (single CPU core: about 2 to 3 hours).
# Run from the repository root:  bash scripts/reproduce_all.sh
set -e
mkdir -p models results fig logs
echo "[1/5] Training VIGIL-LoRa, ablation variants and DQN baseline"
for v in vigil no_voi no_lyap no_constr no_central ideal_orth no_shield; do
  python vigil.py $v 150 | tee logs/train_$v.log
done
python dqn.py | tee logs/train_dqn.log
echo "[2/5] Main experiments (ablation, trace, multi-gateway, scalability)"
for p in abl trace gw scale; do python experiments.py $p | tee logs/eval_$p.log; done
echo "[3/5] Conformal reserve calibration, paired Wilcoxon tests, runtime"
for p in conformal wilcoxon runtime; do python exp2.py $p | tee logs/exp2_$p.log; done
echo "[4/5] Summary tables"
python summarize.py | tee logs/summary.log
echo "[5/5] Figures"
python figs.py && python fig_conf.py
echo "Done. Tables: results/summary.json and results/res_*.json; figures: fig/"
