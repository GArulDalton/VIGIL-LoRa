#!/bin/bash
# Re-run all experiments with the pretrained models in models/ (no training; about 40 minutes).
set -e
mkdir -p results fig logs
for p in abl trace gw scale; do python experiments.py $p | tee logs/eval_$p.log; done
for p in conformal wilcoxon runtime; do python exp2.py $p | tee logs/exp2_$p.log; done
python summarize.py && python figs.py && python fig_conf.py
