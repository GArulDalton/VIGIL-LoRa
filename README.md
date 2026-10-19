# VIGIL-LoRa

**Conformally Calibrated Multiagent Reinforcement Learning for Spreading Factor and Power Scheduling in Batteryless LoRa Networks**

G. Arul Dalton and A. Bamila Virgin Louis

This repository contains the complete simulator, training code, evaluation scripts, pretrained models, raw results and figure scripts for the article above. Every table and figure in the paper can be regenerated from this repository.

## Overview

VIGIL-LoRa is a node-side uplink scheduler for batteryless, solar-harvesting LoRa sensors. In every 120 s slot, each node decides whether to transmit and which spreading factor (SF 7–12) and transmit power (2/8/14 dBm) to use. It combines:

- **On-sensor value of information (VoI):** each node runs a replica of the gateway's AR(1)/Kalman predictor and computes the normalized error the gateway currently incurs.
- **Constrained multiagent PPO:** one parameter-shared actor (6,227 parameters) trained at the network server with a centralized critic, Lagrangian dual ascent on the energy-outage constraint and Lyapunov energy-neutrality shaping. Execution is fully decentralized.
- **Energy-reserve shield:** masks any transmission that would leave the node unable to survive until sunrise (Proposition 1).
- **Conformal reserve calibration:** calibrates the shield margin from measured overnight drain so that night outage stays below a target ε even when the energy model is wrong (Proposition 2).

The simulator models LoRa time-on-air, the Petäjäjärvi path-loss model with shadowing, Rayleigh fading, imperfect SF orthogonality (Goursaud–Gorce SIR matrix), capture effect, multiple gateways, a 1% duty cycle, solar harvesting with a cloud process and supercapacitor storage.

## Repository structure

| File / folder | Content |
|---|---|
| `env.py` | LoRa energy-harvesting network simulator (PHY, collisions, energy, duty cycle, remote estimation, shield) |
| `nn.py` | NumPy MLP with manual back-propagation and Adam |
| `vigil.py` | VIGIL-LoRa training (constrained MAPPO) and ablation variants |
| `dqn.py` | Parameter-shared DQN baseline |
| `evaluate.py` | Baseline policies (ADR, EH-ADR, AoI-greedy, VoI-threshold, Random) and the evaluation loop |
| `experiments.py` | Scalability, multi-gateway, ablation and trace experiments |
| `exp2.py` | Conformal reserve calibration, paired Wilcoxon tests and runtime measurements |
| `summarize.py`, `figs.py`, `fig_conf.py` | Summary tables and all figures |
| `quick_demo.py` | One-minute demo with the pretrained model |
| `models/` | Pretrained VIGIL-LoRa, ablation and DQN models |
| `results/` | Raw results (JSON) used for every table in the paper |
| `fig/` | All figures of the paper |
| `scripts/` | Full reproduction scripts |

## Installation

Python 3.9 or later; no GPU or deep-learning framework is needed.

```bash
git clone https://github.com/GArulDalton/VIGIL-LoRa.git
cd VIGIL-LoRa
pip install -r requirements.txt
```

## Quick start

```bash
python quick_demo.py
```

Evaluates the pretrained policy, ADR and VoI-threshold on a 200-node network for one day (about one minute on a laptop).

## Reproducing the paper

All commands are run from the repository root.

| Goal | Command | Time (1 CPU core) |
|---|---|---|
| Re-run all experiments with the pretrained models | `bash scripts/evaluate_only.sh` | about 40 min |
| Retrain every model and re-run everything | `bash scripts/reproduce_all.sh` | about 2–3 h |
| Train only the main model | `python vigil.py vigil 150` | about 3.5 min |
| Train an ablation variant | `python vigil.py no_voi 150` (also `no_lyap`, `no_constr`, `no_central`, `ideal_orth`, `no_shield`) | about 3.5 min each |
| Train the DQN baseline | `python dqn.py` | about 4 min |
| One experiment | `python experiments.py scale` (`abl`, `gw`, `trace`) or `python exp2.py conformal` (`wilcoxon`, `runtime`) | minutes |

Mapping to the paper: Table 3 and Figs. 3–9 come from `experiments.py scale`; Table 4 from `exp2.py wilcoxon`; Table 5 from `exp2.py runtime`; Table 6 from `experiments.py gw`; Tables 7–8 and Fig. 11 from `exp2.py conformal`; Table 9 and Fig. 12 from `experiments.py abl`; Fig. 10 from `experiments.py trace`; Fig. 2 from the training history stored in `models/vigil.pkl`.

All random seeds are fixed, so results are reproducible up to small floating-point differences between platforms.

## Main results (N = 500 nodes, one gateway, mean over three topologies)

| Method | NMSE ↓ | Outage (%) ↓ | PDR | Detection delay (min) ↓ | Tx energy (J/node/day) ↓ |
|---|---|---|---|---|---|
| ADR | 0.858 | 18.2 | 0.868 | 9.33 | 3.01 |
| EH-ADR | 0.927 | 1.3 | 0.869 | 9.40 | 2.62 |
| AoI-greedy | 0.890 | 7.1 | 0.870 | 9.37 | 2.65 |
| VoI-threshold | 0.633 | 4.9 | 0.857 | 0.84 | 1.41 |
| Random | 1.080 | 43.0 | 0.499 | 13.41 | 2.55 |
| DQN | 0.740 | 0.0 | 0.424 | 2.96 | 4.04 |
| VIGIL-LoRa | 0.409 | 0.0 | 0.570 | 0.75 | 1.53 |

VIGIL-LoRa has a lower NMSE than every baseline on all ten topologies of the paired test (one-sided Wilcoxon p = 9.8e-04).

**Conformal reserve under drain mismatch** (night outage of nodes entering the night above the reserve; target ε = 5%):

| Max. drain mismatch κ_max | Model-based shield (μ = 1.2) | Conformal shield | Calibrated μ̂ |
|---|---|---|---|
| 1.5 | 13.7% | 0.6% | 1.51 |
| 2.0 | 43.8% | 2.0% | 1.96 |
| 2.5 | 60.0% | 3.1% | 2.41 |

## Notes and limitations

All data are generated by simulation; the monitored process and solar input are synthetic models described in the paper. On-device inference cost was not measured on microcontroller hardware.

## Citation

If you use this code, please cite the article (details will be added after publication). See `CITATION.cff`.

## License

MIT License (see `LICENSE`).

## Contact

G. Arul Dalton, Department of Computer Science and Engineering, Saveetha Engineering College, Chennai, India — aruldalton@saveetha.ac.in
