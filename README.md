# Q-LoRa: energy-efficient SF allocation via multi-agent Q-learning

Official code for the simulation study described in the accompanying paper (see
[Citation](#citation)): a parameter-driven command-line simulator that stores its results
as CSV:

```
simulate.py  →  CSV + Q-table (.npy) in data/
```

Each end device runs a lightweight tabular Q-learning agent that picks its spreading
factor (SF7–SF12). The reward trades off Packet Delivery Ratio (PDR) against normalised
time-on-air (energy). Two reward designs are supported, plus baseline strategies for
comparison.

## Contents

| File | Role |
|---|---|
| `lora_sim.py` | Simulation core: LoRa physical layer, collision model, `LoRaNetwork` env. |
| `simulate.py` | CLI to **train** a policy and **eval** strategies; writes CSV + the Q-table. |

`lora_sim.py` provides one environment with two reward designs plus an ADR baseline:

- **Cooperative** — shared, network-wide reward (Eq. 2): `r = PDR − β·(ToA−ToA_SF7)/(ToA_SF12−ToA_SF7)`.
- **Competitive** — per-node reward (Eq. 1), same shape computed per device.
- **ADR** — a distance/SNR-based Adaptive Data Rate controller, used as an evaluation baseline.

Select the reward with `--mode {cooperative,competitive}`; ADR is a `--strategies` choice.

## Install

```bash
pip install -r requirements.txt
```

Python ≥ 3.9, NumPy, SciPy, Pandas. The environment uses a plain NumPy RNG and needs no RL
framework (no `gym`). `wandb` is optional and only used by `simulate.py train --wandb`.

## Quick start

```bash
# 1. Train the cooperative policy (N=8 devices, T=4 threshold levels)
python simulate.py train -n 8 -t 4 --episodes 4000 --max-steps 1000 --out data

# 2. Evaluate every strategy over 20 seeds
python simulate.py eval  -n 8 -t 4 --seeds 20 --max-steps 1000 --out data
```

## `simulate.py`

Common options (both subcommands): `-n/--agents N`, `-t/--threshold T`,
`--mode {cooperative,competitive}`, `--max-steps`, `--period`, `--tx-prob`, `--beta`,
`--loc-range`, `--seed`, `--out`, `--verbose`.

### `train`
Runs ε-greedy tabular Q-learning and saves the learned Q-table plus per-episode metrics.

Extra options: `--episodes`, `--lr`, `--lr-schedule {none,step}`, `--gamma`, `--eps-max`,
`--eps-min`, `--decay`, `--init-sf`, `--wandb`.

Outputs:
- `data/q_table_<mode>_n{N}_t{T}.npy` — learned Q-table (model artefact).
- `data/training_<mode>_n{N}_t{T}.csv` — `episode,total_reward,avg_reward,pdr,toa,epsilon,lr,visited_pct`.

### `eval`
Evaluates strategies over seeds `range(0, --seeds)` (or an explicit `--seeds-list`) and
aggregates per-seed means with 95% t-confidence intervals.

Extra options: `--strategies {min,max,random,smart,adr}...` (default: all),
`--seeds`, `--seeds-list`, `--q-table`.

Strategies: `min`=fixed SF7, `max`=fixed SF12, `random`=random SF each round,
`smart`=learned Q-table (needs a trained `q_table_*.npy`), `adr`=ADR controller.

Outputs:
- `data/eval_<mode>_n{N}_t{T}.csv` — one row per strategy: `pdr_*`, `toa_*` (mean/std/CI),
  `eta_mean`, `reward_mean`, `energy_mean`, and embedded `min_air_time`/`max_air_time`/`beta`.
- `data/selection_<mode>_n{N}_t{T}.csv` — the learned policy's SF histogram (`sf,count,pct`).

## Reproducing the paper

The defaults follow the most up-to-date (cooperative) configuration: `lr=0.01`, `gamma=0.9`,
`eps ∈ [0.1, 1]`, `beta=0.25`, `seed=18`, `tx_prob=0.99`, `period=12000`,
`loc_range ≈ 63 m` (SF7 range ≈ the paper's 64 m). The paper's Table 2 lists `gamma=0.99`
and `eps_min=0.2`; add `--gamma 0.99 --eps-min 0.2` to match it.

Paper-scale runs, e.g.:

```bash
for N in 32 64 128; do
  python simulate.py train -n $N -t 4 --gamma 0.99 --eps-min 0.2 --episodes 4000 --max-steps 4000
  python simulate.py eval  -n $N -t 4 --seeds 20 --max-steps 4000
done
```

**Cost warning:** the tabular Q-table has shape `N × T⁶ × 6`, so memory grows fast
(e.g. `N=128, T=8` ≈ 1.6 GB), and paper-scale runs (`episodes=4000`, `max_steps=4000`)
take a long time in pure Python. Start small (`--episodes 100 --max-steps 200`) to smoke-test.

## Citation

If you use this code, please cite the paper
([doi:10.1145/3603166.3632564](https://doi.org/10.1145/3603166.3632564)):

```bibtex
@inproceedings{scarvaglieri2023ucc,
  author    = {Scarvaglieri, Antonio and Palazzo, Sergio and Busacca, Fabio},
  title     = {A Lightweight, Fully-Distributed {AI} Framework for
               Energy-Efficient Resource Allocation in {LoRa} Networks},
  booktitle = {Proc. 16th IEEE/ACM International Conference on Utility and
               Cloud Computing (UCC 2023)},
  address   = {Taormina, Italy},
  month     = dec,
  year      = {2023},
  doi       = {10.1145/3603166.3632564},
}
```
