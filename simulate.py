#!/usr/bin/env python3
"""Command-line driver for the Q-LoRa spreading-factor allocation study.

Two subcommands:

    train   Train a tabular multi-agent Q-learning policy and write the learned
            Q-table (.npy) plus per-episode learning metrics (.csv).

    eval    Evaluate one or more allocation strategies (Min/Max/Random/Smart/
            ADR) over a set of seeds and write aggregate per-strategy metrics
            (.csv) plus, for the learned policy, the SF-selection histogram
            (.csv).

The CSV files written here are exactly what ``plot.py`` consumes to reproduce
the figures of the paper.

Examples
--------
    # Train the cooperative policy for N=8 devices, T=4 threshold levels
    python simulate.py train --agents 8 --threshold 4 --episodes 4000

    # Evaluate every strategy for the same configuration over 20 seeds
    python simulate.py eval --agents 8 --threshold 4 --seeds 20

    # Reproduce the paper's Q-learning hyper-parameters (Table 2)
    python simulate.py train -n 8 -t 4 --gamma 0.99 --eps-min 0.2

Defaults follow the most up-to-date (cooperative) notebook: lr=0.01, gamma=0.9,
eps in [0.1, 1]. The paper's Table 2 lists gamma=0.99 and eps_min=0.2; pass the
flags above to match it.
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

import lora_sim
from lora_sim import LoRaNetwork, Ptx

try:
    import scipy.stats as st
except ImportError:  # confidence intervals degrade gracefully without scipy
    st = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_env(args, adr=False):
    """Build a LoRaNetwork from parsed CLI arguments."""
    return LoRaNetwork(
        max_steps=args.max_steps,
        n_agents=args.agents,
        n_threshold=args.threshold,
        period=args.period,
        loc_range=args.loc_range,
        tx_prob=args.tx_prob,
        beta=args.beta,
        reward_mode=args.mode,
        adr=adr,
        init_sf=getattr(args, "init_sf", 7),
        verbose=args.verbose,
    )


def tag(args):
    """Filename stem shared by all artefacts of one configuration."""
    return "{}_n{}_t{}".format(args.mode, args.agents, args.threshold)


def step_decay(episode, initial_lr, episodes):
    """Halve the learning rate every ``episodes/5`` episodes."""
    drop = 0.5
    episode_drop = episodes / 5
    return initial_lr * math.pow(drop, math.floor((1 + episode) / episode_drop))


def epsilon_at(episode, eps_max, eps_min, decay):
    """Exponentially decayed epsilon for the given episode."""
    return eps_min + (eps_max - eps_min) * math.exp(-decay * episode)


def mean_ci(values, confidence=0.95):
    """Return (mean, std, ci_low, ci_high) for a 1-D sample."""
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    std = float(a.std())
    if a.size < 2 or st is None:
        return mean, std, mean, mean
    low, high = st.t.interval(confidence, a.size - 1, loc=mean, scale=st.sem(a))
    return mean, std, float(low), float(high)


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def run_train(args):
    env = make_env(args)
    n_states = env._n_threshold ** 6
    print("Node placement radius (loc_range): {:.2f} m".format(env.loc_range))
    print("Thresholds SF7-SF10: {} | SF11-SF12: {}".format(
        env.arr_thresholds, getattr(env, "arr_thresholds_high", "n/a")))
    print("Q-table shape: ({}, {}, 6)  ~ {:.1f} MB"
          .format(env.n_agents, n_states, env.n_agents * n_states * 6 * 8 / 1e6))

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project="q-lora", config=vars(args))

    Q = np.ones((env.n_agents, n_states, env.N_ACTIONS))
    idx_agents = np.arange(env.n_agents)
    train_rng = np.random.default_rng(args.seed)
    visited = np.zeros(n_states, dtype=bool)

    records = []
    epsilon = args.eps_max
    for episode in range(args.episodes):
        lr = (step_decay(episode, args.lr, args.episodes)
              if args.lr_schedule == "step" else args.lr)

        state = env.reset(seed=args.seed)          # same topology every episode
        state_idx = env.state_index(state)
        visited[state_idx] = True
        action_count = np.zeros(env.N_ACTIONS)
        total_reward = 0.0
        info = {"pdr": 0.0, "avg_airtime": 0.0, "del_factor": 0.0}

        for _ in range(env._max_steps):
            explore = train_rng.random(env.n_agents) < epsilon
            greedy = Q[idx_agents, state_idx, :].argmax(-1)
            random_a = train_rng.integers(0, env.N_ACTIONS, size=env.n_agents)
            action = np.where(explore, random_a, greedy)
            action_count += np.bincount(action, minlength=env.N_ACTIONS)

            new_state, reward, done, info = env.step(action)
            new_idx = env.state_index(new_state)

            r_vec = reward if np.ndim(reward) > 0 else np.full(env.n_agents, reward)
            cur = Q[idx_agents, state_idx, action]
            best_next = Q[idx_agents, new_idx, :].max(-1)
            Q[idx_agents, state_idx, action] = cur + lr * (
                r_vec + args.gamma * best_next - cur)

            total_reward += float(np.mean(r_vec))
            state_idx = new_idx
            visited[state_idx] = True

        epsilon = epsilon_at(episode, args.eps_max, args.eps_min, args.decay)
        records.append({
            "episode": episode,
            "total_reward": total_reward,
            "avg_reward": total_reward / env._max_steps,
            "pdr": info["pdr"],
            "toa": info["avg_airtime"],
            "epsilon": epsilon,
            "lr": lr,
            "visited_pct": 100.0 * visited.mean(),
        })
        if episode % max(1, args.episodes // 20) == 0 or episode == args.episodes - 1:
            print("[ep {:>5}/{}] eps={:.3f} lr={:.4f} pdr={:.3f} toa={:.1f} "
                  "avg_r={:.4f} visited={:.1f}%".format(
                      episode, args.episodes, epsilon, lr, info["pdr"],
                      info["avg_airtime"], total_reward / env._max_steps,
                      100.0 * visited.mean()))
        if wandb_run is not None:
            wandb_run.log(records[-1])

    os.makedirs(args.out, exist_ok=True)
    q_path = os.path.join(args.out, "q_table_" + tag(args) + ".npy")
    csv_path = os.path.join(args.out, "training_" + tag(args) + ".csv")
    np.save(q_path, Q)
    pd.DataFrame(records).to_csv(csv_path, index=False)
    if wandb_run is not None:
        wandb_run.finish()
    print("\nSaved Q-table -> {}".format(q_path))
    print("Saved metrics -> {}".format(csv_path))


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def run_episode(env, seed, strategy, Q=None, selection=None):
    """Run one evaluation episode; return (pdr, toa, del_factor) means over steps."""
    state = env.reset(seed=seed)
    pdrs, toas, dels = [], [], []
    fixed_action = None
    if strategy == "min":
        fixed_action = np.zeros(env.n_agents, dtype=int)        # SF7
    elif strategy == "max":
        fixed_action = np.full(env.n_agents, 5, dtype=int)       # SF12

    for _ in range(env._max_steps):
        if strategy == "smart":
            state_idx = env.state_index(state)
            action = Q[np.arange(env.n_agents), state_idx, :].argmax(-1)
            if selection is not None:
                selection += np.bincount(action, minlength=env.N_ACTIONS)
        elif strategy == "random" or strategy == "adr":
            action = env.sample_action()          # ignored internally when ADR
        else:
            action = fixed_action

        state, reward, done, info = env.step(action)
        pdrs.append(info["pdr"])
        toas.append(info["avg_airtime"])
        dels.append(info["del_factor"])

    return float(np.mean(pdrs)), float(np.mean(toas)), float(np.mean(dels))


def run_eval(args):
    seeds = list(range(args.seeds)) if args.seeds_list is None else args.seeds_list
    strategies = args.strategies

    Q = None
    if "smart" in strategies:
        q_path = args.q_table or os.path.join(args.out, "q_table_" + tag(args) + ".npy")
        if not os.path.exists(q_path):
            sys.exit("Q-table not found: {}\nRun `simulate.py train` first or pass "
                     "--q-table.".format(q_path))
        Q = np.load(q_path)
        print("Loaded Q-table <- {}".format(q_path))

    # A reference env only for its (config-independent) airtime bounds.
    ref = make_env(args)
    min_toa, max_toa, beta = ref.min_air_time, ref.max_air_time, ref.beta

    rows = []
    selection = np.zeros(ref.N_ACTIONS) if "smart" in strategies else None
    for strat in strategies:
        env = make_env(args, adr=(strat == "adr"))
        per_seed = [run_episode(env, s, strat, Q=Q,
                                selection=selection if strat == "smart" else None)
                    for s in seeds]
        pdr_vals = [p for p, _, _ in per_seed]
        toa_vals = [t for _, t, _ in per_seed]
        eta_vals = [d for _, _, d in per_seed]

        pdr_m, pdr_s, pdr_lo, pdr_hi = mean_ci(pdr_vals)
        toa_m, toa_s, toa_lo, toa_hi = mean_ci(toa_vals)
        eta_m = float(np.mean(eta_vals))
        reward_m = pdr_m - beta * (toa_m - min_toa) / (max_toa - min_toa)
        energy_m = Ptx * toa_m / args.max_steps       # avg energy proxy [kJ]

        rows.append({
            "strategy": strat,
            "pdr_mean": pdr_m, "pdr_std": pdr_s,
            "pdr_ci_low": pdr_lo, "pdr_ci_high": pdr_hi,
            "toa_mean": toa_m, "toa_std": toa_s,
            "toa_ci_low": toa_lo, "toa_ci_high": toa_hi,
            "eta_mean": eta_m, "reward_mean": reward_m, "energy_mean": energy_m,
            "n_seeds": len(seeds),
            "min_air_time": min_toa, "max_air_time": max_toa, "beta": beta,
            "n_agents": args.agents, "n_threshold": args.threshold,
            "max_steps": args.max_steps,
        })
        print("{:>7}: PDR={:.3f}  ToA={:8.2f} ms  eta={:.2e}  r={:.4f}".format(
            strat, pdr_m, toa_m, eta_m, reward_m))

    os.makedirs(args.out, exist_ok=True)
    eval_path = os.path.join(args.out, "eval_" + tag(args) + ".csv")
    pd.DataFrame(rows).to_csv(eval_path, index=False)
    print("\nSaved eval -> {}".format(eval_path))

    if selection is not None and selection.sum() > 0:
        sel_df = pd.DataFrame({
            "sf": [7, 8, 9, 10, 11, 12],
            "count": selection,
            "pct": 100.0 * selection / selection.sum(),
        })
        sel_path = os.path.join(args.out, "selection_" + tag(args) + ".csv")
        sel_df.to_csv(sel_path, index=False)
        print("Saved selection -> {}".format(sel_path))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_common(p):
    p.add_argument("-n", "--agents", type=int, default=8, help="number of end devices N")
    p.add_argument("-t", "--threshold", type=int, default=4, help="threshold levels T")
    p.add_argument("--mode", choices=["cooperative", "competitive"],
                   default="cooperative", help="reward function")
    p.add_argument("--max-steps", type=int, default=1000,
                   help="timesteps per episode (paper: 4000)")
    p.add_argument("--period", type=int, default=12000, help="arrival jitter window [ms]")
    p.add_argument("--tx-prob", type=float, default=0.99, help="per-node tx probability")
    p.add_argument("--beta", type=float, default=0.25, help="energy/PDR trade-off weight")
    p.add_argument("--loc-range", type=float, default=None,
                   help="placement radius [m] (default: SF7 range ~63 m)")
    p.add_argument("--seed", type=int, default=18, help="base RNG seed")
    p.add_argument("--out", default="data", help="output directory for CSV/npy")
    p.add_argument("--verbose", action="store_true", help="enable diagnostic prints")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Q-LoRa SF-allocation simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="train a Q-learning policy",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common(pt)
    pt.add_argument("--episodes", type=int, default=4000, help="training episodes")
    pt.add_argument("--lr", type=float, default=0.01, help="learning rate (alpha)")
    pt.add_argument("--lr-schedule", choices=["none", "step"], default="none",
                    help="'step' halves lr every episodes/5")
    pt.add_argument("--gamma", type=float, default=0.9,
                    help="discount factor (paper Table 2: 0.99)")
    pt.add_argument("--eps-max", type=float, default=1.0)
    pt.add_argument("--eps-min", type=float, default=0.1,
                    help="minimum epsilon (paper Table 2: 0.2)")
    pt.add_argument("--decay", type=float, default=0.00042, help="epsilon decay rate")
    pt.add_argument("--init-sf", type=int, default=7, help="initial SF at reset")
    pt.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    pt.set_defaults(func=run_train)

    pe = sub.add_parser("eval", help="evaluate allocation strategies",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common(pe)
    pe.add_argument("--strategies", nargs="+",
                    default=["min", "random", "smart", "adr", "max"],
                    choices=["min", "max", "random", "smart", "adr"],
                    help="strategies to evaluate")
    pe.add_argument("--seeds", type=int, default=20,
                    help="evaluate over seeds range(0, SEEDS)")
    pe.add_argument("--seeds-list", type=int, nargs="+", default=None,
                    help="explicit list of seeds (overrides --seeds)")
    pe.add_argument("--q-table", default=None,
                    help="path to Q-table .npy (default: derived from config)")
    pe.set_defaults(func=run_eval)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
