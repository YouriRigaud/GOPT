"""
ts_test_heuristic.py

Evaluates the OnlineBPH heuristic on the same gym environment used by the RL
agent (ts_test.py), producing the exact same metrics:
  - average space utilisation (ratio)
  - average number of packed items (num)
  - standard deviation of space utilisation (ratio_std)
  - (optional) centre-of-gravity and fragility violation rate

The environment drives the episode: at each step it exposes one item via the
observation, OnlineBPH decides where to place it (or skips it if no placement
is found), and the episode ends when the env returns `terminated=True`.

Usage
-----
    python ts_test_heuristic.py --config <your_config> [--kb 1] [--ke 5] \
        [--test_episode 100] [--seed 5]

All other flags (env, constraints, render…) are forwarded to `arguments`.
"""

import os
import sys

curr_path   = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

import time
import numpy as np
import gymnasium as gym
import torch

import arguments
from tools import set_seed, registration_envs, CategoricalMasked
from envs.Packing.online_bph import OnlineBPH, _infinite_containers, _item_rotations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_mask(obs) -> np.ndarray:
    """Extract the action mask from the observation dict."""
    return np.asarray(obs["mask"]).flatten()


def _get_candidates(env) -> np.ndarray:
    """
    Return current candidate positions from env.
    Shape: (k_placement, 3) — each row is (x, y, z).
    """
    return env.candidates[:, :3].copy()


# ---------------------------------------------------------------------------
# Mapping: OnlineBPH placement → env action index
# ---------------------------------------------------------------------------

def _onlinebph_to_action(bph: OnlineBPH, env, obs) -> int:
    """
    Run OnlineBPH on the current item to get the best (position, rotation),
    then find the env candidate index that best matches that placement.

    Steps
    -----
    1. OnlineBPH computes the best (container, rotation, EMS) internally.
       We extract the position (x, y, z) and rotation dims it chose.
    2. We search among env's valid candidates for the one closest to that
       position, with matching rotation (rot=0 original, rot=1 swapped x/y).
    3. We return the corresponding action index.

    If OnlineBPH finds no valid placement, we fall back to the first valid
    candidate in the mask — this triggers a failed step in the env and ends
    the episode naturally, consistent with the env's own termination logic.

    Rotation convention (mirrors env's idx2pos):
      rot=0 : (l, w, h)  — action indices [0 .. k-1)
      rot=1 : (w, l, h)  — action index   k-1
    """
    mask       = _get_mask(obs)                          # (k_placement,)
    k          = env.k_placement
    candidates = _get_candidates(env)                    # (k, 3)
    item_dims  = tuple(float(x) for x in env.next_box[:5])  # l, w, h, weight, fragility
    l, w, h    = item_dims[:3]
    L, W, H    = (float(x) for x in env.bin_size)

    # --- Step 1: ask OnlineBPH for its best placement --------------------
    # We peek into the last open container's EMS list after a dry-run pack.
    # To avoid mutating bph's state we query its internal _find_best_placement.
    containers_iter = _infinite_containers((L, W, H))

    # Ensure at least one open container exists in bph for the query
    if not bph.open_containers:
        # Simulate opening a first container mirroring the env state
        from envs.Packing.online_bph import OnlineBPHContainer
        c = OnlineBPHContainer(0, L, W, H)
        bph.open_containers.append(c)

    best = bph._find_best_placement([item_dims], bph.open_containers)

    if best is not None:
        _, _, bph_rotation, bph_ems = best
        bph_pos = bph_ems.min_corner          # (x, y, z) chosen by OnlineBPH
        bph_rot_dims = bph_rotation           # (rl, rw, rh)

        # Determine rotation index: rot=1 if x/y are swapped vs original
        is_rot1 = (abs(bph_rot_dims[0] - w) < 1e-9 and
                   abs(bph_rot_dims[1] - l) < 1e-9)
        target_rot = 1 if is_rot1 else 0

        # --- Step 2: find closest valid env candidate with same rotation --
        if target_rot == 0:
            valid_indices = [i for i in range(k - 1) if i < len(mask) and mask[i]]
        else:
            valid_indices = [k - 1] if k - 1 < len(mask) and mask[k - 1] else []

        if valid_indices:
            bph_xy = np.array(bph_pos[:2])
            cand_xy = candidates[valid_indices, :2]           # (n, 2)
            dists   = np.linalg.norm(cand_xy - bph_xy, axis=1)
            best_local = valid_indices[int(np.argmin(dists))]

            # Now actually pack in bph so its internal state stays in sync
            bph.pack(item_dims, containers_iter)
            return best_local

    # --- Step 3: fallback — first valid candidate (episode will end) -----
    valid = np.where(mask)[0]
    return int(valid[0]) if len(valid) > 0 else 0


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

def run_episode(env, kb: int, ke: int) -> dict:
    """
    Run one full episode using OnlineBPH.

    At every step:
      1. OnlineBPH computes the best placement for the current item.
      2. We map that placement to the closest valid env candidate index.
      3. We send the action index to env.step().
      4. The env executes the actual placement and updates its state.

    OnlineBPH and the env thus stay in sync: both track the same items
    being packed in the same order.

    Returns
    -------
    dict: ratio, num, cog, fragility_violated
    """
    obs, _ = env.reset()

    L, W, H = (float(x) for x in env.bin_size)

    # Fresh OnlineBPH instance — one per episode
    bph = OnlineBPH(kb=kb, ke=ke)

    last_ratio     = 0.0
    last_cog       = None
    last_fragility = None
    terminated     = False
    truncated      = False

    while not (terminated or truncated):
        action = _onlinebph_to_action(bph, env, obs)
        obs, _, terminated, truncated, info = env.step(action)

        last_ratio     = float(info.get("ratio",              last_ratio))
        last_cog       = info.get("cog",                      last_cog)
        last_fragility = info.get("fragility_violated",       last_fragility)

    num_packed = int(info.get("counter", 0))

    return {
        "ratio":              last_ratio,
        "num":                num_packed,
        "cog":                np.asarray(last_cog, dtype=float) if last_cog is not None else None,
        "fragility_violated": bool(last_fragility) if last_fragility is not None else None,
    }


# ---------------------------------------------------------------------------
# Main test function (mirrors ts_test.py)
# ---------------------------------------------------------------------------

def test(args):
    set_seed(args.seed, False, False)

    constraints = args.get("constraints", None)

    env = gym.make(
        args.env.id,
        container_size=args.env.container_size,
        enable_rotation=args.env.rot,
        data_type=args.env.box_type,
        item_set=args.env.box_size_set,
        reward_type=args.train.reward_type,
        action_scheme=args.env.scheme,
        k_placement=args.env.k_placement,
        is_render=args.render,
        use_weight=constraints.weight               if constraints else False,
        weight_range=list(constraints.weight_range) if constraints else [0.5, 5.0],
        use_fragility=constraints.fragility         if constraints else False,
        fragility_probability=constraints.fragility_probability if constraints else 0.3,
    )

    kb = getattr(args, "kb", 1)
    ke = getattr(args, "ke", 5)

    ratios      = []
    nums        = []
    cogs        = []
    fragilites  = []

    t0 = time.time()

    for ep in range(args.test_episode):
        result = run_episode(env, kb=kb, ke=ke)

        ratios.append(result["ratio"])
        nums.append(result["num"])

        cog  = result["cog"]
        frag = result["fragility_violated"]

        has_cog = cog is not None
        cog_str = f"cog=({cog[0]:.2f},{cog[1]:.2f},{cog[2]:.2f})" if has_cog else ""
        frag_str = str(int(frag)) if frag is not None else "-"

        print(
            f"episode {ep+1}\t => \t"
            f"ratio: {result['ratio']:.4f} \t| "
            f"total: {result['num']} \t| "
            f"{cog_str} \t| "
            f"fragility_violated: {frag_str}"
        )

        if has_cog:
            cogs.append(cog)
        if frag is not None:
            fragilites.append(int(frag))

    elapsed = time.time() - t0

    ratios_arr = np.array(ratios)
    nums_arr   = np.array(nums)

    print("All cases have been done!")
    print("----------------------------------------------")
    print(f"heuristic          : OnlineBPH  (kb={kb}, ke={ke})")
    print(f"average space utilization: {ratios_arr.mean():.4f}")
    print(f"average put item number  : {nums_arr.mean():.4f}")
    print(f"standard variance        : {ratios_arr.std():.4f}")
    print(f"total elapsed time       : {elapsed:.1f}s")

    if cogs:
        cog_mean = np.mean(cogs, axis=0)
        print(f"average cog: ({cog_mean[0]:.4f}, {cog_mean[1]:.4f}, {cog_mean[2]:.4f})")

    if fragilites:
        viol_rate = np.mean(fragilites)
        print(f"fragility violation rate : {viol_rate:.4f}")

    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    # --- Pre-parse heuristic-specific arguments BEFORE passing argv to
    #     arguments.get_args(), which would reject unknown flags.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--kb",           type=int, default=1,   help="OnlineBPH look-ahead (default 1)")
    pre_parser.add_argument("--ke",           type=int, default=5,   help="OnlineBPH top-EMSs per container (default 5)")
    pre_parser.add_argument("--test_episode", type=int, default=None, help="Override number of test episodes")
    pre_parser.add_argument("--test-episode", type=int, default=None, dest="test_episode_dash")

    heuristic_args, remaining_argv = pre_parser.parse_known_args()

    # Temporarily replace sys.argv so arguments.get_args() only sees its
    # own flags (config, ckp, render, …)
    import sys as _sys
    _sys.argv = [_sys.argv[0]] + remaining_argv

    registration_envs()
    args = arguments.get_args()
    args.train.algo = args.train.algo.upper()
    args.train.step_per_collect = args.train.num_processes * args.train.num_steps

    # Inject heuristic params into args
    args.kb = heuristic_args.kb
    args.ke = heuristic_args.ke

    # --test_episode / --test-episode override (either form accepted)
    episode_override = heuristic_args.test_episode or heuristic_args.test_episode_dash
    if episode_override is not None:
        args.test_episode = episode_override

    if args.render:
        args.test_episode = 1

    args.seed = 5
    print(f"dimension: {args.env.container_size}")
    print(f"OnlineBPH heuristic test  (kb={args.kb}, ke={args.ke})")
    test(args)