""" 
refer:
- https://github.com/albertcity/OCARL
- https://github.com/pioneer-innovation/Real-3D-Embodied-Dataset

"""
import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)  
sys.path.append(parent_path) 
import warnings
warnings.filterwarnings("ignore")

import time
import pprint
import shutil
import random

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
import tianshou as ts
from tianshou.utils import TensorboardLogger, LazyLogger
from tianshou.data import VectorReplayBuffer
from tianshou.utils.net.common import ActorCritic, DataParallelNet
from tianshou.trainer import onpolicy_trainer_iter

import model
import arguments
from tools import *
from masked_ppo import MaskedPPOPolicy
from masked_a2c import MaskedA2CPolicy
from mycollector import PackCollector
 

def safe_copy(src, dst_dir):
    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    shutil.copy(src, dst)


def format_epoch_log(epoch, epoch_stat):
    return (
        f"epoch={epoch} "
        f"env_step={epoch_stat['env_step']} "
        f"gradient_step={epoch_stat['gradient_step']} "
        f"train_rew={epoch_stat['rew']:.6f} "
        f"train_len={epoch_stat['len']} "
        f"n_ep={epoch_stat['n/ep']} "
        f"n_st={epoch_stat['n/st']} "
        f"loss={epoch_stat['loss']:.6f} "
        f"loss_clip={epoch_stat['loss/clip']:.6f} "
        f"loss_ent={epoch_stat['loss/ent']:.6f} "
        f"loss_vf={epoch_stat['loss/vf']:.6f} "
        f"test_reward={epoch_stat['test_reward']:.6f} "
        f"test_reward_std={epoch_stat['test_reward_std']:.6f} "
        f"best_reward={epoch_stat['best_reward']:.6f} "
        f"best_reward_std={epoch_stat['best_reward_std']:.6f} "
        f"best_epoch={epoch_stat['best_epoch']}"
    )


def append_train_log(train_log_path, line):
    with open(train_log_path, "a") as file:
        file.write(line + "\n")


def make_pack_env(args):
    registration_envs()
    constraints = args.get("constraints", None)
    return gym.make(
        args.env.id,
        container_size=args.env.container_size,
        enable_rotation=args.env.rot,
        data_type=args.env.box_type,
        item_set=args.env.box_size_set,
        reward_type=args.train.reward_type,
        action_scheme=args.env.scheme,
        k_placement=args.env.k_placement,
        use_weight=constraints.weight if constraints else False,
        weight_range=list(constraints.weight_range) if constraints else [0.5, 5.0],
        use_fragility=constraints.fragility if constraints else False,
        fragility_probability=constraints.fragility_probability if constraints else 0.3,
        lambda_cog=constraints.get("lambda_cog", 0.0) if constraints else 0.0,
        observe_cog=constraints.get("observe_cog", False) if constraints else False,
        observe_fragility=constraints.get("observe_fragility", False) if constraints else False,
    )


def make_envs(args):

    train_envs = ts.env.SubprocVectorEnv(
        [lambda: make_pack_env(args)
                          for _ in range(args.train.num_processes)]
    )
    test_envs = ts.env.SubprocVectorEnv(
        [lambda: make_pack_env(args)
                          for _ in range(1)]
    )
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    return train_envs, test_envs


def build_net(args, device):
    constraints = args.get("constraints", None)
    observe_cog = constraints.get("observe_cog", False) if constraints else False
    feature_net = model.ShareNet(
        k_placement=args.env.k_placement,
        box_max_size=args.env.box_big,
        container_size=args.env.container_size,
        embed_size=args.model.embed_dim,
        num_layers=args.model.num_layers,
        forward_expansion=args.model.forward_expansion,
        heads=args.model.heads,
        dropout=args.model.dropout,
        device=device,
        place_gen=args.env.scheme,
        observe_cog=observe_cog,
        observe_fragility=constraints.get("observe_fragility", False) if constraints else False,
    )

    actor = model.ActorHead(
        preprocess_net=feature_net, 
        embed_size=args.model.embed_dim, 
        padding_mask=args.model.padding_mask,
        device=device, 
    ).to(device)

    critic = model.CriticHead(
        preprocess_net=feature_net, 
        k_placement=args.env.k_placement,
        embed_size=args.model.embed_dim,
        padding_mask=args.model.padding_mask,
        device=device, 
    ).to(device)

    return actor, critic


def train(args):

    date = time.strftime(r'%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
    time_str = args.env.id + "_" + \
        str(args.env.container_size[0]) + "-" + str(args.env.container_size[1]) + "-" + str(args.env.container_size[2]) + "_" + \
        args.env.scheme + "_" + str(args.env.k_placement) + "_" +\
        args.env.box_type + "_" + \
        args.train.algo  + '_' \
        'seed' + str(args.seed) + "_" + \
        args.opt.optimizer + "_" \
        + date

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda", args.device)
    else:
        device = torch.device("cpu")

    set_seed(args.seed, args.cuda, args.cuda_deterministic)

    # environments 
    train_envs, test_envs = make_envs(args)  # make envs and set random seed

    # network
    actor, critic = build_net(args, device)
    actor_critic = ActorCritic(actor, critic)

    if args.opt.optimizer == 'Adam':
        optim = torch.optim.Adam(actor_critic.parameters(), lr=args.opt.lr, eps=args.opt.eps)
    elif args.opt.optimizer == 'RMSprop':
        optim = torch.optim.RMSprop(
            actor_critic.parameters(),
            lr=args.opt.lr,
            eps=args.opt.eps,
            alpha=args.opt.alpha,
        )
    else:
        raise NotImplementedError

    lr_scheduler = None
    if args.opt.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(args.train.step_per_epoch / args.train.step_per_collect) * args.train.epoch
        lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)


    # RL agent 
    dist = CategoricalMasked
    if args.train.algo == 'PPO':
        policy = MaskedPPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist,
            discount_factor=args.train.gamma,
            eps_clip=args.train.clip_param,
            advantage_normalization=False,
            vf_coef=args.loss.value,
            ent_coef=args.loss.entropy,
            gae_lambda=args.train.gae_lambda,
            lr_scheduler=lr_scheduler
        )
    elif args.algo == 'A2C':    
        policy = MaskedA2CPolicy(
            actor,
            critic,
            optim,
            dist,
            discount_factor=args.train.gamma,
            vf_coef=args.loss.value,
            ent_coef=args.loss.entropy,
            gae_lambda=args.train.gae_lambda,
            lr_scheduler=lr_scheduler
        )
    else:
        raise NotImplementedError

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"No checkpoint found at: {args.resume}")
        log_path = os.path.dirname(os.path.abspath(args.resume))
    else:
        log_path = './logs/' + time_str
    train_log_path = os.path.join(log_path, "train.log")
    
    is_debug = True if sys.gettrace() else False
    if not is_debug:
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(
            writer=writer,
            train_interval=args.log_interval,
            update_interval=args.log_interval
        )
        os.makedirs(log_path, exist_ok=True)
        # backup the config file, os.path.join(,)
        safe_copy(args.config, log_path)
        safe_copy("model.py", log_path)
        safe_copy("arguments.py", log_path)
    else:
        logger = LazyLogger()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        policy.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])
        if lr_scheduler is not None and "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        elif lr_scheduler is not None:
            warnings.warn(
                "Checkpoint has no lr_scheduler state; resuming without restoring "
                "the scheduler state."
            )

    # ======== callback functions used during training =========
    def train_fn(epoch, env_step):
        # monitor leraning rate in tensorboard
        # writer.add_scalar('train/lr', optim.param_groups[0]["lr"], env_step)
        pass

    def save_best_fn(policy):
        if not is_debug:
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy_step_best.pth'))
        else:
            pass

    def final_save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy_step_final.pth'))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        if not is_debug:
            # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
            ckpt_path = os.path.join(log_path, "checkpoint.pth")
            # Example: saving by epoch num
            # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
            state = {
                "model": policy.state_dict(),
                "optim": optim.state_dict(),
            }
            if lr_scheduler is not None:
                state["lr_scheduler"] = lr_scheduler.state_dict()
            torch.save(state, ckpt_path)
            return ckpt_path
        else:
            return None
    
    def watch(train_info):
        print("Setup test envs ...")
        policy.eval()
        test_envs.seed(args.seed)
        print("Testing agent ...")
        test_collector.reset()
        result = test_collector.collect(n_episode=1000)
        ratio = result["ratio"]
        ratio_std = result["ratio_std"]
        total = result["num"]
        cog = result['cog']
        imbalance = result['imbalance']
        frag_rate = result['fragility_violated_rate']
        print(f"The result (over {result['n/ep']} episodes): ratio={ratio}, ratio_std={ratio_std}, total={total}")
        print(f"  avg cog: ({cog[0]:.4f}, {cog[1]:.4f}, {cog[2]:.4f}) | imbalance: {imbalance:.4f} | fragility_violation_rate: {frag_rate:.4f}")
        append_train_log(
            train_log_path,
            f"final_test episodes={result['n/ep']} ratio={ratio:.6f} "
            f"ratio_std={ratio_std:.6f} total={total:.6f} "
            f"cog=({cog[0]:.4f},{cog[1]:.4f},{cog[2]:.4f}) imbalance={imbalance:.4f} fragility_violated_rate={frag_rate:.4f}"
        )
        with open(os.path.join(log_path, f"{ratio:.4f}_{ratio_std:.4f}_{total}.txt"), "w") as file:
            file.write(str(train_info).replace("{", "").replace("}", "").replace(", ", "\n"))

    buffer = VectorReplayBuffer(total_size=10000, buffer_num=len(train_envs))
    train_collector = PackCollector(policy, train_envs, buffer)
    test_collector = PackCollector(policy, test_envs)
    
    # trainer
    trainer = onpolicy_trainer_iter(
        policy,
        train_collector,
        test_collector,
        max_epoch=args.train.epoch,
        step_per_epoch=args.train.step_per_epoch,
        repeat_per_collect=args.train.repeat_per_collect,
        episode_per_test=10, # args.test_num,
        batch_size=args.train.batch_size,
        step_per_collect=args.train.step_per_collect,
        # episode_per_collect=args.episode_per_collect,
        train_fn=train_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        resume_from_log=bool(args.resume),
        logger=logger,
        test_in_train=False
    )
    result = {}
    for epoch, epoch_stat, info in trainer:
        append_train_log(train_log_path, format_epoch_log(epoch, epoch_stat))
        result = info

    final_save_fn(policy)
    pprint.pprint(f'Finished training! \n{result}')
    watch(result)


if __name__ == '__main__':
    registration_envs()
    args = arguments.get_args()
    args.train.algo = args.train.algo.upper()
    args.train.step_per_collect = args.train.num_processes * args.train.num_steps  

    train(args)
