import argparse
import os
import glob
import re
import time
import numpy as np
import importlib
import torch
from torch_ac.utils.penv import ParallelEnv
import tensorboardX

import utils
from utils import device
import crafter


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True,
                        help="name of the environment (REQUIRED)")
    parser.add_argument("--model", required=True,
                        help="name of the trained model (REQUIRED)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="number of episodes of evaluation (default: 100)")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed (default: 0)")
    parser.add_argument("--procs", type=int, default=16,
                        help="number of processes (default: 16)")
    parser.add_argument("--argmax", action="store_true", default=False,
                        help="action with highest probability is selected")
    parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                        help="how many worst episodes to show")
    parser.add_argument("--record-video", action="store_true", default=False,
                        help="record evaluation videos")
    parser.add_argument("--all-checkpoints", action="store_true", default=False,
                        help="evaluate all checkpoints saved in model dir")
    args = parser.parse_args()
    # assertions to ensure that metrics are logged properly
    assert(args.episodes % args.procs == 0)

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device
    print(f"Device: {device}\n")

    # Load environments
    env_module = importlib.import_module(f'envs.env_{args.env}')
    envs = []
    model_dir = utils.get_model_dir(args.model)
    for i in range(args.procs):
        env = env_module.Env(seed=args.seed + 100 * i)
        if args.record_video:
            env = crafter.Recorder(
                env, f"{model_dir}",
                save_stats=False,
                save_video=True,
                save_episode=False,
            )
        envs.append(env)
    env = ParallelEnv(envs)
    print("Environments loaded\n")

    if args.all_checkpoints:
        # Load Tensorboard writer
        tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Find checkpoints
    checkpoints = []
    if args.all_checkpoints:
        filepath = os.path.join(os.getcwd(), model_dir)
        files = glob.glob(filepath + "/status_*.pt")
        checkpoints = [re.search(f"{filepath}/status_(.*).pt", f).group(1) for f in files]
    checkpoints = sorted(checkpoints, key=lambda x: (len(x), x))
    checkpoints += [""]

    for checkpoint in checkpoints:
        # Load agent
        agent = utils.Agent.dir_init(env.observation_space, env.action_space, model_dir,
                                     argmax=args.argmax, num_envs=args.procs, model_suffix=checkpoint)
        print("Agent loaded\n")

        # Initialize logs
        logs = {"num_frames_per_episode": [], "return_per_episode": []}
        logs_info_startkeys = [
            # "craftscore", "craftscore_followed",
            # "craftscore_int",
            # "craftscore_bor_followed", "craftscore_int_followed"
        ]
        logs_info = {k: [] for k in logs_info_startkeys}

        # Run agent
        start_time = time.time()
        obss = env.reset()
        log_done_counter = 0
        log_episode_return = torch.zeros(args.procs, device=device)
        log_episode_num_frames = torch.zeros(args.procs, device=device)
        while log_done_counter < args.episodes:
            actions = agent.get_actions(obss)
            obss, rewards, terminateds, truncateds, infos = env.step(actions)
            dones = tuple(a | b for a, b in zip(terminateds, truncateds))
            agent.analyze_feedbacks(rewards, dones)

            log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
            log_episode_num_frames += torch.ones(args.procs, device=device)

            for i, done in enumerate(dones):
                if done:
                    log_done_counter += 1
                    logs["return_per_episode"].append(log_episode_return[i].item())
                    logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())
                    # get metrics in logs_info
                    for key in logs_info_startkeys:
                        logs_info[key].append(infos[i][key])
                    # get follow, given achievemenets metrics
                    if "taskcond" in args.env:
                        all_given_counts = 0
                        all_follow_counts = 0
                        for key, val in infos[i]["given_achs"].items():
                            logs_key = f'given_{key}'
                            logs_info[logs_key] = logs_info.get(logs_key, [])
                            logs_info[logs_key].append(val)
                            all_given_counts += val
                            logs_key = f'fpercent_{key}'
                            fpercent = val and infos[i]["follow_achs"][key] / val or 0
                            logs_info[logs_key] = logs_info.get(logs_key, [])
                            logs_info[logs_key].append(fpercent)
                            all_follow_counts += infos[i]["follow_achs"][key]
                        logs_info["follow_percent"] = logs_info.get("follow_percent", [])
                        logs_info["follow_percent"].append(all_follow_counts / all_given_counts)

            mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
            log_episode_return *= mask
            log_episode_num_frames *= mask
        end_time = time.time()

        # Print logs
        num_frames = sum(logs["num_frames_per_episode"])
        fps = num_frames / (end_time - start_time)
        duration = int(end_time - start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
              .format(num_frames, fps, duration,
                      *return_per_episode.values(),
                      *num_frames_per_episode.values()))

        # Write to Tensorboard
        header = [f"eval/{h}" for h in logs_info.keys()]
        data = [np.median(v) if type(v)==list else v/args.episodes for k, v in logs_info.items()]
        if checkpoint:
            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, int(checkpoint))

        # Print worst episodes
        n = args.worst_episodes_to_show
        if n > 0:
            print("\n{} worst episodes:".format(n))

            indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
            for i in indexes[:n]:
                print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
