import argparse
import time
import datetime
import torch_ac
import tensorboardX
import sys
import numpy as np
import importlib
import torch

from torch_ac.utils import ParallelEnv
import utils
from utils import device
from utils.prep_train import get_rdn_tsr
from model import ACModel
from envs.env_utils import ach_to_string


def eval_all_tasks(acmodel, penv, num_eps=1):
    given_counts = np.zeros(len(penv.envs[0].given_achievements))
    follow_counts = np.zeros(len(penv.envs[0].follow_achievements))
    agent = utils.Agent.model_init(penv.observation_space, acmodel, num_envs=len(penv.envs))
    with torch.no_grad():
        obss = penv.reset()
        ep_counter = 0
        while ep_counter < num_eps:
            actions = agent.get_actions(obss)
            obss, rewards, terminateds, truncateds, infos = penv.step(actions)
            dones = tuple(a | b for a, b in zip(terminateds, truncateds))
            for i, done in enumerate(dones):
                if done:
                    given_counts += list(infos[i]['given_achs'].values())
                    follow_counts += list(infos[i]['follow_achs'].values())
                    ep_counter += 1
    acmodel.train()
    follow_counts = np.concatenate((follow_counts, np.zeros(len(given_counts) - len(follow_counts))))
    task_success_rates = np.divide(follow_counts, given_counts, out=np.zeros_like(follow_counts), where=given_counts!=0)
    return task_success_rates


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument("--algo", default='ppo',
                        help="algorithm to use: a2c | ppo")
    parser.add_argument("--env", default='custom',
                        help="name of the environment to train on")
    parser.add_argument("--model", default=None,
                        help="name of the model (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="number of updates between two logs (default: 50)")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="number of updates between two saves (default: 50, 0 means no saving)")
    parser.add_argument("--eval-interval", type=int, default=25,
                        help="number of updates between two evals (default: 25, 0 means no evaluating)")
    parser.add_argument("--procs", type=int, default=32,
                        help="number of processes (default: 32)")
    parser.add_argument("--frames", type=int, default=10**7,
                        help="number of frames of training (default: 1e7)")
    # Parameters for main algorithm
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="batch size for PPO (default: 2048)")
    parser.add_argument("--frames-per-proc", type=int, default=1024,
                        help="number of frames per process before update (default: 5 for A2C and 1024 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate (default: 0.0001)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--ac-size", type=int, default=128,
                        help="actor-critic layer size (default: 128)")
    parser.add_argument("--activation", default='tanh',
                        help="activation to use: tanh | relu")
    # Parameters for learning progress
    parser.add_argument("--eval-procs", type=int, default=20,
                        help="number of processes (default: 20)")
    parser.add_argument("--ema-alpha", type=float, default=0.1,
                        help="smoothing value for ema in claculating learning progress (default: 0.1)")
    parser.add_argument("--p-theta", type=float, default=0.1,
                        help="parameter for reweighing learning progress (default: 0.1)")
    parser.add_argument("--eval-num", type=int, default=20,
                        help="number of times to evaluate each task for learning progress (default: 20)")
    args = parser.parse_args()
    args.recurrence = 2

    # Set run dir
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir)
    # csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device
    txt_logger.info(f"Device: {device}\n")

    # Load environments
    env_module = importlib.import_module(f'envs.env_{args.env}')
    eval_envs = [env_module.Env() for _ in range(args.eval_procs)]
    eval_envs = ParallelEnv(eval_envs)
    eval_envs.reset()
    eval_envs.set_curriculum(train=False)
    eval_eps = args.eval_num * len(eval_envs.envs[0].target_achievements) * 300 / eval_envs.envs[0]._length
    # eval_eps = 1  # for testing
    envs = []
    for i in range(args.procs):
        envs.append(env_module.Env())
    txt_logger.info("Environments loaded\n")

    # Load training status
    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {
            "num_frames": 0, "update": 0,
            "p_fast": None, "p_slow": None, "raw_tsr": None, "ema_tsr": None,
        }
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    txt_logger.info("Observations preprocessor loaded")

    # Load model
    acmodel = ACModel(obs_space, envs[0].action_space, acsize=args.ac_size, activation=args.activation)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo
    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model
    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()
    if args.eval_interval > 0:
        rdn_tsr = get_rdn_tsr(eval_envs.envs[0])
        # rdn_tsr = np.zeros(len(eval_envs.given_achievements))
        raw_tsr = status["raw_tsr"]
        ema_tsr = status["ema_tsr"]
        p_fast = status["p_fast"]
        p_slow = status["p_slow"]
        # push saved info to envs
        if p_fast is not None:  # checking one is enough
            p_theta = args.p_theta
            p_fast_reweigh = ((1 - p_theta) * p_fast) / (p_fast + p_theta * (1 - 2 * p_fast))
            p_slow_reweigh = ((1 - p_theta) * p_slow) / (p_slow + p_theta * (1 - 2 * p_slow))
            learning_progress = np.abs(p_fast_reweigh - p_slow_reweigh)
            info = {
                'learning_progress': learning_progress,
                'raw_tsr': raw_tsr,
                'ema_tsr': ema_tsr,
            }
            algo.env.push_info(info)

        task_sampled_rates = np.zeros(len(eval_envs.envs[0].given_achievements))
        # evaluate once before any training
        raw_tsr = eval_all_tasks(acmodel, eval_envs, num_eps=eval_eps)
        # Write task success rates to tensorboard
        header = [f'train_eval/{ach_to_string(ach)}-sr' for ach in envs[0].target_achievements]
        for field, value in zip(header, raw_tsr):
            tb_writer.add_scalar(field, value, num_frames)

    while num_frames < args.frames:
        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Save relevant env info
        if args.eval_interval > 0:
            done_envinfo = logs["done_envinfo"]
            for einfo in done_envinfo:
                task_sampled_rates += list(einfo['given_achs'].values())

        # Evaluate for learning progress
        if args.eval_interval > 0 and update % args.eval_interval == 0:
            raw_tsr = eval_all_tasks(acmodel, eval_envs, num_eps=eval_eps)
            # normalize task success rates with random baseline rates
            norm_tsr = np.maximum(raw_tsr - rdn_tsr, np.zeros(raw_tsr.shape)) / (1.0 - rdn_tsr)
            # exponential mean average learning progress
            ema_tsr = raw_tsr * args.ema_alpha + ema_tsr * (1 - args.ema_alpha) if ema_tsr is not None else raw_tsr
            p_fast = norm_tsr * args.ema_alpha + p_fast * (1 - args.ema_alpha) if p_fast is not None else norm_tsr
            p_slow = p_fast * args.ema_alpha + p_slow * (1 - args.ema_alpha) if p_slow is not None else p_fast
            # NOTE: weighting to give more focus to tasks with lower success probabilities
            p_theta = args.p_theta
            p_fast_reweigh = ((1 - p_theta) * p_fast) / (p_fast + p_theta * (1 - 2 * p_fast))
            p_slow_reweigh = ((1 - p_theta) * p_slow) / (p_slow + p_theta * (1 - 2 * p_slow))
            # learning progress is the change in probability to task success rate
            # NOTE: using bidirectional LP
            learning_progress = np.abs(p_fast_reweigh - p_slow_reweigh)

            # Push information to each environment
            info = {
                'learning_progress': learning_progress,
                'raw_tsr': raw_tsr,
                'ema_tsr': ema_tsr,
            }
            algo.env.push_info(info)
            # Write task success rates to tensorboard
            header = [f'train_eval/{ach_to_string(ach)}-sr' for ach in envs[0].target_achievements]
            for field, value in zip(header, raw_tsr):
                tb_writer.add_scalar(field, value, num_frames)
            # Write saved env info to tensorboard
            sum_tsar = np.sum(task_sampled_rates)
            task_sampled_rates /= sum_tsar if sum_tsar > 0 else 1
            header = [f'train_sampled/{ach_to_string(ach)}' for ach in envs[0].target_achievements]
            header.append('train_sampled/dummy')  # all dummy tasks are the same, so keep track of one
            for field, value in zip(header, task_sampled_rates):
                tb_writer.add_scalar(field, value, num_frames)
            task_sampled_rates = np.zeros(len(eval_envs.envs[0].given_achievements))

        # Print logs
        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            # if status["num_frames"] == 0:
            #     csv_logger.writerow(header)
            # csv_logger.writerow(data)
            # csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status
        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "p_fast": p_fast, "p_slow": p_slow, "raw_tsr": raw_tsr, "ema_tsr": ema_tsr,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            utils.save_status(status, model_dir, suffix=str(update))
            txt_logger.info("Status saved")
            # Update final checkpoint status
            if update % (args.save_interval * 1) == 0:
                utils.save_status(status, model_dir)
                txt_logger.info("Final status updated")

    # Save final checkpoint status
    status = {"num_frames": num_frames, "update": update,
              "p_fast": p_fast, "p_slow": p_slow, "raw_tsr": raw_tsr, "ema_tsr": ema_tsr,
              "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
    utils.save_status(status, model_dir)
    txt_logger.info("Final status saved")
