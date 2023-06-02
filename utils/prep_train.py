import os
import argparse
import json
import numpy as np

from envs import env_alltasks
from envs.env_utils import ach_to_string
from utils.other import root_path
from torch_ac.utils import ParallelEnv


def get_rdn_tsr(env):
    with open(os.path.join(root_path, './utils/prep_train.json'), 'r') as f:
        data = json.load(f)
    all_rdntsr = data['rdn_tsr']
    try:
        # synonym tasks
        envtasks = env.target_achievements_base
    except:
        envtasks = env.target_achievements
    rdntsr = []
    for t in envtasks:
        rdntsr.append(all_rdntsr[ach_to_string(t)])
    rdntsr = np.concatenate((rdntsr, np.zeros(len(env.given_achievements) - len(rdntsr))))
    return rdntsr

def eval_all_tasks_random(penv, num_eps=1):
    penv.reset()
    given_counts = np.zeros(len(penv.envs[0].given_achievements))
    follow_counts = np.zeros(len(penv.envs[0].follow_achievements))
    ep_counter = 0
    while ep_counter < num_eps:
        actions = [penv.envs[0].action_space.sample() for _ in range(len(penv.envs))]
        _, _, terminateds, truncateds, infos = penv.step(actions)
        dones = tuple(a | b for a, b in zip(terminateds, truncateds))
        for i, done in enumerate(dones):
            if done:
                given_counts += list(infos[i]['given_achs'].values())
                follow_counts += list(infos[i]['follow_achs'].values())
                ep_counter += 1
                if ep_counter % 25 == 0:
                    print('eps done:', ep_counter)
    follow_counts = np.concatenate((follow_counts, np.zeros(len(given_counts) - len(follow_counts))))
    task_success_rates = np.divide(follow_counts, given_counts, out=np.zeros_like(follow_counts), where=given_counts!=0)
    return task_success_rates


if __name__ == "__main__":
    ''' 
    prep materials used in train.py 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-eps", type=int, default=200,
                        help="number of episodes to evaluate for learning progress (default: 200)")
    parser.add_argument("--use-prev", action="store_true", default=False,
                        help="use previously saved file and continue from there")
    parser.add_argument("--eval-procs", type=int, default=10,
                        help="number of processes (default: 10)")
    args = parser.parse_args()

    if args.use_prev:
        with open('./utils/prep_train.json', 'r') as f:
            data = json.load(f)
    else:
        data = {}

    # save random policy task success rates
    eval_envs = [env_alltasks.Env() for _ in range(args.eval_procs)]
    eval_envs = ParallelEnv(eval_envs)
    eval_envs.reset()
    eval_envs.set_curriculum(train=False)
    task_success_rates = eval_all_tasks_random(eval_envs, num_eps=args.eval_eps)
    tsr_data = {}
    for i, task in enumerate(eval_envs.envs[0].target_achievements):
        tsr_data[ach_to_string(task)] = task_success_rates[i]
    if args.use_prev:
        data['rdn_tsr'] = {
            k: (tsr_data[k]*args.eval_eps + v*data['eval_eps']) / (args.eval_eps + data['eval_eps'])
            for k, v in data['rdn_tsr'].items()
        }
        data['eval_eps'] += args.eval_eps
    else:
        data['rdn_tsr'] = tsr_data
        data['eval_eps'] = args.eval_eps

    # save data to json
    with open('./utils/prep_train.json', 'w') as f:
        json.dump(data, f, indent=4)

    # check data saved
    eval_env = env_alltasks.Env()
    eval_env.reset()
    tsr = get_rdn_tsr(eval_env)
    print(len(tsr))
