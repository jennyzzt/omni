import numpy as np

from envs import env_trc_uni
from envs.env_utils import sigmoid
from envs.moi_utils import get_task_predictions


DUMMY_BITS = env_trc_uni.DUMMY_BITS
DUMMY_TASKS = env_trc_uni.DUMMY_TASKS


class Env(env_trc_uni.Env):

  def __init__(
      self, area=(64, 64), view=(9, 9), size=(64, 64),
      reward=True, length=1500, seed=None, eager_task=False, **kwargs):
    super().__init__(area, view, size, reward, length, seed, **kwargs)
    # LP attributes
    self.learning_progress = np.zeros(len(self.target_achievements) + DUMMY_TASKS)
    self.task_probs = np.ones(len(self.target_achievements) + DUMMY_TASKS) / (len(self.target_achievements) + DUMMY_TASKS)
    # MoI attributes
    self.moi_int_tasks = set()
    self.moi_bor_tasks = set()
    self.moi_predicts = get_task_predictions(setting='rc')

  def _specify_curri_task(self):
    self.task_idx = np.random.choice(np.arange(len(self.target_achievements) + DUMMY_TASKS), size=1, p=self.task_probs)[0]
    self.task_steps = 0
    self.task_enc = self._encode_task(self.task_idx)

  def is_boring_task(self, inttaskidx, taskidx):
    int_task = self.target_achievements[inttaskidx]
    det_task = self.target_achievements[taskidx]
    if len(det_task) > 1:
        # need to look at the entire set of int tasks
        isbor = np.zeros(len(det_task))
        for intx in self.moi_int_tasks:
            int_tasks = self.target_achievements[intx]
            for i, t in enumerate(det_task):
                if not isbor[i] and not self.moi_predicts['__and__'.join(int_tasks)][t]:
                    isbor[i] = 1
        return all(isbor)
    else:
        return not self.moi_predicts['__and__'.join(int_task)]['__and__'.join(det_task)]

  def push_info(self, info):
    super().push_info(info)
    self.learning_progress = info['learning_progress']
    posidxs = [i for i, lp in enumerate(self.learning_progress) if lp > 0 or self.eval_tsr[i] > 0]
    zeroout = len(posidxs) > 0
    subprobs = self.learning_progress[posidxs] if zeroout else self.learning_progress
    std = np.std(subprobs)
    subprobs = (subprobs - np.mean(subprobs)) / (std if std else 1)  # z-score
    subprobs = sigmoid(subprobs)  # sigmoid
    subprobs = subprobs / np.sum(subprobs)  # normalize
    if zeroout:
      probs = np.zeros(len(self.learning_progress))
      probs[posidxs] = subprobs
    else:
      probs = subprobs

    # MoI: LM identifies the boring tasks with respect to tasks done sufficiently well
    # boring tasks get their probabilities decreased
    self.moi_int_tasks = set()
    self.moi_bor_tasks = set()
    sorted_task_idxs = np.argsort(self.eval_tsr[:len(self.target_achievements)])
    for idx in sorted_task_idxs[::-1]:
      if idx not in self.moi_int_tasks and idx not in self.moi_bor_tasks:
        self.moi_int_tasks.add(idx)
        # update boring set immediately
        for i in range(len(self.target_achievements)):
          if i not in self.moi_int_tasks and i not in self.moi_bor_tasks and \
             self.is_boring_task(idx, i):
            self.moi_bor_tasks.add(i)

    moi_weight = np.ones(len(probs))
    for i in self.moi_bor_tasks:
      moi_weight[i] = 0.001
    probs = probs * moi_weight

    probs = probs / np.sum(probs)  # normalize
    self.task_probs = probs
