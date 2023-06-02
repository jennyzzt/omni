import numpy as np

from envs import env_trc_uni
from envs.env_utils import sigmoid


DUMMY_BITS = env_trc_uni.DUMMY_BITS
DUMMY_TASKS = env_trc_uni.DUMMY_TASKS


class Env(env_trc_uni.Env):

  def __init__(
      self, area=(64, 64), view=(9, 9), size=(64, 64),
      reward=True, length=1500, seed=None, eager_task=False, **kwargs):
    super().__init__(area, view, size, reward, length, seed, **kwargs)
    # LP attributes
    self.learning_progress = np.zeros(len(self.target_achievements) + DUMMY_TASKS)
    self.task_probs = np.ones(len(self.target_achievements)  + DUMMY_TASKS) / (len(self.target_achievements) + DUMMY_TASKS)
    # PMI attributes
    self.pmi_reweigh = np.ones(len(self.target_achievements) + DUMMY_TASKS)
    for i, task in enumerate(self.target_achievements):
      if len(task) > 1:
        self.pmi_reweigh[i] = 0.001
      else:
        task_desc = task[0].split("_")
        try:
          task_count = int(task_desc[-1])
          self.pmi_reweigh[i] = 0.001
        except:
          pass

  def _specify_curri_task(self):
    self.task_idx = np.random.choice(np.arange(len(self.target_achievements) + DUMMY_TASKS), size=1, p=self.task_probs)[0]
    self.task_steps = 0
    self.task_enc = self._encode_task(self.task_idx)

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
    # PMI: reweigh according to interestingness
    probs = probs * self.pmi_reweigh
    probs = probs / np.sum(probs)  # normalize
    self.task_probs = probs
