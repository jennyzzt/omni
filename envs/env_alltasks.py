import re
import gym
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from crafter import constants
from crafter import engine
from crafter import objects
from crafter import worldgen
from crafter import env
from envs.env_utils import get_repeat_tasks, get_compound_tasks, ach_to_string


DiscreteSpace = gym.spaces.Discrete
BoxSpace = gym.spaces.Box
DictSpace = gym.spaces.Dict
BaseClass = gym.Env

OBJS = ['table', 'furnace']
MATS = ['wood', 'stone', 'coal', 'iron', 'diamond', 'drink']
TOOLS = ['sword', 'pickaxe']
INSTRS = ['collect', 'make', 'place']
COUNT = [str(i) for i in range(2, 11)]
ENC_ORDER = INSTRS + OBJS + MATS + TOOLS + COUNT


class Env(env.Env):

  def __init__(
      self, area=(64, 64), view=(9, 9), size=(64, 64),
      reward=True, length=1500, seed=None, **kwargs):
    super().__init__(area, view, size, reward, length, seed, **kwargs)
    self.target_achievements, self.isreptask = get_repeat_tasks(constants.achievements, counts=10)
    self.target_achievements = [[xs] for xs in self.target_achievements]
    self.max_taskcom = 2
    self.target_achievements += get_compound_tasks(constants.achievements, maxcomp=self.max_taskcom, naive=False)
    self.task_progress = np.zeros(self.max_taskcom)
    # task condition attributes
    self.task_idx = 0
    self.task_enc = np.zeros(len(ENC_ORDER))
    self.task_steps = 0
    self.past_achievements = None
    self.follow_achievements = None
    self.given_achievements = None
    self.length_achievements = None
    self._specify_task = self._specify_curri_task
    self.eval_tsr = np.zeros(len(self.target_achievements))  # evaluated task success rates

  @property
  def observation_space(self):
    return DictSpace({
      'task_enc': BoxSpace(0, 1, (len(ENC_ORDER), ), np.uint8),
    })

  def reset(self):
    center = (self._world.area[0] // 2, self._world.area[1] // 2)
    self._episode += 1
    self._step = 0
    self._world.reset(seed=hash((self._seed, self._episode)) % (2 ** 31 - 1))
    self._update_time()
    self._player = objects.Player(self._world, center)
    self._world.add(self._player)
    self._unlocked = set()
    worldgen.generate_world_og(self._world, self._player)

    self.past_achievements = self._player.achievements.copy()
    self.follow_achievements = {ach_to_string(k): 0 for k in self.target_achievements}
    self.given_achievements = {ach_to_string(k): 0 for k in self.target_achievements}
    self.length_achievements = {ach_to_string(k): 0 for k in self.target_achievements}
    self._specify_task()
    self.update_given_ach()
    self.task_progress = np.zeros(self.max_taskcom)
    return self._obs(), {}

  def update_given_ach(self):
    self.given_achievements[ach_to_string(self.target_achievements[self.task_idx])] += 1

  def step(self, action):
    obs, reward, done, other_done, info = super().step(action)
    # additional info
    info['given_achs'] = self.given_achievements.copy()
    info['follow_achs'] = self.follow_achievements.copy()
    info['length_achs'] = self.length_achievements.copy()
    return obs, reward, done, other_done, info

  def _encode_task(self, task_idx):
    encoding = np.zeros((self.max_taskcom, len(ENC_ORDER)))
    if self.task_idx < len(self.target_achievements):
      comtask = self.target_achievements[self.task_idx]
      for n, task in enumerate(comtask):
        task_words = task.split('_')
        # bag of words encoding
        for i, word in enumerate(ENC_ORDER):
          if word in task_words:
            encoding[n, i] = 1
    return encoding

  def _decode_task(self, task_enc):
    taskname = ''
    for n, tenc in enumerate(task_enc):
      taskname += ' '.join([ENC_ORDER[int(i)] for i, c in enumerate(tenc) if c])
      if n < len(task_enc) - 1 and any(task_enc[n+1]):
        taskname += ' and '
    return taskname

  def _specify_curri_task(self):
    # choose random next task
    self.task_idx = np.random.choice(np.arange(len(self.target_achievements)), size=1)[0]
    self.task_steps = 0
    self.task_enc = self._encode_task(self.task_idx)

  def _specify_eval_task(self):
    # choose next task
    self.task_idx = self.eval_task_seq[self.eval_id]
    self.eval_id += 1
    if self.eval_id >= len(self.target_achievements):
      self.eval_id %= len(self.target_achievements)
      np.random.shuffle(self.eval_task_seq)
    self.task_steps = 0
    self.task_enc = self._encode_task(self.task_idx)

  def render(self, size=None, semantic=False, add_desc=False):
    canvas = super().render(size=size, semantic=semantic)
    if not semantic and add_desc:
      img = Image.fromarray(canvas)
      draw = ImageDraw.Draw(img)
      font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 25)
      draw.text((0, 0), self._decode_task(self.task_enc), (255,255,255), font=font)
      draw.text((0, 20), self._player.action,(255,255,255), font=font)
      canvas = np.asarray(img)
    return canvas

  def _get_reward(self):
    reward = 0

    task_failed = True
    if self.task_idx < len(self.isreptask) and self.isreptask[self.task_idx]:
      # repeat task
      task_desc = self.target_achievements[self.task_idx][0]
      task_words = task_desc.split('_')
      subtask_desc = '_'.join(task_words[:-1])
      if self._player.achievements[subtask_desc] - self.past_achievements[subtask_desc] > 0:
        self.task_progress[0] += 1.0 / float(task_words[-1])
      if self.task_progress[0] >= 1.0:
        reward += 1.0
        self.follow_achievements[task_desc] += 1
        self.length_achievements[task_desc] += self.task_steps
        self._specify_task()
        self.update_given_ach()
        self.task_progress = np.zeros(self.max_taskcom)
        task_failed = False
    elif self.task_idx < len(self.target_achievements):
      task_desc = self.target_achievements[self.task_idx]
      for i, tdesc in enumerate(task_desc):
        if not self.task_progress[i] and self._player.achievements[tdesc] - self.past_achievements[tdesc] > 0:
          self.task_progress[i] = 1
          break  # only 1 task can be done at 1 timestep
      if sum(self.task_progress) == len(self.target_achievements[self.task_idx]):
        # agent successfully completed given task
        reward += 1.0
        self.follow_achievements[ach_to_string(task_desc)] += 1
        self.length_achievements[ach_to_string(task_desc)] += self.task_steps
        self._specify_task()
        self.update_given_ach()
        self.task_progress = np.zeros(self.max_taskcom)
        task_failed = False

    if task_failed:
      # increase task step, check if agent is taking too long to complete given task
      self.task_steps += 1
      if self.task_steps > 300:
        self._specify_task()
        self.update_given_ach()
        self.task_progress = np.zeros(self.max_taskcom)

    self.past_achievements = self._player.achievements.copy()
    return reward

  def _obs(self):
    return {
      'task_enc': self.task_enc,
    }

  def set_curriculum(self, train=False):
    if train:
      self._specify_task = self._specify_curri_task
    else:
      self.eval_task_seq = np.arange(len(self.target_achievements))
      np.random.shuffle(self.eval_task_seq)
      self.eval_id = 0
      self._specify_task = self._specify_eval_task
