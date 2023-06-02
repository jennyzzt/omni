import numpy as np
from itertools import combinations


SYNONYMS = {
  'collect': [
    'gather',
    'harvest',
    'procure',
    'acquire',
    'amass',
  ],
  'make': [
    'craft',
    'acquire',
    'build',
    'construct',
    'create',
    ],
  'place': [
    'put',
    'deploy',
    'install',
    'putdown',
    'position',
  ],
}


def ach_to_string(xs):
  if isinstance(xs, str):
    return xs
  return '__and__'.join(xs)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def get_repeat_tasks(tasks, counts=5):
  counts = [counts]*len(tasks) if isinstance(counts, int) else counts
  alltasks = []
  isreptask = []
  for i, task in enumerate(tasks):
    for c in range(1, counts[i]+1):
      if c == 1:
        alltasks.append(task)
        isreptask.append(0)
      elif c > 1:
        alltasks.append(f'{task}_{c}')
        isreptask.append(1)
      else:
        continue
  return alltasks, isreptask

def get_synonym_tasks(tasks, nsyns=-1):
  alltasks = []
  basetasks = []
  for ach in tasks:
    ach_words = ach.split('_')
    btask = '_'.join(ach_words)
    alltasks.append(btask)
    basetasks.append(btask)
    syns = SYNONYMS[ach_words[0]]
    syns = syns if nsyns < 0 else syns[:nsyns]
    for sn in syns:
      alltasks.append('_'.join([sn, *ach_words[1:]]))
      basetasks.append(btask)
  return alltasks, basetasks

def get_compound_tasks(tasks, maxcomp=2, naive=True):
  alltasks = []
  newalltasks = []
  if naive:
    # if naive=True, return repetitions and different orders as different tasks
    for i in range(maxcomp):
      newalltasks = [[xs] for xs in tasks]
      for comptask in alltasks:
        for ach in tasks:
          newalltasks.append([*comptask, ach])
      alltasks = newalltasks
      newalltasks = []
  else:
    # else, don't return repetitions and different orders are the same task
    for i in range(2, maxcomp + 1):
      alltasks += [list(xs) for xs in combinations(tasks, i)]
  return alltasks
