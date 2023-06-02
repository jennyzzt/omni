import os
import json

from utils.other import root_path


def preprocess_achievement(ach):
    # return proper human-readable sentences
    return ach.replace('_', ' ')

def get_task_predictions(setting='r'):
    with open(os.path.join(root_path, f'./moi_saved/preds_{setting}.json'), 'r') as jfile:
        return json.load(jfile)
