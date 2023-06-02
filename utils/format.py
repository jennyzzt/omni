import os
import json
import numpy as np
import re
import torch
import torch_ac

import utils


def get_obss_preprocessor(obs_space):
    prep_obs_space = {}
    for key in obs_space.keys():
        prep_obs_space[key] = obs_space[key].shape

    def preprocess_obss(obss, device=None):
        prep_obss = dict()
        for key in obss[0].keys():
            if key == 'image':
                prep_obss[key] = preprocess_images([obs['image'] for obs in obss], device=device)
            else:
                prep_obss[key] = preprocess_totensor([obs[key] for obs in obss], device=device)
        return torch_ac.DictList(prep_obss)

    return prep_obs_space, preprocess_obss


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = np.array(images) / 255.0
    return torch.tensor(images, device=device, dtype=torch.float)


def preprocess_totensor(xs, device=None):
    xs = np.array(xs)
    return torch.tensor(xs, device=device, dtype=torch.float)
