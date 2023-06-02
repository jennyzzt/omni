import torch

import utils
from .other import device
from model import ACModel


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, preprocess_obss, acmodel, argmax, num_envs):
        self.preprocess_obss = preprocess_obss
        self.acmodel = acmodel
        self.argmax = argmax
        self.num_envs = num_envs
        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)
        self.acmodel.eval()

    @classmethod
    def dir_init(cls, obs_space, action_space, model_dir,
                 argmax=False, num_envs=1, model_suffix=""):
        obs_space, preprocess_obss = utils.get_obss_preprocessor(obs_space)
        acmodel = ACModel(obs_space, action_space)
        acmodel.load_state_dict(utils.get_model_state(model_dir, model_suffix))
        acmodel.to(device)
        return cls(preprocess_obss, acmodel, argmax, num_envs)

    @classmethod
    def model_init(cls, obs_space, acmodel, num_envs=1):
        _, preprocess_obss = utils.get_obss_preprocessor(obs_space)
        acmodel = acmodel
        argmax = False
        num_envs = num_envs
        return cls(preprocess_obss, acmodel, argmax, num_envs)

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
