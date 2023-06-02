import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, acsize=128, activation='tanh'):
        super().__init__()

        self.rnn_input_size = 0

        # Define image embedding
        if 'image' in obs_space.keys():
            self.image_embedding_size = 256
            image_conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )
            # Compute shape by doing one forward pass
            with torch.no_grad():
                x, y, z = obs_space['image']
                n_flatten = image_conv(
                    torch.zeros((z, x, y)).unsqueeze(0)
                ).shape[1]
            self.image_extractor = nn.Sequential(
                image_conv,
                nn.Linear(n_flatten, self.image_embedding_size),
                nn.ReLU()
            )
            self.rnn_input_size += self.image_embedding_size

        # Define taskenc embedding
        if 'task_enc' in obs_space.keys():
            self.taskenc_extractor = nn.Flatten()
            self.rnn_input_size += np.product(obs_space['task_enc'])

        # Define memory
        self.memory_rnn = nn.LSTMCell(self.rnn_input_size, self.semi_memory_size)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size

        if activation == 'tanh':
            # Define actor's model
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, acsize),
                nn.Tanh(),
                nn.Linear(acsize, action_space.n)
            )

            # Define critic's model
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, acsize),
                nn.Tanh(),
                nn.Linear(acsize, 1)
            )
        elif activation == 'relu':
            # Define actor's model
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, acsize),
                nn.ReLU(),
                nn.Linear(acsize, action_space.n)
            )

            # Define critic's model
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, acsize),
                nn.ReLU(),
                nn.Linear(acsize, 1)
            )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        try:
            return self.image_embedding_size
        except:
            return self.rnn_input_size

    def forward(self, obs, memory):
        x_inputs = []

        if 'image' in obs.keys():
            x_image = obs.image.transpose(1, 3).transpose(2, 3)
            x_image = self.image_extractor(x_image)
            x_image = x_image.reshape(x_image.shape[0], -1)
            x_inputs.append(x_image)

        if 'task_enc' in obs.keys():
            x_taskenc = obs.task_enc
            x_taskenc = self.taskenc_extractor(x_taskenc)
            x_taskenc = x_taskenc.reshape(x_taskenc.shape[0], -1)
            x_inputs.append(x_taskenc)

        x = torch.concat(x_inputs, axis=-1)

        hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
        hidden = self.memory_rnn(x, hidden)
        embedding = hidden[0]
        memory = torch.cat(hidden, dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory
