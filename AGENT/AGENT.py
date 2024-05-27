import numpy as np
import torch
import random

from AGENT.NET import Actor, Critic


class AGENT:
    def __init__(self, par):
        self.par = par
        self.eval_actor, self.target_actor = (Actor(self.par.dim_state, self.par.dim_action),
                                              Actor(self.par.dim_state, self.par.dim_action))
        self.eval_critic, self.target_critic = (Critic(self.par.dim_state, self.par.dim_action),
                                                Critic(self.par.dim_state, self.par.dim_action))

        # self.learn_step_counter = 0
        self.memory_counter = 0
        self.buffer = []

        self.target_actor.load_state_dict(self.eval_actor.state_dict())
        self.target_critic.load_state_dict(self.eval_critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.eval_actor.parameters(), lr=self.par.lr_ac)
        self.critic_optim = torch.optim.Adam(self.eval_critic.parameters(), lr=self.par.lr_cr)

    def choose_action(self, state):
        if np.random.uniform() > self.par.epsilon:
            action = np.random.uniform(-1, 1, 1)
        else:
            inputs = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            action = self.eval_actor(inputs).squeeze(0)
            action = action.detach().numpy()
        return action

    def store_transition(self, *transition):
        if len(self.buffer) == self.par.memory_capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self):

        if len(self.buffer) < self.par.batch_size:
            return

        samples = random.sample(self.buffer, self.par.batch_size)

        s0, a0, r1, s1 = zip(*samples)

        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.par.batch_size, -1)
        s1 = torch.tensor(s1, dtype=torch.float)

        def critic_learn():
            a1 = self.target_actor(s1).detach()
            y_true = r1 + self.par.gamma * self.target_critic(s1, a1).detach()

            y_pred = self.eval_critic(s0, a0)

            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        def actor_learn():
            loss = -torch.mean(self.eval_critic(s0, self.eval_actor(s0)))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

        def soft_update(net_target, net, tau):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        critic_learn()
        actor_learn()
        soft_update(self.target_critic, self.eval_critic, self.par.tau)
        soft_update(self.target_actor, self.eval_actor, self.par.tau)
