import gym
import visdom

from AGENT import AGENT
# from ENV.ENV import ENV
from UTILS.parameters import define_parameters


def train():
    for index_epi in range(par.max_episode):
        reward_epi = 0
        print("The training in the " + str(index_epi) + " episode is beginning.")
        state_now = env.reset()[0]
        for index_step in range(par.max_step):
            action = agent.choose_action(state_now)
            state_next, reward, _, done, info = env.step(action)
            reward_epi += reward
            agent.store_transition(state_now, action, reward, state_next)
            state_now = state_next
            agent.learn()
            if done or index_step >= par.max_step:
                print("The reward at episode " + str(index_epi) + " is " + str(reward_epi))
                if par.visdom_flag:
                    viz.line(X=[index_epi + 1], Y=[reward_epi], win='reward', opts={'title': 'reward'},
                             update='append')
                break


if __name__ == '__main__':
    par = define_parameters()
    if par.visdom_flag:
        viz = visdom.Visdom()
        viz.close()
    env = gym.make('Pendulum-v1')
    par.dim_action = env.action_space.shape[0]
    par.dim_state = env.observation_space.shape[0]
    agent = AGENT.AGENT(par)
    train()
