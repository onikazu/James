import gym
import numpy as np
import logging
from memory import Memory
from actor import Actor
from critic import Critic

logging.basicConfig(filename='logger.log', level=logging.INFO)
env = gym.make('MountainCarContinuous-v0')
# print('action space high:{}'.format(env.action_space.high))
# print('action space low:{}'.format(env.action_space.low))
# print('observation space high:{}'.format(env.observation_space.high))
# print('observation space low:{}'.format(env.observation_space.low))

# action space high:[ 1.]
# action space low:[-1.]
# observation space high:[ 0.6   0.07]
# observation space low:[-1.2  -0.07]
memory = Memory()

# pretrain
env.reset()
# 行動はすべてランダムに決定
action = env.action_space.sample()
state, reward, done, _ = env.step(action)
for i in range(1, 100):
    # env.render()
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    # print(i, action, state, reward, done)
    memory.add((state, action, reward, next_state))
    # ミス？
    next_state = state
    if done:
        break

actor = Actor(env.action_space, env.observation_space)
critic = Critic(env.action_space, env.observation_space, actor.sess)
for ep in range(1000):
    # batch train
    total_reward = 0
    env.reset()
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    for _ in range(1000):
        # training
        states, actions, rewards, next_states = memory.sample(20)
        next_actions = actor.get_actions(next_states)
        next_qs = critic.get_qs(next_states, next_actions)
        loss, q = critic.train(states, actions, rewards, next_qs)
        action_gradients = critic.get_action_gradients(states, actions)
        actor.train(states, action_gradients[0])

        env.render()
        action = actor.get_action_for_train(state, ep)
        next_state, reward, done, _ = env.step(action)
        memory.add((state, action, reward, next_state))
        # print(state, action, reward, next_state)
        total_reward += reward
        # print(action, reward, total_reward)
        state = next_state
        if done:
            break
    # if ep % 10 == 0:
    # critic.update_network_params()
    logging.info('Episode: {}'.format(ep) +
                 ' Total Reward: {:.4f}'.format(total_reward) +
                 ' Q: {:.4f}'.format(np.max(q)) +
                 ' loss: {:.4f}'.format(loss))

actor.save()