import gym

env = gym.make('MountainCarContinuous-v0')
env.reset()
action = env.action_space.sample()
print(action)
state, reward, done, _ = env.step(action)

print(state, reward, done, _)
print(env.action_space, env.observation_space)

"""
action float （hitotu）
state リストで状態変数(2つ)
reward 報酬（float）
done 
_ info(必要なさそう、dict)
env.action_space, env.observation_space Box


"""


from box import Box

# dictionary
users = {"name": "Michel", "age": 34, "city": "New York"}
# 変数'user_box'内でBoxオブジェクトの作成
user_box = Box(users)

print(user_box.name) # Michel
print(user_box)
