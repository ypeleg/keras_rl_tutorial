
from useful_stuff import *

env = JoypadSpace(gym_super_mario_bros.make('SuperMarioBros-v0'), SIMPLE_MOVEMENT)
nb_actions = env.action_space.n
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

model = build_model(input_shape, nb_actions)

memory = SequentialMemory(
    limit=1000000,
    window_length=WINDOW_LENGTH)

dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    memory=memory,
    processor=SimpleProcessor())
dqn.compile('adam')
dqn.fit(env, nb_steps=1750000, visualize=True)
