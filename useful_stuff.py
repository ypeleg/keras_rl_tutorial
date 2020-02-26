import gym
import numpy as np
from PIL import Image
import keras.backend as K
import gym_super_mario_bros
from rl.core import Processor
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from rl.agents.dqn import DQNAgent
from keras.models import Sequential
from rl.memory import SequentialMemory
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy

def build_model(input_shape, nb_actions):
    inp = Input(input_shape)
    out = Permute([3, 2, 1])(inp)
    out = Convolution2D(32, (3, 3))(out)
    out = Activation('relu')(out)
    out = Convolution2D(32, (3, 3))(out)
    out = Activation('relu')(out)
    out = Flatten()(out)
    out = Dense(nb_actions)(out)
    out = Activation('linear')(out)
    model = Model(inp, out)
    return model

class SimpleProcessor(Processor):
    def process_observation(self, observation):
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.0
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4