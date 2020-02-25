#!/usr/bin/env python
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import json
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf


def gray_and_resize(x_t1_colored):
    x_t1 = skimage.color.rgb2gray(x_t1_colored)
    x_t1 = skimage.transform.resize(x_t1, (80, 80))
    x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
    x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
    return x_t1



# MODE = 'Run'
MODE = 'Train'



N_ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
ANNEAL = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

img_rows , img_cols = 80, 80
img_channels = 4 #

def buildmodel():
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows,img_cols,img_channels)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    return model

def trainNetwork(model,args):
    env = game.GameState()
    D = deque()
    do_nothing = np.zeros(N_ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = env.frame_step(do_nothing)

    # x_t = skimage.color.rgb2gray(x_t)
    # x_t = skimage.transform.resize(x_t,(80,80))
    # x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))
    x_t = x_t.reshape( x_t.shape[1], x_t.shape[2])
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
    if args['mode'] == 'Run':
        OBSERVATION_PHASE = 999999999
        epsilon = FINAL_EPSILON
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)

    else:
        OBSERVATION_PHASE = 500
        epsilon = INITIAL_EPSILON




    t = 0
    while (True):
        a_t = np.zeros([N_ACTIONS])
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon: a_t[random.randrange(N_ACTIONS)] = 1            # Randomize action
            else: a_t[np.argmax(model.predict(s_t))] = 1                                   # max_a(Q(s,a))
        if t > OBSERVATION_PHASE: epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / ANNEAL    # epsilon Decay
        x_t1, r_t, terminal_state = env.step(a_t)                                          # Env -> (State, Reward, Terminal?)
        s_t1 = np.append(gray_and_resize(x_t1), s_t[:, :, :, :3], axis=3)                  # s_t1 - Current next state
        D.append((s_t, np.argmax(a_t), r_t, s_t1, terminal_state))                         # (State, Action, Reward, Next State, Terminal?) -> Replay Memory
        if t > OBSERVATION_PHASE:
            batch = random.sample(D, BATCH)
            X = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
            y = np.zeros((X.shape[0], N_ACTIONS))
            for i in range(0, len(batch)):
                state_t, action_t, reward_t, state_t1, terminal_state = batch[i]           # Replay Memory -> (State, Action, Reward, Next State, Terminal?)
                X[i:i + 1] = state_t                                                       # Training Samples: States
                y[i] = model.predict(state_t)                                              # Training Targets: max_a(Q(s,a)
                if terminal_state: y[i, action_t] = reward_t
                else: y[i, action_t] = reward_t + GAMMA * np.max(model.predict(state_t1))  # Bellman Equation: reward_t + GAMMA * max_a(Q(s,a))
            loss = model.fit(X, y, epochs=1)
        s_t = s_t1
        t = t + 1





        # if t <= OBSERVE:
        #     state = "observe"
        # elif t > OBSERVE and t <= OBSERVE + ANNEAL:
        #     state = "explore"
        # else:
        #     state = "train"
        # print("TIMESTEP", t, "/ STATE", state,  "/ EPSILON", epsilon, "/ ACTION", action, "/ REWARD", r_t, "/ Q_MAX " , np.max(Q_sa))

def playGame(args):
    model = buildmodel()
    trainNetwork(model,args)

def main():
    playGame({'mode':MODE})

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()
