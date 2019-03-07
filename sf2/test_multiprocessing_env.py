import os
import random

import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import retro                 # Retro Environment
from retro.retro_env import RetroEnv

from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames



# from IPython import display

from collections import deque # Ordered collection with ends

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
#warnings.filterwarnings('ignore') 

import gym


from multiprocessing import Process
from multiprocessing import Pool

"""
This module exposes the RetroWrapper class.
"""
import multiprocessing
import retro
import gc


class SF2Env(RetroEnv):
    KEY_LIST = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
    
    def __init__(self, game, state=retro.State.DEFAULT, scenario=None, info=None, use_restricted_actions=retro.Actions.FILTERED,
                 record=False, players=1, inttype=retro.data.Integrations.STABLE, obs_type=retro.Observations.IMAGE, press_button_print=False):
        # action 입력은 DISCRETE 모드
        use_restricted_actions = retro.Actions.DISCRETE
        self.press_button_print = press_button_print
        
        RetroEnv.__init__(self, game, state, scenario, info, use_restricted_actions,
                 record, players, inttype, obs_type)
        # retro의 버그 : scenario.json의 actions를 인식하지 못하기 때문에 수동으로 세팅
        self.button_combos = self.data.valid_actions()
        if self.use_restricted_actions == retro.Actions.DISCRETE:
            combos = 1
            for combo in self.button_combos:
                combos *= len(combo)
            self.action_space = gym.spaces.Discrete(combos ** players)
        elif self.use_restricted_actions == retro.Actions.MULTI_DISCRETE:
            self.action_space = gym.spaces.MultiDiscrete([len(combos) if self.gym_version >= (0, 9, 6) else (0, len(combos) - 1) for combos in self.button_combos] * players)
        else:
            self.action_space = gym.spaces.MultiBinary(self.num_buttons * players)
    
    def step(self, a):
        ob, rew, done, info = RetroEnv.step(self, a)
        
        if self.press_button_print:
            print(self.action_array_to_keys(self.action_to_array(a)))
        
        return self.get_state_from(), rew, done, info        
    
    def reset(self):
        RetroEnv.reset(self)
        return self.get_state_from()
            
    def action_array_to_keys(self, action_array):
        press_keys = []
        for i,v in enumerate(action_array[0]):
            if v == 1:
                press_keys.append(self.KEY_LIST[i])
        return press_keys
    
    # env로부터 state 값을 변환
    def get_state_from(self):
        state = [0] * (188*71 + 2)
        
        distance_x_between_players = int(self.data.lookup_value('distance_x_between_players'))
        distance_y_between_players = int(self.data.lookup_value('distance_y_between_players'))
        
        distance_inx = distance_x_between_players * (distance_y_between_players + 1)
        # TODO 유효하지 않은 인덱스가 없도록 처리
        if distance_inx >= len(state):
            state[0] = 1
        else:
            state[distance_inx] = 1
        
        first_player_x = int(self.data.lookup_value('first_player_x'))
        second_player_x = int(self.data.lookup_value('second_player_x'))
        
        left = 0
        right = 0
        if first_player_x > second_player_x:
            right = 1
        elif first_player_x < second_player_x: 
            left = 1
        state[188*71] = left
        state[188*71 + 1] = right
        
        return np.asarray(state)

    
rom_path = '/Users/metsmania/Works/sf2-env/StreetFighterIISpecialChampionEdition-Genesis'


def make(game, **kwargs):
    return SF2Env(game, 
                   state='rvsb.state', 
                   scenario='scenario',
                   players=1,
                   use_restricted_actions=retro.Actions.DISCRETE)



MAKE_RETRIES = 5

def set_retro_make( new_retro_make_func ):
    RetroWrapper.retro_make_func = new_retro_make_func

def _retrocom(rx, tx, game, kwargs):
    """
    This function is the target for RetroWrapper's internal
    process and does all the work of communicating with the
    environment.
    """
    env = RetroWrapper.retro_make_func(game, **kwargs)

    # Sit around on the queue, waiting for calls from RetroWrapper
    while True:
        attr, args, kwargs = rx.get()

        # First, handle special case where the wrapper is asking if attr is callable.
        # In this case, we actually have RetroWrapper.symbol, attr, and {}.
        if attr == RetroWrapper.symbol:
            result = env.__getattribute__(args)
            tx.put(callable(result))
        elif attr == "close":
            env.close()
            break
        else:
            # Otherwise, handle the request
            result = getattr(env, attr)
            if callable(result):
                result = result(*args, **kwargs)
            tx.put(result)


class RetroWrapper():
    """
    This class is a thin wrapper around a retro environment.
    The purpose of this class is to protect us from the fact
    that each Python process can only have a single retro
    environment at a time, and we would like potentially
    several.
    This class gets around this limitation by spawning a process
    internally that sits around waiting for retro environment
    API calls, asking its own local copy of the environment, and
    then returning the answer.
    Call functions on this object exactly as if it were a retro env.
    """
    symbol = "THIS IS A SPECIAL MESSAGE FOR YOU"
    #retro_make_func = retro.make
    retro_make_func = make

    def __init__(self, game, **kwargs):
        tempenv = None
        retry_counter = MAKE_RETRIES
        while True:
            try:
                tempenv = RetroWrapper.retro_make_func(game, **kwargs)
            except RuntimeError: # Sometimes we need to gc.collect because previous tempenvs haven't been cleaned up.
                gc.collect()
                retry_counter -= 1
                if retry_counter > 0:
                    continue
            break

        if tempenv == None:
            raise RuntimeError( 'Unable to create tempenv' )

        tempenv.reset()

        if hasattr( tempenv, 'unwrapped' ): # Wrappers don't have gamename or initial_state
            tempenv_unwrapped = tempenv.unwrapped
            self.gamename = tempenv_unwrapped.gamename
            self.initial_state = tempenv_unwrapped.initial_state

        self.action_space = tempenv.action_space
        self.metadata = tempenv.metadata
        self.observation_space = tempenv.observation_space
        self.reward_range = tempenv.reward_range
        tempenv.close()

        self._rx = multiprocessing.Queue()
        self._tx = multiprocessing.Queue()
        self._proc = multiprocessing.Process(target=_retrocom, args=(self._tx, self._rx, game, kwargs), daemon=True)
        self._proc.start()

    def __del__(self):
        """
        Make sure to clean up.
        """
        self.close()

    def __getattr__(self, attr):
        """
        Any time a client calls anything on our object, we want to check to
        see if we can answer without having to ask the retro process. Usually,
        we will have to ask it. If we do, we put a request into the queue for the
        result of whatever the client requested and block until it comes back.
        Otherwise we simply give the client whatever we have that they want.
        BTW: This doesn't work for magic methods. To get those working is a little more involved. TODO
        """
        # E.g.: Client calls env.step(action)
        ignore_list = ['class', 'mro', 'new', 'init', 'setattr', 'getattr', 'getattribute']
        if attr in self.__dict__ and attr not in ignore_list:
            # 1. Check if we have a step function. If so, return it.
            return attr
        else:
            # 2. If we don't, return a function that calls step with whatever args are passed in to it.
            is_callable = self._ask_if_attr_is_callable(attr)

            if is_callable:
                # The result of getattr(attr) is a callable, so return a wrapper
                # that pretends to be the function the user was trying to call
                def wrapper(*args, **kwargs):
                    self._tx.put((attr, args, kwargs))
                    return self._rx.get()
                return wrapper
            else:
                # The result of getattr(attr) is not a callable, so we should just
                # execute the request for the user and return the result
                self._tx.put((attr, [], {}))
                return self._tx.get()

    def _ask_if_attr_is_callable(self, attr):
        """
        Returns whether or not the attribute is a callable.
        """
        self._tx.put((RetroWrapper.symbol, attr, {}))
        return self._rx.get()

    def close(self):
        """
        Shutdown the environment.
        """
        if "_tx" in self.__dict__ and "_proc" in self.__dict__:
            self._tx.put(("close", (), {}))
            self._proc.join()



    

if __name__ == '__main__':
    env_1 = RetroWrapper(rom_path, 
                       state='rvsb.state', 
                       scenario='scenario',
                       players=1,
                       use_restricted_actions=retro.Actions.DISCRETE)

    possible_actions = np.array(list(range(1, env_1.action_space.n+1)))





    total_test_rewards = []
    
    # Load the model
    
    for episode in range(1):
        total_rewards = 0
        print('1')
        
        state = env_1.reset()
        
        t = 0
        while True:
            t += 1

            env_2 = RetroWrapper(rom_path, 
                     state='rvsb.state', 
                     scenario='scenario',
                     players=1,
                     press_button_print=False)
            env_2.reset()
            
            
            for i in range(10):
                print('sim_{}'.format(i))
                action = random.randint(0,len(possible_actions))
                    
                #Perform the action and get the next_state, reward, and done information
                next_state, reward, done, _ = env_2.step(action)
                env_2.render()
                
            env_2.close()
            

            action = random.randint(0,len(possible_actions))
            #Perform the action and get the next_state, reward, and done information
            next_state, reward, done, _ = env_1.step(action)
            env_1.render()
            print('real_{}'.format(t))
            
            total_rewards += reward

            if done:
                print ("Score", total_rewards)
                total_test_rewards.append(total_rewards)
                break
            
    env_1.close()