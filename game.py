
import gym
from typing import List
from gym.spaces import Discrete
from mcts import Node
from gamecomponents import Action, Player, ActionHistory, Policy
from helpers import TFMinMaxStats
import numpy as np


gameStateMinMax = TFMinMaxStats([4])

## functions as interface with the environment and a record of the subsequent game
class Game(object):

    environment = [] 
    
    def __init__(self, discount: float): 
        self.environment = gym.make(Game.environment)
        self.history = [] # history of actions taken in this game
        self.discount = discount # local copy of the discount to be applied to rewards
        self.action_space_size = self.environment.action_space.n # count of the number of action types

        self.action_list = {Action(i):i for i in range(self.action_space_size)} # list of the abstract action types and corresponding index used internally
        
        self.recorded_states = [self.environment.reset()] # record of external observations at each step
        self.rewards = [] # rewards at each step
        self.done = False # is it game over?

        self.child_visits = [] # a list of the proportion of visits for each child during MCTS (indexed by action)
        self.root_values = [] # value of each node as calculated during the MCTS

        
    # (for cartpole, the actions are just integers, so the abstract representation 
    # is a little overkill, but allows generalisation)

    ##        

    # legal actions as defined by the environment
    def legal_actions(self) -> List[Action]:
        return self.action_list
      
    ##

    # whose turn is it? Not relevant for cartpole
    def to_play(self) -> Player: 
        return Player()  
    
    # return the observation of the external environment at a certain step
    # for cartpole, we just provide the state as-is (normalised)
    def make_image(self, state_index: int): 
        
        return gameStateMinMax.normalize(self.recorded_states[state_index])
        
    # apply the action to the environment and record the results (reward and new state)
    def apply(self, action: Action):

        new_state, reward, self.done, _ = self.environment.step((self.action_list[action]))  
        self.history.append(action)
        self.rewards.append(reward)

        gameStateMinMax.update([new_state])
        
        self.recorded_states.append(new_state) 
        
    # is it over, yet?
    def terminal(self) -> bool:
        return self.done
     
    ##


    def store_search_statistics(self, root: Node):

        sum_visits = sum(child.visit_count for child in root.children.values()) 
        action_space = (Action(index) for index in range(self.action_space_size)) 
        self.child_visits.append([root.children[a].visit_count / sum_visits if a in root.children 
                                  else 0 
                                  for a in action_space])
        self.root_values.append(root.value())

    ##

    def update_values(self, value_list: List[float]):
        self.root_values = value_list

    ##

    # packages up all the essential information to allow training of the network
    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player = 0):
        
        # Paper: The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then. 

        targets = []

        flat_child_visits = [1./self.action_space_size for _ in range(self.action_space_size)]

        # step over the required unroll steps
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
           
            # calculate the value of the current state with TD
            # start at the point td_steps in the future
            bootstrap_index = current_index + td_steps 
            # find the value, accounting for discounting
            # if we are off the end of the game then return zero
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index][0] * self.discount**td_steps 
            else:
                value = 0.

            # add up all the rewards between now and the td_steps position, with discounting

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]): 

                value += reward * self.discount**i 

            # Paper: For simplicity the network always predicts the most recently received 
            # reward, even for the initial representation network where we already 
            # know this reward.

            if current_index > state_index and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1] 
            else:
                last_reward = 0.
            

            if current_index < len(self.root_values):
                targets.append((value, last_reward, self.child_visits[current_index], True))
            else:
                # Paper: States past the end of games are treated as absorbing states.
                targets.append((0., last_reward, flat_child_visits, current_index == len(self.root_values)))
                # paper has empty array for policy but I need something to make the 
                # tensors the right size

        return targets
    
    
    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)
    
    def length(self) -> int:
        return len(self.root_values)
    

    ##

