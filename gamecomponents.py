

from typing import List, Dict, NamedTuple, Optional
import math
import random
import statistics

## object used to represent the actions
class Action(object):
    max_ind = 2
    def __init__(self, index: int): 
        self.index = index
    def __hash__(self): 
        return self.index
    def __eq__(self, other):
        return self.index == other.index
    def __gt__(self, other):
        return self.index > other.index

def RandomAction() -> Action:
    return Action(random.randint(0,Action.max_ind-1))


## Policy object containing policy information and providing basic calculations 
# (NB Not present in the original pseudocode)
class Policy(object):
    
    action_list = []
    
    def __init__(self, vals: Optional[List[float]] = None):
      
        if vals is None:
            self.content = [0.5 for _ in Policy.action_list]
        else:

            self.content = vals
            
    def as_list(self):
        return self.content
    
    def as_dict(self):
        return {a: self.content[0][a.__hash__()] for a in Policy.action_list}
    
    def items(self):
        return zip(Policy.action_list,self.content)
    
    def as_probabilities(self):
  
        # TODO
        # during training, got an overflow error here!!
        mina = statistics.mean(self.content)
        tmp = [math.exp(a-mina) for a in self.content]
        policy_sum = sum(tmp)

        tmp = [i/policy_sum for i in tmp]
        return zip(Policy.action_list,tmp)


## Not relevant for cartpole
class Player:
    pass


## Container for history of actions over a game instance
class ActionHistory(object):
    def __init__(self, history: List[Action], action_space_size: int): 
        self.history = list(history)
        self.action_space_size = action_space_size
        
    def clone(self):
        return ActionHistory(self.history, self.action_space_size)
    
    def add_action(self, action: Action): 
        self.history.append(action)
        
    def last_action(self) -> Action: 
        return self.history[-1]
    
    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]
    
    def to_play(self) -> Player: 
        return Player()

