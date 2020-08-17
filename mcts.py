

from typing import List, Dict, NamedTuple
from gamecomponents import Action, ActionHistory, Player
from network import NetworkOutput, Network
import math
import numpy as np
from helpers import MinMaxStats, KnownBounds
import collections



MCTSConfig = collections.namedtuple('MCTSConfig', ['known_bounds', 'discount','num_simulations'])


# Node definition -- building block of the MCTS tree

class Node(object):
    
    # Class attributes common to all instances. 
    # These are parameters governing the search functions.
    root_dirichlet_alpha = 0.
    root_exploration_fraction = 0.
    pb_c_base = 0.
    pb_c_init = 0.
    
    def __init__(self, prior: float): 
        self.visit_count = 0        # total number of visits to this node
        self.to_play = 1            # player to play at this node
        self.prior = prior          # prior as provided by network
        self.value_sum = 0.         # total value of all visits
        self.children = {}          # dict of children indexed by actions
        self.hidden_state = None    # state as determined by the network
        self.reward = 0.            # reward as predicted by network
    
    def expanded(self) -> bool: 
        return len(self.children) > 0
    
    # expected value as determined by MCTS process
    def value(self) -> float: 
        if self.visit_count == 0:
            return 0.
        return self.value_sum / self.visit_count
    
    # populate node based on network generated information
    def expand_node(self, to_play: Player, 
                    actions: List[Action],
                    network_output: NetworkOutput): 
        self.to_play = to_play
        self.hidden_state = network_output.hidden_state
        self.reward = network_output.reward

        # the policy logits from the network act as a prior to enable the 
        # MCTS to focus on likely actions.
        # However, for the 2 actions of cartpole this is unnecessary.
        for action, p in network_output.policy_logits.as_probabilities():
            self.children[action] = Node(p)
            #self.children[action] = Node(0.5)
        
    # alter the prior to aid exploration    
    def add_exploration_noise(self):
        actions = list(self.children.keys())
        noise = np.random.dirichlet([Node.root_dirichlet_alpha] * len(actions)) 
        frac = Node.root_exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac
    
    # choose a child based on the UCB score 
    # - used during MCTS
    def select_child(self, discount: float, min_max_stats: MinMaxStats):
        _, action, child = max((self._ucb_score(child, discount, min_max_stats), action, child) for action, child in self.children.items())
        return action, child

    # calculate the UCB score used to select child 
    def _ucb_score(self, child, discount: float, min_max_stats: MinMaxStats) -> float:
        # essentially, this is a weighted combination of the prior and the value estimate 
        # coming from the UCB process. as the number of visits increases the value is 
        # increasingly biased towards the MCTS determined value
        pb_c = math.log((self.visit_count + Node.pb_c_base + 1) / Node.pb_c_base) + Node.pb_c_init
        pb_c *= math.sqrt(self.visit_count+1) / (child.visit_count + 1)
        prior_score = pb_c * child.prior 

        if child.visit_count > 0 and self.visit_count > 3:
            value_score = min_max_stats.normalize(child.reward + discount * child.value())
        else:
            value_score = 0.

        return prior_score + value_score


    # Select an action based on the number of recorded visits to the child nodes 
    # - used for the final decision
    def select_action_with_temperature(self, T: float, epsilon: float = 0.0):
        
        visit_counts = [(child.visit_count, action, child) for action, child in self.children.items()]
        
        tmp = [math.exp(c/T) for c,_, _ in visit_counts]
        exp_sum = sum(tmp)
        tmp = list(np.asarray(tmp)/exp_sum + epsilon)
        exp_sum = sum(tmp)
        
        n = np.random.choice(len(visit_counts), 1, p=np.asarray(tmp)/exp_sum)
 
        return visit_counts[n[0]][1], visit_counts[n[0]][2]


## high-level MCTS functions 

# backpropagate the value from leaf to the starting point accumulating discounted 
# rewards along the way
# - used after every MCTS simulation
def backpropagate(search_path: List[Node], value: float, to_play: Player, discount: float, min_max_stats: MinMaxStats): 
    for node in reversed(search_path):
        node.value_sum += value #if node.to_play == to_play else -value 
        node.visit_count += 1
        min_max_stats.update(node.value())
        
        value = node.reward + discount * value


# the MCTS process
def run_mcts(config: MCTSConfig, root: Node, action_history: ActionHistory, network: Network):
    
    min_max_stats = MinMaxStats(config.known_bounds)
 
    # Run a number of simulated futures
    for _ in range(config.num_simulations): 

        # start a new simulation starting from the root node 
        history = action_history.clone()
        node = root
        search_path = [node]
    
        # Run one simulation iteratively through the tree
        while node.expanded():
            action, node = node.select_child(config.discount, min_max_stats) 
            history.add_action(action)
            search_path.append(node)

        # We've reached a leaf, so fill it in using the network output.

        # Paper: Inside the search tree we use the dynamics function to obtain 
        # the next hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        network_output = network.recurrent_inference(parent.hidden_state, history.last_action()) 

        # update the leaf node using the network_output
        node.expand_node(history.to_play(), history.action_space(), network_output)

        # update the value totals along the path
        backpropagate(search_path, network_output.value, history.to_play(), config.discount, min_max_stats)

