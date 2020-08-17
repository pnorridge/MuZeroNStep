
from game import Game, gameStateMinMax
import numpy.random as random
import pickle
import numpy as np

# Replay buffer stores old games and extracts random batches from the stored set
class ReplayBuffer():
    def __init__(self):
        self.window_size = 1000
        self.batch_size = 32
        self.buffer = []
        self.game_len = []
    
    def save_game(self, game: Game):
        if len(self.buffer) > self.window_size:
            n = self.game_len.index(min(self.game_len))
            # n = 0
            self.buffer.pop(0)
            self.game_len.pop(0)
        self.buffer.append(game)
        self.game_len.append(game.length())

    # Select a random game from the buffer
    def sample_game(self) -> Game:
        return random.choice(self.buffer)

    def sample_game_with_bias(self) -> Game:
        bias = np.array(range(1,len(self.buffer)+1))+25
        return random.choice(self.buffer, p = bias/sum(bias))

    
    # Identify a suitable game position.    
    def sample_position(self, game: Game) -> int:
        # Paper: Sample position from game either uniformly or according to some priority. 
        return random.randint(0,game.length()/8)*8

    # Extract one batch from the stored games
    def sample_batch(self, num_unroll_steps: int, td_steps: int, with_bias = False, with_target = False): 
        
        # select a random selection of games
        if with_bias:
            games = [self.sample_game_with_bias() for _ in range(self.batch_size)] 
        else:    
            games = [self.sample_game() for _ in range(self.batch_size)] 
         
        # for each game select a random starting position
        game_pos = [(g, self.sample_position(g)) for g in games] 

        # for each game and position, return the initial external environment observation, 
        # the list of actions taken and the target for the training. 
        batch = [(g.make_image(i), g.history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, g.to_play())) 
                for (g, i) in game_pos]
        if with_target:
            (g,i) = game_pos[-1]
            batch.append((g.make_endgame_image(), g.history[i:i + num_unroll_steps],
                          g.make_endgame_target(i, num_unroll_steps, td_steps, g.to_play()) ))
            
        return batch

    def save(self, filename: str):

        outfile = open(filename, 'wb')
        pickle.dump(self.buffer, outfile)
        pickle.dump(self.game_len, outfile)
        pickle.dump(gameStateMinMax, outfile)
        outfile.close()

    def load(self, filename: str):

        infile = open(filename, 'rb')
        self.buffer = pickle.load(infile)
        self.game_len = pickle.load(infile)
        gameStateMinMax = pickle.load(infile)
        infile.close()


