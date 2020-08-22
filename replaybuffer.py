
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

    
    # Identify a suitable game position.    
    def sample_position(self, game: Game, prediction_steps: int) -> int:
        # Paper: Sample position from game either uniformly or according to some priority. 
        if game.length() > prediction_steps-1:
            return random.randint(0, game.length()/prediction_steps)*prediction_steps
        else:
            return 0

    # Extract one batch from the stored games
    def sample_batch(self, num_unroll_steps: int, td_steps: int, prediction_steps: int): 
        
        # select a random selection of games
        games = [self.sample_game() for _ in range(self.batch_size)] 
         
        # for each game select a random starting position
        game_pos = [(g, self.sample_position(g)) for g in games] 

        # for each game and position, return the initial external environment observation, 
        # the list of actions taken and the target for the training. 
        batch = [(g.make_image(i), g.history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, g.to_play())) 
                for (g, i) in game_pos]

            
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


