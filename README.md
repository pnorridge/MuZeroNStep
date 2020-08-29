# MuZeroNStep
 
Implementation of MuZero on Cartpole, but with the change that every call to the MCTS is used to identify the next N actions to take, rather than just the next step.  This is possible with a MuZero-like approach because it is learning the game dynamics, so should be able to predict the game state without being fed the true game images/state at every step. Cartpole is systematic with no opponent, so entirely predictable. Consequently, it is a good candidate for this approach.

In this case we used N=4. So, the game state is provided to the algorithm only every 4 steps.

Experimentation suggests that - at least durng the initial training steps - things work better when the replay samples start at game steps where the agent was fed a true game state. So, in this case, the replay buffer has sample positions 4k (k an integer). Possibly, this can be relaxed after the initial convergence (see plot). 

Basic code follows the psuedocode from 'Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model' (arXiv:1911.08265v2). Most notable change is to the training function so that it does more in parallel and minimises the number of transfers to the GPU. Also, added some more conventional exploration via a small epsilon in the action selection.
 
Some example results with CartpoleV1 (total reward for each game run during training)

![MuZeroNStep results](/img/results1.jpg)

Obviously not an entirely stable solution, but not bad considering the task we are asking it to do.
