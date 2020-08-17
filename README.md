# MuZeroTwoStep
 
This attempts to implement MuZero on AI Gym, but with the change that every call to the MCTS identifies the next two actions to take, rather than just the next step. 

Experimentation suggests that you need to make sure the replay start points coincide with the steps where a true game image was read.
 