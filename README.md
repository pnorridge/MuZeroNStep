# MuZeroNStep
 
This attempts to implement MuZero on Cartpole, but with the change that every call to the MCTS identifies the next N actions to take, rather than just the next step. (In this case N = 8). Cartpole is systeamtic with no opponent, so entrirely predictable. Consequently, it is a good candidate for MuZero to learn the dynamics and predict actions farther than one step ahead.

Experimentation suggests that you need to make sure the replay start points coincide with the steps where a true game image was read.
 
