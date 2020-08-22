# MuZeroNStep
 
Implementation of MuZero on Cartpole, but with the change that every call to the MCTS identifies the next N actions to take, rather than just the next step.  Cartpole is systematic with no opponent, so entirely predictable. Consequently, it is a good candidate for MuZero to learn the dynamics and predict actions further into the future.

In this case we used N=4. Experimentation suggests that you need to make sure the replay start points coincide with the steps where a true game image was read. So, N=4 fits with unrolling 4 steps during training. 
 

