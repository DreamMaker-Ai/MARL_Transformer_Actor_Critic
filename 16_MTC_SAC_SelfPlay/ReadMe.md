# 16_MTC_SAC_POMDP_SelfPlay

## This is preliminary check for self_play

 - Use 440K trained neural network as blue_agents.
 - Initial network of red_agents is also 440K trained network.
 - trial_0: use 440K trained alpha as initial alpha
 - trial_1: use log(0.5) as initial alpha

## Copied from "14_MTC_SAC_POMDP_2", then modified.

 - Individual reward
 - Use global state for training