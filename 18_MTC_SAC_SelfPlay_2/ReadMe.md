## Copied from 16_MTC_SAC_POMDP_SelfPlay, then modified.

 - Individual reward in "16_MTC_SAC_POMDP_SelfPlay".
 - Use global state for the training.
 - For the workers, initial network for the red agents is 100K trained network in "15_MATC_SAC_DecPOMDP_2". The initial network for the blue agents is stationary or the 100K trained network. The checkpoints will be added to the pool of networks.
 - For the tester, use the 440K trained network in "16_MTC_SAC_POMDP_SelfPlay" for the blue_agents.
 - For the learner, the initial network s defined in config_selfplay_2.py. (The 100K trained network).The checkpoints for the pool of networks are saved here. 