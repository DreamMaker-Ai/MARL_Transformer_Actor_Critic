# 19_MTC_SAC_Hierarchy

## Hierarchy Architecture:
### - Seperior: Commander
### - Sabodinate: Local agents (platoons or companies)


Copied from "18_MTC_SAC_SelfPlay_2", then modified.

- POMDP agents, fov=2, com=2
- Individual reward
- SelfPlay from scratch
- Blue agents strategy: stationary or random choice from a pool of checkpoints of red agents
- Evaluation during the training: blue agents are stationary