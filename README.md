# Implementation: BCQ

My implementation code of Batch Constrained Q-learning (BCQ)


[Off-Policy Deep Reinforcement Learning without Exploration, 2019 ICML](https://arxiv.org/pdf/1812.02900.pdf)


# Overview

- BCQ is an algorithm that limits the policy through VAE so that the target policy can traverse only in batches in order to prevent extrapolation errors that occur from mismatch of transitions in off-policy learning.
- BCQ is trained in an off-line setting using samples generated in the process of training DDPG.
- BCQ is trained on the Hopper-v4 transition dataset in MuJoco.
