# Policy Gradients

In this project, you will implement your own Policy Gradient in your loop of reinforcement learning (by using the Monte-Carlo policy gradient algorithm, also called `REINFORCE`).

## Resources

Read or watch:

* [How Policy Gradient Reinforcement Learning Works](https://intranet-dlh.hbtn.io/rltoken/_l5Zt_dz3HTJCQq1MRWcRQ)
* [Policy Gradients in a Nutshell](https://intranet-dlh.hbtn.io/rltoken/S9wQi_nyR6zeHrsHDXAv-g)
* [RL Course by David Silver - Lecture 7: Policy Gradient Methods](https://intranet-dlh.hbtn.io/rltoken/KohJ4gblj8EQrAaGQqXyQQ)
* [Reinforcement Learning 6: Policy Gradients and Actor Critics](https://intranet-dlh.hbtn.io/rltoken/jmxDbQyligJQDpVoJeSzag)
* [Policy Gradient Algorithms](https://intranet-dlh.hbtn.io/rltoken/SzkzQW-ojOMzCPB3Gxcudw)

## Learning Objectives

* What is Policy?
* How to calculate a Policy Gradient?
* What and how to use a Monte-Carlo policy gradient?

## Requirements

- Python 3.9
- numpy 1.25.2
- gymnasium 0.29.1
- pycodestyle 2.11.1

## Files

| File | Description |
|------|-------------|
| [policy_gradient.py](./policy_gradient.py) | `policy(matrix, weight)` computes the policy with a weight of a matrix |
| [policy_gradient.py](./policy_gradient.py) | `policy_gradient(state, weight)` computes the Monte-Carlo policy gradient and returns the action and the gradient |
