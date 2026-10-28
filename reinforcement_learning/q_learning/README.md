# Q-learning

Reinforcement learning project using gymnasium (FrozenLake-v1) and numpy.

## Resources

Read or watch:

* [MIT 6.S191: Reinforcement Learning](https://intranet-dlh.hbtn.io/rltoken/saQI8BvlKV60MENEf_43MA)
* [An introduction to Reinforcement Learning](https://intranet-dlh.hbtn.io/rltoken/gOyUyOcWBr5QnuXhgJuVlw)
* [An Introduction to Q-Learning: A Tutorial For Beginners](https://intranet-dlh.hbtn.io/rltoken/c9NpSay4kyBNipR9cwPouw)
* [Q-Learning](https://intranet-dlh.hbtn.io/rltoken/EYIHlSinClcDSXQ9TnuXLg)
* [Markov Decision Processes (MDPs) - Structuring a Reinforcement Learning Problem](https://intranet-dlh.hbtn.io/rltoken/YUp-gcE1R9mC1cjjfvFh2w)
* [Expected Return - What Drives a Reinforcement Learning Agent in an MDP](https://intranet-dlh.hbtn.io/rltoken/BKs-9m9ep0sTz8V-EjoGrQ)
* [Policies and Value Functions - Good Actions for a Reinforcement Learning Agent](https://intranet-dlh.hbtn.io/rltoken/i3LsCe2sGI5kM1Qzr0d6Rg)
* [What do Reinforcement Learning Algorithms Learn - Optimal Policies](https://intranet-dlh.hbtn.io/rltoken/3nofOjrYYD1ghpMOEkG02g)
* [Q-Learning Explained - A Reinforcement Learning Technique](https://intranet-dlh.hbtn.io/rltoken/v3Bnyaow4gRx6K0os1GsSA)
* [Exploration vs. Exploitation - Learning the Optimal Reinforcement Learning Policy](https://intranet-dlh.hbtn.io/rltoken/igctuWgpbqhykidsjEwZxQ)
* [Q-Learning Tutorial 1: Train Gymnasium FrozenLake-v1 with Python Reinforcement Learning](https://intranet-dlh.hbtn.io/rltoken/U-tHCRc7pwL9rQNcfcJNag)
* [Q-Learning: Implementation](https://intranet-dlh.hbtn.io/rltoken/Z42tZf3ntSP9nXsoX7PfXg)
* [Markov Decision Processes](https://intranet-dlh.hbtn.io/rltoken/IRn9ww1sX2MQSCsqwgUhSA)

Definitions to skim:

* [Reinforcement Learning](https://intranet-dlh.hbtn.io/rltoken/5UL4o3rRKzaKgdn_KZbk4g)
* [Markov Decision Process](https://intranet-dlh.hbtn.io/rltoken/qdx8Ngc-KwHS4tEqYFhgcw)
* [Q-learning](https://intranet-dlh.hbtn.io/rltoken/6PTuLv2ydxZMNSoHo2H__A)

References:

* [Gymnasium](https://intranet-dlh.hbtn.io/rltoken/SsIl9ONmw0QdnKM3JCrJEw)
* [Gymnasium: Frozen Lake env](https://intranet-dlh.hbtn.io/rltoken/JV_5-FJjBJWz86dKtlkeSg)
* [Gymnasium: Q-Learning with Frozenlake env](https://intranet-dlh.hbtn.io/rltoken/RoH_yN8UOWnfH8NwrYQ_cw)
* [Gymnasium: Env](https://intranet-dlh.hbtn.io/rltoken/lI4cmpbu0UYJGH7G4moxew)

## Requirements

- Python 3.9
- numpy 1.25.2
- gymnasium 0.29.1
- pycodestyle 2.11.1

## Files

| File | Description |
|------|-------------|
| [0-load_env.py](./0-load_env.py) | `load_frozen_lake(desc=None, map_name=None, is_slippery=False)` loads the pre-made FrozenLake environment with `render_mode="ansi"` |
| [1-q_init.py](./1-q_init.py) | `q_init(env)` initializes the Q-table as a numpy.ndarray of zeros |
| [2-epsilon_greedy.py](./2-epsilon_greedy.py) | `epsilon_greedy(Q, state, epsilon)` uses epsilon-greedy to determine the next action |
| [3-q_learning.py](./3-q_learning.py) | `train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05)` performs Q-learning and returns the Q-table and rewards per episode |
| [4-play.py](./4-play.py) | `play(env, Q, max_steps=100)` has the trained agent play an episode and returns the total rewards and the rendered board states |
