# Simple Implementation of REPAIR

- Implementation of Interactive Reinforcement Learning with Inaccurate Feedback [paper](https://sim.ece.utexas.edu/static/papers/REPaIR-ICRA.pdf)

- Applied to a custom Breakout game environment

# Rewards

- 1 for each brick broken, -10 for dying, 0 otherwise

# Feedback

- 1 if ball is aligned with the paddle (ie. if ball were to fall down vertically, the paddle would catch it), -1 otherwise

# Files

- QLearning_Breakout.py -> Simple Q learning algorithm no feedback.

- QLearning_with_feedback_Breakout.py -> Simple Q learning algorithm with extra feedback.

- QLearning_with_incorrect_feedback_Breakout.py -> Simple Q learning algorithm with extra incorrect feedback.

- REPAIR.py -> Simple Q learning algorithm with extra incorrect feedback filtered by REPAIR.

- UseQTable.py -> Watch a policy play the game.
