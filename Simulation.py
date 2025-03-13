from Maze import Maze
from Iterative_Method import Iterative_Method
from Monte_Carlo import Monte_Carlo
from Sarsa import Sarsa
from Q_learning import Q_learning
from DQN import DQN
from REINFORCE import REINFORCE
from A2C_off_policy import A2C_off_policy

if __name__ == '__main__':
    H = 5
    W = 5
    blocks = 6
    start = (0, 0)
    maze = Maze(H, W, blocks)

    iterative_method = Iterative_Method(maze)
    maze.show_opt(start, iterative_method.policy)
    maze.show_maze()
    # monte_carlo = Monte_Carlo(puzzle)
    # puzzle.show_opt(start, monte_carlo.policy)

    # sarsa = Sarsa(puzzle)
    # puzzle.show_opt(start, sarsa.policy)

    # q_learning = Q_learning(puzzle)
    # puzzle.show_opt(start, q_learning.policy)

    # dqn = DQN(puzzle)
    # puzzle.show_opt(start, dqn.policy)

    # reinforce = REINFORCE(puzzle)
    # puzzle.show_opt(start, reinforce.policy)

    a2c_off_policy = A2C_off_policy(maze)
    maze.show_opt(start, a2c_off_policy.policy)



