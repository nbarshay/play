from test import *


if __name__ == '__main__':
    traj = BasicTrajectory(seed=0)
    critic1 = TestCritic()
    optimize(traj, critic1, maxiter=100, verbose=True)
