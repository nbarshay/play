import test

traj = test.TestModule3.genStill()
sim = test.ArmSim()

sim.run(traj, traj.loop_sec, show=False)

sim2 = test.ArmSim()
