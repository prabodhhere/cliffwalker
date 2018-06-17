from lib.envs.cliff_walking import CliffWalkingEnv

shape = (4, 12)
start = (3, 0)
end = [(3, 11)]
cliff = tuple((3, i+1) for i in range(11))

env = CliffWalkingEnv(shape, start, end, cliff)
env.render()
