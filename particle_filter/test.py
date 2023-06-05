import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import wasserstein_distance
from randomWalk import randomWalk

def get_distance(prevM, v, m):
        act_a = m-v-prevM
        pred_a = -prevM*4
        pred_a = pred_a*0.15*0.016289174978068626
        # if(abs(pred_a)>0.00875/16):
        #     pred_a = (pred_a/abs(pred_a))*0.00875/16
        #want distribution around pred_a... now just generate new values and return
        new_v = v+act_a#FIXME
        new_m = m
        dist = act_a-pred_a
        return dist, new_m, new_v


prevM = 0
v = 0
rw = randomWalk(0.1)
a_s = []
for i in range(0,100):
    val, mean, vel = rw.step()
    #now we have updated the GPS measurement
    #we want to see what the "prediction" is....
    prevM = prevM
    a_diff, prevM, v = get_distance(prevM, v, val)
    a_s.append(a_diff)
    print("a_diff at iteration " + str(i) +": " + str(a_diff))

dist_distr = [abs(random.gauss(0, 0.016289174978068626)) for _ in range(100)]
print(str(wasserstein_distance(a_s, dist_distr)))
