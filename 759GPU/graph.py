import matplotlib.pyplot as plt
import numpy as np
import csv

state = []
pred = []
obs = []
# Read in data from ./out.csv,
with open('./out.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
    #remove the first row
    data.pop(0)
    #convert to numpy array
    for d in data:
        #if d[0] is STATE, push to state array without the first element (d[0])
        if d[0] == 'STATE':
            state.append(d[1:])
        #if d[0] is PRED, push to pred array without the first element (d[0])
        elif d[0] == 'PRED':
            pred.append(d[1:])
        #if d[0] is OBS, push to obs array without the first element (d[0])
        elif d[0] == 'OBS':
            obs.append(d[1:])
        else:
            print('error')
#convert to numpy array
state = np.array(state)
pred = np.array(pred)
obs = np.array(obs)
#manually convert all elements to floats


x = []
y = []
xp = []
yp = []
xo = []
yo = []
for i in state:
    x.append(float(i[0]))
    y.append(float(i[1]))
for i in pred:
    xp.append(float(i[0]))
    yp.append(float(i[1]))
for i in obs:
    xo.append(float(i[0]))
    yo.append(float(i[1]))
#plot the graph for the state[i][0] and state[i][1] for all i
plt.plot(x,y, label='state')
plt.plot(xp,yp, label='pred')
plt.plot(xo,yo, label='obs')
plt.legend()
#save the graph as a png file
plt.savefig('graph.png')



