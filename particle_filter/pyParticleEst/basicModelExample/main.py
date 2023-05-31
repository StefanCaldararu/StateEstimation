import numpy
import pyparticleest.utils.kalman as kalman
import pyparticleest.interfaces as interfaces
import matplotlib.pyplot as plt
import pyparticleest.simulator as simulator
from model import Model

def generate_dataset(steps, P0, Q, R):
    x = numpy.zeros((steps + 1,))
    y = numpy.zeros((steps,))
    x[0] = 2.0 + 0.0 * numpy.random.normal(0.0, P0)
    for k in range(1, steps + 1):
        x[k] = x[k - 1] + numpy.random.normal(0.0, Q)
        y[k - 1] = x[k] + numpy.random.normal(0.0, R)

    return (x, y)

steps = 50
num = 50
P0 = 1.0
Q = 1.0
R = numpy.asarray(((1.0,),))
numpy.random.seed(1)
(x, y) = generate_dataset(steps, P0, Q, R)
model = Model(P0, Q, R)
sim = simulator.Simulator(model, u=None, y=y)
sim.simulate(num, num, smoother='ancestor')
(vals, _) = sim.get_filtered_estimates()
svals = sim.get_smoothed_estimates()

plt.plot(range(steps + 1), x, 'r-')
plt.plot(range(1, steps + 1), y, 'bx')
plt.plot(range(steps + 1), vals[:, :, 0], 'k.', markersize=0.8)
plt.plot(range(steps + 1), svals[:, :, 0], 'b--')
plt.plot(range(steps + 1), x, 'r-')
plt.xlabel('t')
plt.ylabel('x')
plt.show()