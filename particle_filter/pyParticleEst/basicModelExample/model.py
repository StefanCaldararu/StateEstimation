import numpy
import pyparticleest.utils.kalman as kalman
import pyparticleest.interfaces as interfaces
import matplotlib.pyplot as plt
import pyparticleest.simulator as simulator


class Model(interfaces.ParticleFiltering):
    """ x_{k+1} = x_k + v_k, v_k ~ N(0,Q)
        y_k = x_k + e_k, e_k ~ N(0,R),
        x(0) ~ N(0,P0) """

    def __init__(self, P0, Q, R):
        self.P0 = numpy.copy(P0)
        self.Q = numpy.copy(Q)
        self.R = numpy.copy(R)

    def create_initial_estimate(self, N):
        return numpy.random.normal(0.0, self.P0, (N,)).reshape((-1, 1))

    def sample_process_noise(self, particles, u, t):
        """ Return process noise for input u """
        N = len(particles)
        return numpy.random.normal(0.0, self.Q, (N,)).reshape((-1, 1))

    def update(self, particles, u, t, noise):
        """ Update estimate using 'data' as input """
        particles += noise

    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """
        logyprob = numpy.empty(len(particles), dtype=float)
        for k in range(len(particles)):
            logyprob[k] = kalman.lognormpdf(particles[k].reshape(-1, 1) - y, self.R)
        return logyprob

    def logp_xnext_full(self, part, past_trajs, pind,
                        future_trajs, find, ut, yt, tt, cur_ind):

        diff = future_trajs[0].pa.part[find] - part

        logpxnext = numpy.empty(len(diff), dtype=float)
        for k in range(len(logpxnext)):
            logpxnext[k] = kalman.lognormpdf(diff[k].reshape(-1, 1), numpy.asarray(self.Q).reshape(1, 1))
        return logpxnext