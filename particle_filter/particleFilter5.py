import numpy as np
import math
import matplotlib.pyplot as plt
import random
from scipy.stats import wasserstein_distance

class particleFilter(object):

    def __init__(self, dt):
        self.show_animation = True
        self.dt = dt
        self.pmin = 70
        self.pmax = 100
        #weight is going ot be out of 100. pruned when weight goes below 3.
        self.prune_weight = 0.8
        self.particles = []
        self.particle_weights =[]
        self.hpx = []
        self.hpy = []
        self.particle_distr_dist = []
        self.particle_distr_head = []

        self.dist_distr = [random.gauss(0, 0.8) for _ in range(1000)]
        self.head_distr = [random.gauss(0,0.1) for _ in range(1000)]
        #counts, bins = np.histogram(points, bins = 30)

        for i in range(0,self.pmax):
            self.hpx.append([])
            self.hpy.append([])
            self.particle_distr_dist.append([])
            self.particle_distr_head.append([])
            #self.particles.append(np.zeros((4,1)))
            self.particles.append(np.array([[random.gauss(0,0.8)], [random.gauss(0,0.8)], [0],[0]]))
            self.particle_weights.append(1)
        self.num_particles = self.pmax
        self.old_np = self.num_particles
        self.weights = np.array([0.4, 0.4, 0.2])

        self.red_weight = (100/(self.pmax*1.1))

    
    def update(self, u, obs, htx, hty):
        #first, update the motion model
        for i in range(0, self.num_particles):
            myu = np.array([[u[0,0]+np.random.normal(0,0.02)], [u[1,0]+np.random.normal(0,0.02)]])
            self.particles[i] = self.motion_model(self.particles[i], myu)
            self.hpx[i].append(self.particles[i][0,0])
            self.hpy[i].append(self.particles[i][1,0])
        self.update_dist(obs)
        #then, reassign weights in relation to the observation
        self.assign_weights(obs)
        #then, prune
        self.prune()
        #then, repopulate as necessary,
        #if(self.num_particles<self.pmin):
        #    self.repopulate()
        self.repopulate()
        self.normalizeWeights()
        #then, return final value.
        state = np.zeros((4,1))
        for i in range(0,self.num_particles):
            for j in range(0,4):
                state[j,0] = state[j,0]+self.particles[i][j,0]*self.particle_weights[i]
        for i in range(0,4):
            state[i,0] = state[i,0]/100
        
        if self.show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            for i in range(0,self.num_particles):
                plt.plot(self.hpx[i], self.hpy[i], linewidth = self.particle_weights[i]/10)
            plt.plot(htx, hty, label = 'true position', color = 'k', linewidth = 4.0)
            plt.pause(0.0003)



        return state
    
    def update_dist(self,obs):
        for i in range(0, self.num_particles):
            distance = math.sqrt((self.particles[i][0,0]-obs[0,0])**2+(self.particles[i][1,0]-obs[1,0])**2)
            self.particle_distr_dist[i].append(distance)
            self.particle_distr_head[i].append(self.particles[i][2,0]-obs[2,0])
            if(len(self.particle_distr_dist[i])>10):
                self.particle_distr_dist[i].pop(0)
                self.particle_distr_head[i].pop(0)
            

    def repopulate(self):
        if(self.num_particles == self.old_np):
            self.red_weight = self.red_weight*0.95
        else:
            self.red_weight = (100/(self.pmax*1.1))
            self.old_np = self.num_particles
            
        for i in range(0,self.num_particles):
            while(self.particle_weights[i]>self.red_weight):
                self.particle_weights[i] = self.particle_weights[i]-self.red_weight/2
                self.particle_weights.append(self.red_weight/2)
                self.particle_distr_dist.append(self.particle_distr_dist[i].copy())
                self.particle_distr_head.append(self.particle_distr_head[i].copy())
                

                self.particles.append(np.array([[self.particles[i][0,0]+random.gauss(0,0.1)],[self.particles[i][1,0]+random.gauss(0,0.1)],[self.particles[i][2,0]],[self.particles[i][3,0]]]))
                self.hpx.append(self.hpx[i].copy())
                self.hpy.append(self.hpy[i].copy())
                self.num_particles = self.num_particles+1

    def assign_weights(self, obs):
        for i in range(0,self.num_particles):
            self.particle_weights[i] = 0.9/wasserstein_distance(self.particle_distr_dist[i], self.dist_distr)+0.1/wasserstein_distance(self.particle_distr_head[i], self.head_distr)
        self.normalizeWeights()

    def normalizeWeights(self):
        total = 0
        for i in range(0,self.num_particles):
            total = total+self.particle_weights[i]
        for i in range(0,self.num_particles):
            self.particle_weights[i] = self.particle_weights[i]*100/total

    def computeLiklihood(self, distance, sigma):
        z = abs(distance)/sigma
        return 1-0.5*(1+math.erf(z/math.sqrt(2)))

    def prune(self):
        i = 0
        self.prune_weight = 50/self.pmax

        while(i<self.num_particles and self.num_particles>self.pmin*0.9):
            if(self.particle_weights[i]<self.prune_weight):
                self.particle_weights.pop(i)
                self.particles.pop(i)
                self.hpx.pop(i)
                self.hpy.pop(i)
                i = i-1
                self.num_particles = self.num_particles-1
            i = i+1
        print("NUM_PARTICLES: " + str(self.num_particles))
    

    def motion_model(self, x, u):
        l = 0.5
        tau_0 = 0.09
        omega_0 = 161.185
        r_wheel = 0.08451952624
        gamma = 1/3
        c_0 = 0.039
        c_1 = 1e-4
        i_wheel = 1e-3
        x[0,0] = x[0,0] + math.cos(x[2,0])*self.dt*x[3,0]
        x[1,0] = x[1,0]+math.sin(x[2,0])*self.dt*x[3,0]
        x[2,0] = x[2,0]+self.dt*x[3,0]*math.tan(u[1,0])/l
        f = tau_0*u[0,0]-tau_0*x[3,0]/(omega_0*r_wheel*gamma)
        x[3,0] = x[3,0]+ self.dt*((r_wheel*gamma)/i_wheel)*(f-(x[3,0]*c_1)/(r_wheel*gamma)-c_0)
        return x
    
    