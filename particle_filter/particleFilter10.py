import numpy as np
import math
import matplotlib.pyplot as plt
import random
from scipy.stats import wasserstein_distance

class particleFilter(object):

    def __init__(self, dt):
        random.seed(1)
        self.show_animation = False
        self.dt = dt
        self.pmin = 2
        self.pmax = 100
        #weight is going ot be out of 100. pruned when weight goes below 3.
        self.prune_weight = 0.8
        self.particles = []
        self.particle_weights =[]
        self.hpx = []
        self.hpy = []
        self.particle_distr_dist = []
        self.particle_distr_head = []

        self.dist_distr = [abs(random.gauss(0, 0.8)) for _ in range(100)]
        self.head_distr = [abs(random.gauss(0,0.1)) for _ in range(100)]
        #counts, bins = np.histogram(points, bins = 30)

        for i in range(0,self.pmax):
            self.hpx.append([])
            self.hpy.append([])
            self.particle_distr_dist.append(self.dist_distr.copy())
            self.particle_distr_head.append(self.head_distr.copy())
            #self.particles.append(np.zeros((4,1)))
            self.particles.append(np.array([[random.gauss(0,0.8)], [random.gauss(0,0.8)], [0],[0]]))
            self.particle_weights.append(0.02)
        #sort the distributions
        self.dist_distr = sorted(self.dist_distr)
        self.head_distr = sorted(self.head_distr)
        self.num_particles = self.pmax
        self.old_np = self.num_particles
        self.weights = np.array([0.4, 0.4, 0.2])
        self.timestep = 0

    
    def update(self, u, obs, htx, hty):
        self.timestep = self.timestep+1
        #first, update the motion model
        for i in range(0, self.num_particles):
            myu = np.array([[u[0,0]+np.random.normal(0,0.02)], [u[1,0]+np.random.normal(0,0.02)]])
            self.particles[i] = self.motion_model(self.particles[i], myu)
            self.hpx[i].append(self.particles[i][0,0])
            self.hpy[i].append(self.particles[i][1,0])
        self.update_dist(obs)
        #then, reassign weights in relation to the observation
        self.assign_weights(obs)

        state = np.zeros((4,1))
        for i in range(0,self.num_particles):
            for j in range(0,4):
                state[j,0] = state[j,0]+self.particles[i][j,0]*self.particle_weights[i]
        if(self.timestep%5 == 0):
            self.resample()
        #then, return final value.
        print(state)
        
        if self.show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            for i in range(0,self.num_particles):
                #if(i%1 == 0):
                plt.plot(self.hpx[i], self.hpy[i], linewidth = self.particle_weights[i]*10)
            plt.plot(htx, hty, label = 'true position', color = 'k', linewidth = 4.0)
            plt.xlabel("X position")
            plt.ylabel("Y position")
            plt.legend()
            plt.pause(0.0003)



        return state
    
    def update_dist(self,obs):
        for i in range(0, self.num_particles):
            distance = math.sqrt((self.particles[i][0,0]-obs[0,0])**2+(self.particles[i][1,0]-obs[1,0])**2)
            self.particle_distr_dist[i].append(distance)
            self.particle_distr_head[i].append(self.particles[i][2,0]-obs[2,0])
            self.particle_distr_dist[i].pop(0)
            self.particle_distr_head[i].pop(0)

    def resample(self):
        cumulative_weights = [sum(self.particle_weights[:i+1]) for i in range(self.num_particles)]
        sampled_points = []
        for _ in range(self.num_particles):
            random_num = random.random()  # Generate a random number between 0 and 1

            # Find the interval where the random number falls
            for i, cum_weight in enumerate(cumulative_weights):
                if random_num < cum_weight:
                    sampled_points.append(i)
                    break
        already_seen = []
        new_particles = []
        new_pdd = []
        new_pdh = []
        new_hpx= []
        new_hpy = []
        for j in range(0,self.num_particles):
            i = sampled_points[j]
            new_pdd.append(self.particle_distr_dist[i].copy())
            new_pdh.append(self.particle_distr_head[i].copy())
            new_hpx.append(self.hpx[i].copy())
            new_hpy.append(self.hpy[i].copy())
            if i not in already_seen:
                already_seen.append(i)
                new_particles.append(self.particles[i].copy())
            else:
                new_particles.append(np.array([[self.particles[i][0,0]+random.gauss(0,0.1)],[self.particles[i][1,0]+random.gauss(0,0.1)],[self.particles[i][2,0]],[self.particles[i][3,0]]]))
        self.particles = new_particles
        self.particle_distr_dist = new_pdd
        self.particle_distr_head = new_pdh
        self.hpx = new_hpx
        self.hpy = new_hpy
                



    def assign_weights(self, obs):
        for i in range(0,self.num_particles):
            sorted_dist = sorted(self.particle_distr_dist[i])
            sorted_head = sorted(self.particle_distr_head[i])
            w1 = 0
            w2 = 0
            for j in range(0,100):
                w1 = w1+abs(sorted_dist[j]-self.dist_distr[j])
                w2 = w2+abs(sorted_head[j]-self.head_distr[j])
            self.particle_weights[i] = 0.9/w1+0.1/w2
        self.normalizeWeights()

    def normalizeWeights(self):
        total = 0
        for i in range(0,self.num_particles):
            total = total+self.particle_weights[i]
        for i in range(0,self.num_particles):
            self.particle_weights[i] = self.particle_weights[i]/total

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
    
    