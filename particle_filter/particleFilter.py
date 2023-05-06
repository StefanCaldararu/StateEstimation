import numpy as np
import math
import matplotlib.pyplot as plt

class particleFilter(object):

    def __init__(self, dt):
        self.show_animation = True
        self.dt = dt
        self.pmin = 5
        self.pmax = 10
        #weight is going ot be out of 100. pruned when weight goes below 3.
        self.prune_weight = 3
        self.particles = []
        self.particle_weights =[]
        self.hpx = []
        self.hpy = []
        for i in range(0,self.pmax):
            self.hpx.append([])
            self.hpy.append([])
            self.particles.append(np.zeros((4,1)))
            self.particle_weights.append(10)
        self.num_particles = 10
        self.weights = np.array([0.4, 0.4, 0.2])

    
    def update(self, u, obs, htx, hty):
        #first, update the motion model
        for i in range(0, self.num_particles):
            myu = np.array([[u[0,0]+np.random.normal(0,0.05)], [u[1,0]+np.random.normal(0,0.05)]])
            self.particles[i] = self.motion_model(self.particles[i], myu)
            self.hpx[i].append(self.particles[i][0,0])
            self.hpy[i].append(self.particles[i][1,0])
        #then, reassign weights in relation to the observation
        self.assign_weights(obs)
        #then, prune
        self.prune()
        #then, repopulate as necessary,
        if(self.num_particles<self.pmin):
            self.repopulate()
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
            plt.pause(0.001)



        return state
    

    def repopulate(self):
        for i in range(0,self.num_particles):
            while(self.particle_weights[i]>15):
                self.particle_weights[i] = self.particle_weights[i]-10
                self.particle_weights.append(10)

                self.particles.append(np.array([[self.particles[i][0,0]],[self.particles[i][1,0]],[self.particles[i][2,0]],[self.particles[i][3,0]]]))
                self.hpx.append(self.hpx[i].copy())
                self.hpy.append(self.hpy[i].copy())
                self.num_particles = self.num_particles+1

    def assign_weights(self, obs):
        total_error = np.zeros((3,1))
        for i in range(0,self.num_particles):
            for j in range(0,3):
                total_error[j,0] = total_error[j,0]+abs(obs[j,0]-self.particles[i][j,0])
        #now we can normalize... just have to use weights to get particle_weights. Want particle weights to add to 100.
        for i in range(0,self.num_particles):
            self.particle_weights[i] = 100*((abs(obs[0,0]-self.particles[i][0,0])/total_error[0,0])*self.weights[0]+ (abs(obs[1,0]-self.particles[i][1,0])/total_error[1,0])*self.weights[1]+ (abs(obs[2,0]-self.particles[i][2,0])/total_error[2,0])*self.weights[2])
        

    def prune(self):
        i = 0
        while(i<self.num_particles):
            if(self.particle_weights[i]<self.prune_weight):
                self.particle_weights.pop(i)
                self.particles.pop(i)
                self.hpx.pop(i)
                self.hpy.pop(i)
                i = i-1
                self.num_particles = self.num_particles-1
            i = i+1
    

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
    
    