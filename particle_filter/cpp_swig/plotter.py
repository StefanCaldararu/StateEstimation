import numpy as np
import matplotlib.pyplot as plt
import math
import particleFilter
from randomWalk import randomWalk

def main():
    dt = 0.1
    t = np.arange(0, 100, dt)
    np.random.seed(1)

    trueState = np.zeros((4,1))
    trueState[0,0] = 0
    pf = particleFilter.particleFilter()
    xnoise = randomWalk(dt)
    ynoise = randomWalk(dt)
    htx = []
    hty = []
    hx = []
    hy = []
    hpfx = []
    hpfy = []
    tv = []
    pfv = []
    for x in t:
        time = x
        u = calc_input(time)
        inp = particleFilter.input()
        inp.throttle = u[0,0]
        inp.steering = u[1,0]

        trueState = motion_model(trueState, u, dt)
        # print("trueState: ")
        # print(trueState)
        obs = observation(trueState, xnoise, ynoise)
        o = particleFilter.observation()
        o.x = obs[0,0]
        o.y = obs[1,0]
        o.theta = obs[2,0]
        htx.append(trueState[0,0])
        hty.append(trueState[1,0])
        hx.append(obs[0,0])
        hy.append(obs[1,0])
        pfstate = pf.step(inp,o)
        pfv.append(pfstate.v)
        tv.append(trueState[3,0])
        hpfx.append(pfstate.x)
        hpfy.append(pfstate.y)
        print("pfstate: ",pfstate.x, " ", pfstate.y)
        print("truestate: ", trueState[0,0], " ", trueState[1,0])

    # fig = plt.figure()
    # #fig.suptitle('Particle Filter', fontsize=20)
    # plt.plot(hx, hy, label='measurements', color='g', linewidth = 0.3)
    # plt.plot(htx, hty, label = 'true position', color='r', linewidth = 1.5)
    # plt.plot(hpfx, hpfy, label = 'particle filter', color='b', linewidth = 0.3)
    # plt.xlabel('Position x (m)', fontsize=20)
    # plt.ylabel('Position y (m)', fontsize=20)
    # plt.legend()
    # plt.show()
    


    # fig,a = plt.subplots(2)

    # #a[0].plot(xs, ys, label = 'measurement', color = 'r', linewidth = 1.5)
    # a[0].plot(hx, hy, label = 'measurement', color = 'g', linewidth = 0.1)
    # a[0].plot(htx, hty, label = 'true position', color = 'r', linewidth = 1.5)
    # a[0].plot(hpfx, hpfy,  label = 'Particle Filter', color = 'b', linewidth = 0.3)
    # a[0].set_ylabel('Y Position')
    # a[0].set_xlabel('X Position')
    # a[0].legend()
    # a[1].plot(t, tv, label = 'true velocity', color = 'b', linewidth = 1.5)
    # a[1].plot(t, pfv, label = 'PF velocity', color = 'g', linewidth = 1.5)
    # a[1].legend()
    # a[1].set_xlabel('time')
    # a[1].set_ylabel('velocity')
    # plt.show()

def calc_input(t):
    throttle = 0.5
    steering = 0.1
    u = np.array([[throttle],[steering]])
    return u

def observation(x, xnoise, ynoise):
    obs = np.zeros((3,1))
    obs[0,0] = x[0,0]+xnoise.step()#np.random.normal(0,0.8)
    obs[1,0] = x[1,0]+ynoise.step()#np.random.normal(0,0.8)
    obs[2,0] = x[2,0]+np.random.normal(0,0.1)
    return obs


def motion_model(x, u, dt):
    l = 0.5
    tau_0 = 0.09
    omega_0 = 161.185
    r_wheel = 0.08451952624
    gamma = 1/3
    c_0 = 0.039
    c_1 = 1e-4
    i_wheel = 1e-3
    x[0,0] = x[0,0] + math.cos(x[2,0])*dt*x[3,0]
    x[1,0] = x[1,0]+math.sin(x[2,0])*dt*x[3,0]
    x[2,0] = x[2,0]+dt*x[3,0]*math.tan(u[1,0])/l
    f = tau_0*u[0,0]-tau_0*x[3,0]/(omega_0*r_wheel*gamma)
    x[3,0] = x[3,0]+ dt*((r_wheel*gamma)/i_wheel)*(f-(x[3,0]*c_1)/(r_wheel*gamma)-c_0)
    return x


if __name__ == '__main__':
    main()