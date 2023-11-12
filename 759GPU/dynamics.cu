#include "dynamics.cuh"
#include <stdio.h>
#include <math.h>

__host__ __device__ void dynamics(float *state, float *control, float dt){
    state[0] += dt*cos(state[2])*state[3];
    state[1] += dt*sin(state[2])*state[3];
    state[2] += dt*state[3]*tan(control[1])/0.5;
    float f = 0.3*control[0]-0.3*state[3]/(30.*0.08451952624*1./3.);
    state[3] += dt*((0.08451952624*(1./3.)) / 0.001) * (f-(state[3]*0.0001) / (0.08451952624*(1./3.))-0.02);
}