#include "dynamics.cuh"
#include <cuda.h>
#include <stdio.h>

int main(int argc, char** argv){
    float total_time = 10.0;
    float dt = 0.1;
    float time = 0.0;
    float * state = (float*) malloc(4*sizeof(float));
    state[0] = 0.0;
    state[1] = 0.0;
    state[2] = 0.0;
    state[3] = 0.0;
    float * control = (float*) malloc(2*sizeof(float));
    control[0] = 0.9;
    control[1] = 0.2;
    while(time < total_time){
        time += dt;
        dynamics(state, control, dt);
        printf("x: %f, y: %f, v: %f, theta: %f\n", state[0], state[1], state[2], state[3]);
    }
    return 0;
}