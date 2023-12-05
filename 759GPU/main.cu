#include "particle_filter.cuh"
#include "dynamics.cuh"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <iostream>

int main(int argc, char** argv){
    //The random generator
    std::random_device rd;
    std::mt19937 gen(rd());
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
    // Assign memory for the PF
    int num_particles = 100;
    float ** particles = (float**) malloc(num_particles*sizeof(float*));
    for(int i = 0;i<num_particles;i++){
        particles[i] = (float*) malloc(4*sizeof(float));
        particles[i][0] = 0.0;
        particles[i][1] = 0.0;
        particles[i][2] = 0.0;
        particles[i][3] = 0.0;
    }
    float ** pd_dist = (float**) malloc(num_particles*sizeof(float *));
    for(int i = 0;i<num_particles;i++){
        pd_dist[i] = (float*) malloc(100*sizeof(float));
    }
    float ** pd_head = (float**) malloc(num_particles*sizeof(float *));
    for(int i = 0;i<num_particles;i++){
        pd_head[i] = (float*) malloc(100*sizeof(float));
    }
    float * d_dist = (float*) malloc(100*sizeof(float));
    float * d_head = (float*) malloc(100*sizeof(float));
    std::normal_distribution<float> dist(0.0, 1.0);
    std::normal_distribution<float> head(0.0, 0.1);
    for(int i = 0;i<100;i++){
        //append a random number with mean 0 and variance 1 to d_dist and a random number with mean 0 and variance 0.1 to d_head
        d_dist[i] = dist(gen);
        d_head[i] = head(gen);
    }
    float * weights = (float*) malloc(num_particles*sizeof(float));
    for(int i = 0;i<num_particles;i++)
        weights[i] = 1.0/num_particles;

    float * obs = (float*) malloc(3*sizeof(float));
    obs[0] = 0.0;
    obs[1] = 0.0;
    obs[2] = 0.0;
    int timestep = 0;
    float * prediction = (float*) malloc(4*sizeof(float));


    while(time < total_time){
        time += dt;
        dynamics(state, control, dt);
        //obs is the x, y, theta observation
        obs[0] = state[0] + dist(gen);
        obs[1] = state[1] + dist(gen);
        obs[2] = state[3] + head(gen);
        //prop the pf
        update_CPU(particles, pd_dist, pd_head, d_dist, d_head, weights, num_particles, control, obs, timestep, prediction);
        // printf("x: %f, y: %f, v: %f, theta: %f\n", state[0], state[1], state[2], state[3]);
    }
    printf("PARTICE FILTER RAN!");
    free(state);
    free(control);
    free(obs);
    free(prediction);
    for(int i = 0;i<num_particles;i++){
        free(particles[i]);
        free(pd_dist[i]);
        free(pd_head[i]);
    }
    free(particles);
    free(pd_dist);
    free(pd_head);
    free(d_dist);
    free(d_head);
    free(weights);

    return 0;
}