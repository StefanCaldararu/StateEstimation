#include "particle_filter.cuh"
#include "dynamics.cuh"
#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdio.h>
#include <random>
//include thrust sort for the GPU implementation
#include <thrust/sort.h>
//include thrust reduce
#include <thrust/reduce.h>
//include cuda random
#include <curand.h>
//The overarching update function for the particle filter. 
__host__ void update_CPU(float ** particles, float** pd_dist, float** pd_head, float* d_dist, float* d_head, float* weights, size_t N, float* control, float* obs, int & timestep, float* prediction, std::mt19937 gen){
    //create a cuda random number generator

    std::normal_distribution<float> dist(0, 0.02);
    //first, updaet each particle via the motion model
    for(size_t i = 0;i<N;i++){
        float* mycontrol = (float*)malloc(2*sizeof(float));
        mycontrol[0] = control[0]+dist(gen);
        mycontrol[1] = control[1]+dist(gen);
        //TODO: modify the control a little bit to create particle diversity
        dynamics(particles[i], mycontrol, 0.1);
        free(mycontrol);
    }
    //update the distance and heading error distributions for each particle
    update_dist_CPU(particles, pd_dist, pd_head, N, 100, timestep%100, obs);
    //assign the weights to each particle
    assign_weights_CPU(pd_dist, pd_head, d_dist, d_head, weights, N, 100);
    for(size_t i = 0;i<4;i++){
        prediction[i] = 0;
    }
    for(size_t i = 0;i<N;i++){
        for(size_t j = 0;j<4;j++){
            prediction[j] += particles[i][j]*weights[i];
        }
    }
    //resample if necessary
    if(timestep%5 == 0)
        resample_CPU(particles, pd_dist, pd_head, weights, N);

}
__host__ void update_GPU(float ** particles, float** pd_dist, float** pd_head, float* d_dist, float* d_head, float* weights, size_t N, float* control, float* obs, int & timestep, float* prediction, std::mt19937 gen){
    //TODO:
}

__global__ void GPU_kernel(float ** particles, float** pd_dist, float** pd_head, float* d_dist, float* d_head, float* weights, size_t N, float* control, float* obs, int & timestep, float* prediction, float** ran_control, float* ran){
    //TODO: a lot of this stuff should probably be in shared memory...
    //Get the thread id
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    //run the dynamics for this particle
    //allocate control with cuda memory
    float* mycontrol;
    cudaMalloc(&mycontrol, 2*sizeof(float));
    mycontrol[0] = control[0]+ran_control[idx][0];
    mycontrol[1] = control[1]+ran_control[idx][1];
    dynamics(particles[idx], mycontrol, 0.1);
    cudaFree(mycontrol);
    //update the distance and heading error distributions for this particle
    update_dist_GPU(particles, pd_dist, pd_head, N, 100, timestep%100, obs, idx);
    //assign the weights to this particle
    assign_weights_GPU(pd_dist, pd_head, d_dist, d_head, weights, N, 100, idx);
    //compute the prediction, but only one thread. This is done linearly, although it could probably be done as a reduction...
    if(idx == 0){
        for(size_t i = 0;i<4;i++){
            prediction[i] = 0;
        }
        for(size_t i = 0;i<N;i++){
            for(size_t j = 0;j<4;j++){
                prediction[j] += particles[i][j]*weights[i];
            }
        }
    }
    //sync the threads
    __syncthreads();
    //resample if necessary
    if(timestep%5 == 0)
        resample_GPU(particles, pd_dist, pd_head, weights, N, idx, ran);
    //sync the threads
    __syncthreads();
}
//When we update each distribution, we append to the end. It is worth noting that a circular buffer is used for each distribution. 
__host__ void update_dist_CPU(float ** particles, float** pd_dist, float** pd_head, size_t N, int dist_len, int current_val, float* obs){
    for(size_t i = 0;i<N;i++){
    //calculate the distance and heading error
        float dist = sqrt(pow(particles[i][0]-obs[0],2)+pow(particles[i][1]-obs[1],2));
        float head = angle_diff(particles[i][3], obs[2]);
        //we want to put this value in at the current_val index
        pd_dist[i][current_val] = dist;
        pd_head[i][current_val] = head;
    }
}
__device__ void update_dist_GPU(float ** particles, float** pd_dist, float** pd_head, size_t N, int dist_len, int current_val, float* obs, int id){
    //calculate the distance and heading error
    float dist = sqrt(pow(particles[id][0]-obs[0],2)+pow(particles[id][1]-obs[1],2));
    float head = angle_diff(particles[id][3], obs[2]);
    //we want to put this value in at the current_val index
    pd_dist[id][current_val] = dist;
    pd_head[id][current_val] = head;
}

//helper function for computing the difference between two angles, modular subtraction
__host__ __device__ float angle_diff(float head, float ref){
    //TODO: FIGURE THIS OUT....
    return head-ref;
}

//The resample is where the majority of the memory movement occurs, which is probably the most time-consuming task for this algorithm. This is mostly linear, and only data movement, so should probably only be done on the cpu (but a gpu implementation that does the data movement is also here...)
__host__ void resample_CPU(float ** particles, float** pd_dist, float** pd_head, float* weights, size_t N){
    //allocate memory for the cumulative weights
    float* cum_weights = (float*)malloc(N*sizeof(float));
    //calculate the cumulative weights
    cum_weights[0] = weights[0];
    for(size_t i = 1; i < N; i++){
        cum_weights[i] = cum_weights[i-1] + weights[i];
    }
    //copy the particles to a old array, so that we can modify the old one.
    float** particles_old = (float**)malloc(N*sizeof(float*));
    for(size_t i = 0; i < N; i++){
        particles_old[i] = (float*)malloc(4*sizeof(float));
        memcpy(particles_old[i], particles[i], 4*sizeof(float));
    }
    //do the same for the particles distributions
    float** pd_dist_old = (float**)malloc(N*sizeof(float*));
    float** pd_head_old = (float**)malloc(N*sizeof(float*));
    for(size_t i = 0; i < N; i++){
        pd_dist_old[i] = (float*)malloc(100*sizeof(float));
        pd_head_old[i] = (float*)malloc(100*sizeof(float));
        memcpy(pd_dist_old[i], pd_dist[i], 100*sizeof(float));
        memcpy(pd_head_old[i], pd_head[i], 100*sizeof(float));
    }
    //resample the particles
    for(size_t i = 0; i < N; i++){
        //generate a random number between 0 and 1
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        //find the particle that corresponds to this random number
        for(size_t j = 0; j < N; j++){
            if(r < cum_weights[j]){
                //copy the particle and its distributions to the old arrays
                memcpy(particles[i], particles_old[j], 4*sizeof(float));
                memcpy(pd_dist[i], pd_dist_old[j], 100*sizeof(float));
                memcpy(pd_head[i], pd_head_old[j], 100*sizeof(float));
                break;
            }
        }
    }
    //free the memory used
    free(cum_weights);
    for(size_t i = 0; i < N; i++){
        free(particles_old[i]);
        free(pd_dist_old[i]);
        free(pd_head_old[i]);
    }
    free(particles_old);
}

__device__ void resample_GPU(float ** particles, float** pd_dist, float** pd_head, float* weights, size_t N, int id, float* ran){
    float particle_old[4], pd_dist_old[100], pd_head_old[100];

    //get our random number between 0 and 1
    float r = ran[id];
    //find the particle that corresponds to this random number
    float current_cum_weight = weights[0];
    int index = 0;
    while(r > current_cum_weight){
        index++;
        current_cum_weight += weights[index];
    }
    //allocate memory for the old particle and its distributions
    for(int i = 0;i<4;i++)
        particle_old[i] = particles[id][i];
    for(int i = 0;i<100;i++){
        pd_dist_old[i] = pd_dist[id][i];
        pd_head_old[i] = pd_head[id][i];
    }

    //sync the threads, so that we don't have any issues with the memory
    __syncthreads();
    //copy the old particle and its distributions to the new memory
    for(int i = 0;i<4;i++)
        particles[id][i] = particle_old[i];
    for(int i = 0;i<100;i++){
        pd_dist[id][i] = pd_dist_old[i];
        pd_head[id][i] = pd_head_old[i];
    }
    //sync threads
    __syncthreads();
}

//Assign the weights of the N particles. This is done by sorting the particle distance and heading error distributions, and then summing the difference to the defined sample distribution. This is a reduction of the EMD. This can be done either on the CPU or GPU. For the GPU, we only do parallelization at the particle level here. This is then expanded to parallelization for the sort and reduce steps in helper kernels.
__host__ void assign_weights_CPU(float** pd_dist, float** pd_head, float* d_dist, float* d_head, float* weights, size_t N, int dist_len){
    for(size_t i = 0;i<N;i++){
        //allocate memory for the particle's distance and heading error distributions
        float* pd_dist_i = (float*)malloc(dist_len*sizeof(float));
        float* pd_head_i = (float*)malloc(dist_len*sizeof(float));
        //copy the particle's distance and heading error distributions to the new memory, which will get sorted
        memcpy(pd_dist_i, pd_dist[i], dist_len*sizeof(float));
        memcpy(pd_head_i, pd_head[i], dist_len*sizeof(float));
        //sort the particle pd's (but only up till the dist_len'th element)
        std::sort(pd_dist_i, pd_dist_i+dist_len);
        std::sort(pd_head_i, pd_head_i+dist_len);
        float w1 = 0;
        float w2 = 0;
        //sum the difference between the sample distribution and the particle distribution
        for(int j = 0; j < dist_len; j++){
            w1 += abs(pd_dist_i[j] - d_dist[j]);
            w2 += abs(pd_head_i[j] - d_head[j]);
        }
        if(w1 == 0)
            w1 = 0.0001;
        if(w2 == 0)
            w2 = 0.0001;
        //sum the two weights (weighted though), and assign to the particle
        weights[i] = 0.9/w1+0.1/w2;
        //free the memory used
        free(pd_dist_i);
        free(pd_head_i);
    }
    //normalize the weights
    normalize_weights_CPU(weights, N);
}
__device__ void assign_weights_GPU(float** pd_dist, float** pd_head, float* d_dist, float* d_head, float* weights, size_t N, int dist_len, int id){
    //allocate memory for the particle's distance and heading error distributions
    float pd_dist_i[100], pd_head_i[100];
    //copy the particle's distance and heading error distributions to the new memory, which will get sorted
    for(int i = 0;i<dist_len;i++){
        pd_dist_i[i] = pd_dist[id][i];
        pd_head_i[i] = pd_head[id][i];
    }
    //sort the particle pd's (but only up till the dist_len'th element)
    thrust::sort(thrust::device, pd_dist_i, pd_dist_i+100);
    thrust::sort(thrust::device, pd_head_i, pd_head_i+100);
    //use thrust reduce to sum the difference between the sample distribution and the particle distribution for the distance and heading error
    float w1 = thrust::reduce(thrust::device, pd_dist_i, pd_dist_i+dist_len, 0.0f, thrust::plus<float>());
    float w2 = thrust::reduce(thrust::device, pd_head_i, pd_head_i+dist_len, 0.0f, thrust::plus<float>());
    if(w1 == 0)
        w1 = 0.0001;
    if(w2 == 0)
        w2 = 0.0001;
    //sum the two weights (weighted though), and assign to the particle
    weights[id] = 0.9/w1+0.1/w2;
    //normalize the weights
    normalize_weights_GPU(weights, N, id);
}

//Normalize the weights of the N particles. This has a host implementation where everything is done linearly, as well as a GPU implementation where the weights are reduced in parallel, and then the normalization factor is also applied in parallel.
__host__ void normalize_weights_CPU(float *weights, size_t N){
    float total = 0;
    for(size_t i = 0; i < N; i++){
        total += weights[i];
    }
    for(size_t i = 0; i < N; i++){
        weights[i] /= total;
    }
}
__device__ void normalize_weights_GPU(float *weights, size_t N, int id){
    //first sync the threads here
    __syncthreads();
    //reduce the weights to total, each thread does this because we aren't using shared memory... probably better to only do this once and write to shared memory, but this is easier for now.
    float total = thrust::reduce(thrust::device, weights, weights+N, 0.0f, thrust::plus<float>());
    //divide each weight by the total
    weights[id] /= total;
    //sync the threads again
    __syncthreads();
} 