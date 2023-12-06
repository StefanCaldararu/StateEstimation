#ifndef PF_H
#define PF_H

#include <cstddef>
#include <random>
//The overarching update function for the particle filter. 
__host__ void update_CPU(float ** particles, float** pd_dist, float** pd_head, float* d_dist, float* d_head, float* weights, size_t N, float* control, float* obs, int & timestep, float* prediction, std::mt19937 gen);
__host__ void update_GPU(float ** particles, float** pd_dist, float** pd_head, float* d_dist, float* d_head, float* weights, size_t N, float* control, float* obs, int & timestep, float* prediction, std::mt19937 gen);
__global__ void GPU_kernel(float ** particles, float** pd_dist, float** pd_head, float* d_dist, float* d_head, float* weights, size_t N, float* control, float* obs, int & timestep, float* prediction, std::mt19937 gen, std::normal_distribution<float> dist);

//When we update each distribution, we append to the end. It is worth noting that a circular buffer is used for each distribution. 
__host__ void update_dist_CPU(float ** particles, float** pd_dist, float** pd_head, size_t N, int dist_len, int current_val, float* obs);
__device__ void update_dist_GPU(float ** particles, float** pd_dist, float** pd_head, size_t N, int dist_len, int current_val, float* obs, int id);
__host__ __device__ float angle_diff(float head, float ref);

//The resample is where the majority of the memory movement occurs, which is probably the most time-consuming task for this algorithm. This is mostly linear, and only data movement, so should probably only be done on the cpu (but a gpu implementation that does the data movement is also here...)
__host__ void resample_CPU(float ** particles, float** pd_dist, float** pd_head, float* weights, size_t N);
__host__ void resample_GPU(float ** particles, float** pd_dist, float** pd_head, float* weights, size_t N);
__global__ void resample_GPU_kernel(float ** particles, float** pd_dist, float** pd_head, float* cum_weights, size_t N);

//Assign the weights of the N particles. This is done by sorting the particle distance and heading error distributions, and then summing the difference to the defined sample distribution. This is a reduction of the EMD. This can be done either on the CPU or GPU. For the GPU, we only do parallelization at the particle level here. This is then expanded to parallelization for the sort and reduce steps in helper kernels.
__host__ void assign_weights_CPU(float** pd_dist, float** pd_head, float* d_dist, float* d_head, float* weights, size_t N, int dist_len);
__global__ void assign_weights_GPU(float** pd_dist, float** pd_head, float* d_dist, float* d_head, float* weights, size_t N, int dist_len);

//Normalize the weights of the N particles. This has a host implementation where everything is done linearly, as well as a GPU implementation where the weights are reduced in parallel, and then the normalization factor is also applied in parallel.
__host__ void normalize_weights_CPU(float *weights, size_t N);
__global__ void normalize_weights_GPU(float *weights, size_t N); 

#endif