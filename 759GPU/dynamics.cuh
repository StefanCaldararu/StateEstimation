#ifndef DYNAMICS_H
#define DYNAMICS_H

#include <cstddef>

__host__ __device__ void dynamics(float *state, float *control, float dt);

#endif