#include <stdio.h>
#include <stdlib.h>
#include "nbody.h"

/*
  1. *i - index of particle to be compute_forced
  2. *nparticles - # of particles
  3. d_p points to the whole particles 
*/

__device__ double atomicAddDouble(double* address, double val)
{
  unsigned long long int* address_as_ull =
  (unsigned long long int*) address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
    __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case
    // of NaN (since NaN != NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}

__global__ void __compute_force__ (int * i, int * nparticles, particle_t * d_p) 
{
  
  particle_t * computed_p = &d_p[*i];  
  int j;

  // __syncthreads();

  for(j = blockIdx.x * blockDim.x + threadIdx.x;
      j < *nparticles;
      j += blockDim.x*gridDim.x) 
  {
    particle_t * p = &d_p[j];
    // change the cpu version to cuda version
    double x_sep, y_sep, dist_sq, grav_base;

    x_sep = p->x_pos - computed_p->x_pos;
    y_sep = p->y_pos - computed_p->y_pos;
    dist_sq = MAX((x_sep*x_sep) + (y_sep*y_sep), 0.01);

    /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
    grav_base = GRAV_CONSTANT * (computed_p->mass) * (p->mass)/dist_sq;

    // computed_p->x_force += grav_base*x_sep;
    // computed_p->y_force += grav_base*y_sep;

    // using atomicAdd
    atomicAddDouble(&(computed_p->x_force), grav_base*x_sep);
    atomicAddDouble(&(computed_p->y_force), grav_base*y_sep);

  }

}

extern "C" void cuda_compute_force(int i, int nparticles, particle_t * p)
{

  particle_t * d_p;
  int * d_i;
  int * d_nparticles;

  // allocate space for device copies
  cudaMalloc((void **)&d_p, nparticles * sizeof(particle_t));
  cudaMalloc((void **)&d_i, sizeof(int));
  cudaMalloc((void **)&d_nparticles, sizeof(int)); 

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    printf(cudaGetErrorString(cudaStatus));
  }
  
  // copy inputs to device
  cudaMemcpy(d_p, p, nparticles * sizeof(particle_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_i, &i, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nparticles, &nparticles, sizeof(int), cudaMemcpyHostToDevice);

  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    printf(cudaGetErrorString(cudaStatus));
  }
  
  // TODO: check the return .
  int thr_per_blk = 1024; // maximum
  int blk_in_grid = (int) ceil( float(nparticles) / thr_per_blk );

  __compute_force__<<<blk_in_grid,thr_per_blk>>>(d_i, d_nparticles, d_p);
  // __compute_force__<<<1,2>>>(d_i, d_nparticles, d_p);
  cudaDeviceSynchronize();

  // copy result back to host
  cudaMemcpy(&p[i], &d_p[i], sizeof(particle_t), cudaMemcpyDeviceToHost);

  // cleanup
  cudaFree(d_p);
  cudaFree(d_i);
  cudaFree(d_nparticles);

}