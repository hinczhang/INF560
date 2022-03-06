#include <stdio.h>
#include <stdlib.h>
#include "nbody.h"

__global__ void __compute__force__ (particle_t*d_p, double*d_x_pos, double*d_y_pos, double*d_mass) 
{
  // obtain the global thread ID
//   int i = blockIdx.x * blockDim.x + threadIdx.x;

  // change the cpu version to cuda version
  double x_sep, y_sep, dist_sq, grav_base;

  x_sep = *d_x_pos - d_p->x_pos;
  y_sep = *d_y_pos - d_p->y_pos;
  dist_sq = MAX((x_sep*x_sep) + (y_sep*y_sep), 0.01);

  /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
  grav_base = GRAV_CONSTANT*(d_p->mass)*(*d_mass)/dist_sq;

  d_p->x_force += grav_base*x_sep;
  d_p->y_force += grav_base*y_sep;

}

extern "C" void cuda_compute_force(particle_t*p, double x_pos, double y_pos, double mass)
{

  particle_t * d_p;
  double * d_x_pos;
  double * d_y_pos;
  double * d_mass;

  // allocate space for device copies
  cudaMalloc((void **)&d_p, sizeof(particle_t));
  cudaMalloc((void **)&d_x_pos, sizeof(double));
  cudaMalloc((void **)&d_y_pos, sizeof(double)); 
  cudaMalloc((void **)&d_mass, sizeof(double));
  
  // copy inputs to device
  cudaMemcpy(d_p, p, sizeof(particle_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x_pos, &x_pos, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y_pos, &y_pos, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mass, &mass, sizeof(double), cudaMemcpyHostToDevice);

  __compute__force__<<<1,1>>>(d_p, d_x_pos, d_y_pos, d_mass);

  // copy result back to host
  cudaMemcpy(p, d_p, sizeof(particle_t), cudaMemcpyDeviceToHost);

  // cleanup
  cudaFree(d_p);
  cudaFree(d_x_pos);
  cudaFree(d_y_pos);
  cudaFree(d_mass);

}