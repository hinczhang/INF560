#ifndef NBODY_TOOLS_H
#define NBODY_TOOLS_H
#include "nbody.cuh"

/* draw recursively the content of a node */
void draw_node(node_t* n);

/* print recursively the particles of a node */
void print_particles(FILE* f, node_t*n);

/* Initialize a node */
__device__ void init_node(node_t* n, node_t* parent, double x_min, double x_max, double y_min, double y_max);
/* Compute the position of a particle in a node and return
 * the quadrant in which it should be placed
 */
__device__ int get_quadrant(particle_t* particle, node_t*node);

/* inserts a particle in a node (or one of its children)  */
__device__ void insert_particle(particle_t* particle, node_t*node);


__global__ void init_alloc(int nb_blocks);

__device__ void free_node(node_t* n);

__device__ node_t* alloc_node();

__device__ void free_root(node_t*root);

#endif	/* NBODY_TOOLS_H */
