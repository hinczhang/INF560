#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ui.h"
#include "nbody.h"
#include "nbody_tools.h"
#include "nbody_alloc.h"

extern node_t* root;

/* draw recursively the content of a node */
void draw_node(node_t* n) {
#ifndef DISPLAY
  return;
#else
  if(!n)
    return;

#if DRAW_BOXES
  int x1 = POS_TO_SCREEN(n->x_min);
  int y1 = POS_TO_SCREEN(n->y_min);
  int x2 = POS_TO_SCREEN(n->x_max);
  int y2 = POS_TO_SCREEN(n->y_max);
  draw_rect(x1, y1, x2, y2);
#endif

  if(n->particle) {
    int x = POS_TO_SCREEN(n->particle->x_pos);
    int y = POS_TO_SCREEN(n->particle->y_pos);
    draw_point (x,y);
  }
  if(n->children) {
#if 0
    /* draw a red point that represents the center of the node */
    int x = POS_TO_SCREEN(n->x_center);
    int y = POS_TO_SCREEN(n->y_center);
    draw_red_point (x,y);
#endif

    int i;
    for(i=0; i<4; i++) {
      draw_node(&n->children[i]);
    }
  }

#endif
}


/* print recursively the particles of a node */
void print_particles(FILE* f, node_t*n) {
  if(!n) {
    return;
  }

  if(n->particle) {
    particle_t*p = n->particle;
    fprintf(f, "particle={pos=(%f,%f), vel=(%f,%f)}\n", p->x_pos, p->y_pos, p->x_vel, p->y_vel);
  }
  if(n->children) {
    int i;
    for(i=0; i<4; i++) {
      print_particles(f, &n->children[i]);
    }
  }
}

/* Initialize a node */
void init_node(node_t* n, node_t* parent, double x_min, double x_max, double y_min, double y_max) {
  n->parent = parent;
  n->children = NULL;
  n->n_particles = 0;
  n->particle = NULL;
  n->x_min = x_min;
  n->x_max = x_max;
  n->y_min = y_min;
  n->y_max = y_max;
  n->depth = 0;

  int depth=1;
  while(parent) {
    if(parent->depth < depth) {
      parent->depth = depth;
      depth++;
    }
    parent = parent->parent;
  }

  n->mass= 0;
  n->x_center = 0;
  n->y_center = 0;

  assert(x_min != x_max);
  assert(y_min != y_max);
}


/* Compute the position of a particle in a node and return
 * the quadrant in which it should be placed
 */
int get_quadrant(particle_t* particle, node_t*node) {
  double x_min = node->x_min;
  double x_max = node->x_max;
  double x_center = x_min+(x_max-x_min)/2;

  double y_min = node->y_min;
  double y_max = node->y_max;
  double y_center = y_min+(y_max-y_min)/2;

  assert(particle->x_pos>=node->x_min);
  assert(particle->x_pos<=node->x_max);
  assert(particle->y_pos>=node->y_min);
  assert(particle->y_pos<=node->y_max);

  if(particle->x_pos <= x_center) {
    if(particle->y_pos <= y_center) {
      return 0;
    } else {
      return 2;
    }
  } else {
    if(particle->y_pos <= y_center) {
      return 1;
    } else {
      return 3;
    }
  }
}

/* inserts a particle in a node (or one of its children)  */
void insert_particle(particle_t* particle, node_t*node) {
#if 0
  assert(particle->x_pos >= node->x_min);
  assert(particle->x_pos <= node->x_max);
  assert(particle->y_pos >= node->y_min);
  assert(particle->y_pos <= node->y_max);

  assert(particle->node == NULL);
#endif
  if(node->n_particles == 0 &&
     node->children == NULL) {
    assert(node->children == NULL);

    /* there's no particle. insert directly */
    node->particle = particle;
    node->n_particles++;

    node->x_center = particle->x_pos;
    node->y_center = particle->y_pos;
    node->mass = particle->mass;

    particle->node = node;
    assert(node->children == NULL);
    return;
  } else {
    /* There's already a particle */

    if(! node->children) {
      /* there's no children yet */
      /* create 4 children and move the already-inserted particle to one of them */
      //assert(node->x_min != node->x_max);
      node->children = alloc_node();
      double x_min = node->x_min;
      double x_max = node->x_max;
      double x_center = x_min+(x_max-x_min)/2;

      double y_min = node->y_min;
      double y_max = node->y_max;
      double y_center = y_min+(y_max-y_min)/2;

      init_node(&node->children[0], node, x_min, x_center, y_min, y_center);
      init_node(&node->children[1], node, x_center, x_max, y_min, y_center);
      init_node(&node->children[2], node, x_min, x_center, y_center, y_max);
      init_node(&node->children[3], node, x_center, x_max, y_center, y_max);

      /* move the already-inserted particle to one of the children */
      particle_t*ptr = node->particle;
      //assert(ptr->node == node);
      int quadrant = get_quadrant(ptr, node);
      node->particle = NULL;
      ptr->node = NULL;

      insert_particle(ptr, &node->children[quadrant]);
    }

    /* insert the particle to one of the children */
    int quadrant = get_quadrant(particle, node);
    node->n_particles++;

    //assert(particle->node == NULL);
    insert_particle(particle, &node->children[quadrant]);

    /* update the mass and center of the node */
    double total_mass = 0;
    double total_x = 0;
    double total_y = 0;
    int i;
    for(i=0; i<4; i++) {
      total_mass += node->children[i].mass;
      total_x += node->children[i].x_center*node->children[i].mass;
      total_y += node->children[i].y_center*node->children[i].mass;
    }
    node->mass = total_mass;
    node->x_center = total_x/total_mass;
    node->y_center = total_y/total_mass;
#if 0
    assert(node->particle == NULL);
    assert(node->n_particles > 0);
#endif
  }
}

/*
  Place particles in their initial positions.
*/
void all_init_particles(int num_particles, particle_t *particles)
{
  int    i;
  double total_particle = num_particles;

  for (i = 0; i < num_particles; i++) {
    particle_t *particle = &particles[i];
#if 0
    particle->x_pos = ((rand() % max_resolution)- (max_resolution/2))*2.0 / max_resolution;
    particle->y_pos = ((rand() % max_resolution)- (max_resolution/2))*2.0 / max_resolution;
    particle->x_vel = particle->y_pos;
    particle->y_vel = particle->x_pos;
#else
    particle->x_pos = i*2.0/nparticles - 1.0;
    particle->y_pos = 0.0;
    particle->x_vel = 0.0;
    particle->y_vel = particle->x_pos;
#endif
    particle->mass = 1.0 + (num_particles+i)/total_particle;
    particle->node = NULL;

    //insert_particle(particle, root);
  }
}


struct memory_t mem_node;

void init_alloc(int nb_blocks) {
  mem_init(&mem_node, 4*sizeof(node_t), nb_blocks);
}

/* allocate a block of 4 nodes */
node_t* alloc_node() {
  node_t*ret = mem_alloc(&mem_node);
  return ret;
}

void free_root(node_t*root) {
  free_node(root);
  mem_free(&mem_node, root);
}

void free_node(node_t* n) {
  if(!n) return;

  if(n->children) {
    //assert(n->n_particles > 0);
    int i;
    for(i=0; i<4; i++) {
      free_node(&n->children[i]);
    }
    mem_free(&mem_node, n->children);
  }
}
