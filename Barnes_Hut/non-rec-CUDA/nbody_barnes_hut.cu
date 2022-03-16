/*
** nbody_barnes_hut.c - nbody simulation that implements the Barnes-Hut algorithm (O(nlog(n)))
**
**/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <unistd.h>

#ifdef DISPLAY
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

#include "ui.cuh"
#include "nbody.cuh"
#include "nbody_tools.cuh"
#include "queue.cuh"
#define THR_PER_BLK 500
#define BLK_IN_GRD 2

FILE* f_out=NULL;

int nparticles=10;      /* number of particles */
float T_FINAL=1.0;     /* simulation end time */

__device__ particle_t* particles;
__device__ node_t *root;
__device__ int device_nparticles=0;
__device__ double sum_speed_sq = 0;
__device__ double max_acc = 0;
__device__ double max_speed = 0;

double max_acc_host = 0;
double max_speed_host = 0;
/* Initialize a node */
__global__ void init_node_outside(double x_min, double x_max, double y_min, double y_max){
  //root->parent = parent;
  root->children = NULL;
  root->n_particles = 0;
  root->particle = NULL;
  root->x_min = x_min;
  root->x_max = x_max;
  root->y_min = y_min;
  root->y_max = y_max;
  root->depth = 0;

  root->mass= 0;
  root->x_center = 0;
  root->y_center = 0;
  assert(x_min != x_max);
  assert(y_min != y_max);
}

/*
  Place particles in their initial positions.
*/
__global__ void all_init_particles(int num_particles)
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
    particle->x_pos = i*2.0/num_particles - 1.0;
    particle->y_pos = 0.0;
    particle->x_vel = 0.0;
    particle->y_vel = particle->x_pos;
#endif
    particle->mass = 1.0 + (num_particles+i)/total_particle;
    particle->node = NULL;

    //insert_particle(particle, root);
  }
}

__global__ void initial_root(){
  root = (node_t*)malloc(sizeof(node_t));
}

void init() {
  init_alloc<<<1,1>>>(4*nparticles);
  cudaDeviceSynchronize();
  initial_root<<<1,1>>>();
  cudaDeviceSynchronize();
  init_node_outside<<<1,1>>>(XMIN, XMAX, YMIN, YMAX);
  cudaDeviceSynchronize();
}

#ifdef DISPLAY
extern Display *theDisplay;  /* These three variables are required to open the */
extern GC theGC;             /* particle plotting window.  They are externally */
extern Window theMain;       /* declared in ui.h but are also required here.   */
#endif

/* compute the force that a particle with position (x_pos, y_pos) and mass 'mass'
 * applies to particle p
 */
__device__ void compute_force(particle_t*p, double x_pos, double y_pos, double mass) {
  double x_sep, y_sep, dist_sq, grav_base;

  x_sep = x_pos - p->x_pos;
  y_sep = y_pos - p->y_pos;
  dist_sq = MAX((x_sep*x_sep) + (y_sep*y_sep), 0.01);

  /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
  grav_base = GRAV_CONSTANT*(p->mass)*(mass)/dist_sq;

  p->x_force += grav_base*x_sep;
  p->y_force += grav_base*y_sep;
}

/* non-recursive stype of compting force on particle*/
__device__ void non_rec_compute_force_on_particle(particle_t *p){
  p->x_force = 0;
  p->y_force = 0;
  Queue *queue = CreateQueue();
  AddQ(queue, root);
  while(!IsEmptyQ(queue)){
    node_t *n=DeleteQ(queue);
    /*
    Computing Area begins
    *//*
    if(! n || n->n_particles==0) {
      continue;
    }*/
    if(n->particle) {
      /* only one particle */
      assert(n->children == NULL);
      compute_force(p, n->x_center, n->y_center, n->mass);
    }else{
      #define THRESHOLD 2
      double size = n->x_max - n->x_min; // width of n
      double diff_x = n->x_center - p->x_pos;
      double diff_y = n->y_center - p->y_pos;
      double distance = sqrt(diff_x*diff_x + diff_y*diff_y);
     
      if(size / distance < THRESHOLD) {
        compute_force(p, n->x_center, n->y_center, n->mass);
      }else{
        int i;
        if(n->children==NULL) continue;
        for(i=0;i<4;i++){
          AddQ(queue, &n->children[i]);
        }
      }
    }
    /*
    Computing Area ends
    */
    
  }
  free(queue);
}

/* compute the force that node n acts on particle p */
__device__ void compute_force_on_particle(node_t* n, particle_t *p) {
  if(! n || n->n_particles==0) {
    return;
  }
  if(n->particle) {
    /* only one particle */
    assert(n->children == NULL);

    /*
      If the current node is an external node (and it is not body b),
      calculate the force exerted by the current node on b, and add
      this amount to b's net force.
    */
    compute_force(p, n->x_center, n->y_center, n->mass);
  } else {
    /* There are multiple particles */

    #define THRESHOLD 2
    double size = n->x_max - n->x_min; // width of n
    double diff_x = n->x_center - p->x_pos;
    double diff_y = n->y_center - p->y_pos;
    double distance = sqrt(diff_x*diff_x + diff_y*diff_y);

#if BRUTE_FORCE
    /*
      Run the procedure recursively on each of the current
      node's children.
      --> This result in a brute-force computation (complexity: O(n*n))
    */
    int i;
    for(i=0; i<4; i++) {
      compute_force_on_particle(&n->children[i], p);
    }
#else
    /* Use the Barnes-Hut algorithm to get an approximation */
    if(size / distance < THRESHOLD) {
      /*
	The particle is far away. Use an approximation of the force
      */
      compute_force(p, n->x_center, n->y_center, n->mass);
    } else {
      /*
        Otherwise, run the procedure recursively on each of the current
	node's children.
      */
      int i;
      for(i=0; i<4; i++) {
	compute_force_on_particle(&n->children[i], p);
      }
    }
#endif
  }
}

__device__ void compute_force_in_node(node_t *n) {
  if(!n) return;

  if(n->particle) {
    particle_t*p = n->particle;
    p->x_force = 0;
    p->y_force = 0;
    compute_force_on_particle(root, p);
  }
  if(n->children) {
    int i;
    for(i=0; i<4; i++) {
      compute_force_in_node(&n->children[i]);
      
    }
  }
}

/* compute the new position/velocity */
__device__ void move_particle(particle_t*p, double step, node_t* new_root) {

  p->x_pos += (p->x_vel)*step;
  p->y_pos += (p->y_vel)*step;
  double x_acc = p->x_force/p->mass;
  double y_acc = p->y_force/p->mass;
  p->x_vel += x_acc*step;
  p->y_vel += y_acc*step;
  //printf("x_vel: %lf, y_vel: %lf, step: %lf, x_acc: %lf, y_acc: %lf\n",p->x_vel,p->y_vel,step,x_acc,y_acc);
  /* compute statistics */
  double cur_acc = (x_acc*x_acc + y_acc*y_acc);
  cur_acc = sqrt(cur_acc);
  double speed_sq = (p->x_vel)*(p->x_vel) + (p->y_vel)*(p->y_vel);
  double cur_speed = sqrt(speed_sq);

  sum_speed_sq += speed_sq;
  max_acc = MAX(max_acc, cur_acc);
  max_speed = MAX(max_speed, cur_speed);

  p->node = NULL;
  if(p->x_pos < new_root->x_min ||
     p->x_pos > new_root->x_max ||
     p->y_pos < new_root->y_min ||
     p->y_pos > new_root->y_max) {
    free(p);
    device_nparticles--;
  } else {
    insert_particle(p, new_root);
  }
}

/* compute the new position of the particles in a node */
__device__ void move_particles_in_node(node_t*n, double step, node_t *new_root) {
  if(!n) return;
  if(n->particle) {
    particle_t*p = n->particle;
    move_particle(p, step, new_root);
  }
  if(n->children) {
    int i;
    for(i=0; i<4; i++) {
      move_particles_in_node(&n->children[i], step, new_root);
    }
  }
}

/*
  Move particles one time step.

  Update positions, velocity, and acceleration.
  Return local computations.
*/
__global__ void compute_all_particles(double step){
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  int total_thread_num = THR_PER_BLK*BLK_IN_GRD;
  int threadTasks = (int)device_nparticles/total_thread_num;
  int up_limit = threadTasks;
  if(idx==total_thread_num-1) up_limit = device_nparticles-(total_thread_num-1)*threadTasks;
  int i;
  for(i=idx*threadTasks;i<idx*threadTasks+up_limit;i++){
    compute_force_in_node(particles[i].node);
    //non_rec_compute_force_on_particle(&particles[i]);
  }
}
__global__ void all_move_particles(double step)
{
  //i=device_nparticles-2;
  /*int i=0;
  for(;i<3;i++){
    printf("{%lf,%lf}\n",particles[i].x_pos,particles[i].y_pos);
  }
  printf("\n");*/
  node_t* new_root = (node_t*)malloc(sizeof(node_t));
  init_node(new_root, NULL, XMIN, XMAX, YMIN, YMAX);
  /* then move all particles and return statistics */
  move_particles_in_node(root, step, new_root);

  free_node(root);
  free(root);
  root = new_root; 
}

void run_simulation() {
  double t = 0.0, dt = 0.01;

  while (t < T_FINAL && nparticles>0) {
    /* Update time. */
    t += dt;
    //printf("\nDT:%lf\n",dt);
    
    /* Move particles with the current and compute rms velocity. */
    compute_all_particles<<<THR_PER_BLK, BLK_IN_GRD>>>(dt);
    cudaDeviceSynchronize();
    all_move_particles<<<1,1>>>(dt);
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&max_speed_host, max_speed, sizeof(double));
    cudaMemcpyFromSymbol(&max_acc_host, max_acc, sizeof(double));
    //printf("DT TIME: {%lf,%lf}\n",max_speed_host, max_acc_host);
    /* Adjust dt based on maximum speed and acceleration--this
       simple rule tries to insure that no velocity will change
       by more than 10% */
    
    dt = 0.1*max_speed_host/max_acc_host;
    
    /* Plot the movement of the particle */
#if DISPLAY
    node_t *n = root;
    clear_display();
    draw_node(n);
    flush_display();
#endif
  }
  //cudaError_t cudaStatus = cudaGetLastError();
  //printf(cudaGetErrorString(cudaStatus));
}

/* create a quad-tree from an array of particles */
__global__ void insert_all_particles(int num) {
  int i;
  for(i=0; i<num; i++) {
    insert_particle(&particles[i], root);
  }
}

/*
  Simulate the movement of nparticles particles.
*/

__global__ void malloc_particles(int num){
  particles = (particle_t*)malloc(sizeof(particle_t)*num);
}

int main(int argc, char**argv)
{
  size_t Stack_Size, heapSize;
  cudaDeviceGetLimit(&Stack_Size, cudaLimitStackSize);
  cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize);
  cudaDeviceSetLimit(cudaLimitStackSize,Stack_Size*8);
  cudaDeviceSetLimit(cudaLimitMallocHeapSize,heapSize*8);
 // cudaDeviceGetLimit(&Stack_Size, cudaLimitStackSize);

  if(argc >= 2) {
    nparticles = atoi(argv[1]);
  }
  if(argc == 3) {
    T_FINAL = atof(argv[2]);
  }
  
  cudaMemcpyToSymbol(device_nparticles,&nparticles,sizeof(int));


  init();

  /* Allocate global shared arrays for the particles data set. */
  malloc_particles<<<1,1>>>(nparticles);
  cudaDeviceSynchronize();
  
  
  all_init_particles<<<1,1>>>(nparticles);
  cudaDeviceSynchronize();
  insert_all_particles<<<1,1>>>(nparticles);
  cudaDeviceSynchronize();
  
  /* Initialize thread data structures */
#ifdef DISPLAY
  /* Open an X window to display the particles */
  simple_init (100,100,DISPLAY_SIZE, DISPLAY_SIZE);
#endif

  struct timeval t1, t2;
  gettimeofday(&t1, NULL);

  /* Main thread starts simulation ... */
  run_simulation();

  gettimeofday(&t2, NULL);

  double duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
  
#ifdef DUMP_RESULT
  FILE* f_out = fopen("particles_cm.log", "w");
  assert(f_out);
  print_particles(f_out, root);
  fclose(f_out);
#endif

  printf("-----------------------------\n");
  printf("nparticles: %d\n", nparticles);
  printf("T_FINAL: %f\n", T_FINAL);
  printf("-----------------------------\n");
  printf("Simulation took %lf s to complete\n", duration);

#ifdef DISPLAY
  node_t *n = root;
  clear_display();
  draw_node(n);
  flush_display();

  printf("Hit return to close the window.");

  getchar();
  /* Close the X window used to display the particles */
  XCloseDisplay(theDisplay);
#endif
  particle_t* host_particles = (particle_t*)malloc(sizeof(particle_t)*nparticles);
  cudaMemcpyFromSymbol(host_particles, particles, sizeof(particle_t)*nparticles);
  cudaDeviceSynchronize();
  free(host_particles);
  return 0;
}
