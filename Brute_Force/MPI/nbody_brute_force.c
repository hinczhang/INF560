/*
** nbody_brute_force.c - nbody simulation using the brute-force algorithm (O(n*n))
**
**/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <unistd.h>
#include <mpi.h>

#ifdef DISPLAY
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

#include "ui.h"
#include "nbody.h"
#include "nbody_tools.h"

FILE* f_out=NULL;

int nparticles=10;      /* number of particles */
float T_FINAL=1.0;     /* simulation end time */
particle_t*particles;

double sum_speed_sq = 0;
double max_acc = 0;
double max_speed = 0;

/* MPI_Tag for two types of jobs  */
int ACC_TAG = 97;
int SPEED_TAG = 96;
double dt = 0.01;
int step = 0;

void init() {
  /* Nothing to do */
}

#ifdef DISPLAY
Display *theDisplay;  /* These three variables are required to open the */
GC theGC;             /* particle plotting window.  They are externally */
Window theMain;       /* declared in ui.h but are also required here.   */
#endif

/* compute the force that a particle with position (x_pos, y_pos) and mass 'mass'
 * applies to particle p
 */
void compute_force(particle_t*p, double x_pos, double y_pos, double mass) {
  double x_sep, y_sep, dist_sq, grav_base;

  x_sep = x_pos - p->x_pos;
  y_sep = y_pos - p->y_pos;
  dist_sq = MAX((x_sep*x_sep) + (y_sep*y_sep), 0.01);

  /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
  grav_base = GRAV_CONSTANT*(p->mass)*(mass)/dist_sq;

  p->x_force += grav_base*x_sep;
  p->y_force += grav_base*y_sep;

}

/* compute the new position/velocity */
void move_particle(particle_t*p, double step) {

  p->x_pos += (p->x_vel)*step;
  p->y_pos += (p->y_vel)*step;
  double x_acc = p->x_force/p->mass;
  double y_acc = p->y_force/p->mass;
  p->x_vel += x_acc*step;
  p->y_vel += y_acc*step;

  /* compute statistics */
  double cur_acc = (x_acc*x_acc + y_acc*y_acc);
  cur_acc = sqrt(cur_acc);
  double speed_sq = (p->x_vel)*(p->x_vel) + (p->y_vel)*(p->y_vel);
  double cur_speed = sqrt(speed_sq);

  sum_speed_sq += speed_sq;
  max_acc = MAX(max_acc, cur_acc);
  max_speed = MAX(max_speed, cur_speed);
}

/* display all the particles */
void draw_all_particles() {
  int i;
  for(i=0; i<nparticles; i++) {
    int x = POS_TO_SCREEN(particles[i].x_pos);
    int y = POS_TO_SCREEN(particles[i].y_pos);
    draw_point(x,y);
  }
}

void print_all_particles(FILE* f) {
  int i;
  for(i=0; i<nparticles; i++) {
    particle_t*p = &particles[i];
    fprintf(f, "particle={pos=(%f,%f), vel=(%f,%f)}\n", p->x_pos, p->y_pos, p->x_vel, p->y_vel);
  }
}

/*
  Simulate the movement of nparticles particles.
*/
int main(int argc, char**argv)
{

  if(argc >= 2) {
    nparticles = atoi(argv[1]);
  }
  if(argc == 3) {
    T_FINAL = atof(argv[2]);
  }

  init();
  int rank, size;

  /* MPI Initialization */
  MPI_Init(&argc, &argv);
  
  /* Get the rank of the current task and the number
   * of MPI processe
   */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size == 1) {
    printf("Please salloc more than 1 rank for MPI!\n");
    exit(1);
  }
  MPI_Status status;

  /* Allocate global shared arrays for the particles data set. */
  particles = malloc(sizeof(particle_t)*nparticles);
  all_init_particles(nparticles, particles);

  /* Initialize thread data structures */
#ifdef DISPLAY
  /* Open an X window to display the particles */
  simple_init (100,100,DISPLAY_SIZE, DISPLAY_SIZE);
#endif

  double t1, t2, duration;

  /* Start simulation */
  double t = 0.0, dt = 0.01;
  int i, j;
  int nums_per_proc = nparticles/(size-1);
  int root_task = nparticles - nums_per_proc*(size-1);
  particle_t* par_per_proc;

  /* Create MPI type for collective communication */
  MPI_Datatype particle_mpi_t;
  int blocklens[1] = {7};
  MPI_Aint offsets[1] = {0};
  MPI_Datatype types[1] =  {MPI_DOUBLE};
  MPI_Type_create_struct(1, blocklens, offsets, types, &particle_mpi_t);
  MPI_Type_commit(&particle_mpi_t);

  /* Pre-define the displacements, counts for gathering particles[..] from slave procs */  
  int *displs = NULL;
  int *counts = NULL;

  if (rank == 0){
    displs = malloc(size * sizeof(int));
    counts = malloc(size * sizeof(int));
    displs[0] = 0;
    counts[0] = root_task;

    for (i = 1; i < size; i++) {
      displs[i] = root_task + nums_per_proc * (i-1);
      counts[i] = nums_per_proc;
    }
  }

  while (t < T_FINAL && nparticles > 0) {
    /* Update time. */
    t += dt;
    /* Move particles with the current and compute rms velocity. */

    /* 1. Computing task */
    if (rank != 0) {
      // normal tasks for nums_per_proc in nparticles
      if(step==0) {
        par_per_proc = malloc(sizeof(particle_t)*nums_per_proc);
        if (par_per_proc == NULL) {
          fprintf(stderr, "Fatal: failed to allocate bytes.\n");
          abort();
        }
      }

      for (i = root_task + nums_per_proc*(rank-1); i < root_task + nums_per_proc * rank; i++){
        particles[i].x_force = 0;
        particles[i].y_force = 0;
        for(j = 0; j < nparticles; j++) {
          particle_t*p = &particles[j];
          compute_force(&particles[i], p->x_pos, p->y_pos, p->mass);
        }
        par_per_proc[i-root_task-nums_per_proc*(rank-1)] = particles[i];
      }

      MPI_Send(&max_acc, 1, MPI_DOUBLE, 0, ACC_TAG, MPI_COMM_WORLD);
      MPI_Send(&max_speed, 1, MPI_DOUBLE, 0, SPEED_TAG, MPI_COMM_WORLD);

    } else {
      
      nums_per_proc = root_task;

      /* Alloc particles arrays for current proc */
      if(step==0) {
        t1 = MPI_Wtime();
        printf("t1 = %f\n", t1);
        par_per_proc = malloc(sizeof(particle_t)*nums_per_proc);
        if (par_per_proc == NULL) {
          fprintf(stderr, "Fatal: failed to allocate bytes.\n");
          abort();
        }
      }

      /* Executing computing task of root */
      for (i = 0; i < nums_per_proc; i++){
        particles[i].x_force = 0;
        particles[i].y_force = 0;
        for(j = 0; j < nparticles; j++) {
          particle_t*p = &particles[j];
          compute_force(&particles[i], p->x_pos, p->y_pos, p->mass);
        }
        par_per_proc[i] = particles[i]; 
      }

      /* Recv the max_acc and max_speed from other procs */
      for (i = 1; i < size; i++) {
        double max_acc_recv, max_speed_recv;
        MPI_Recv(&max_acc_recv, 1, MPI_DOUBLE, i, ACC_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&max_speed_recv, 1, MPI_DOUBLE, i, SPEED_TAG, MPI_COMM_WORLD, &status);

        if (max_acc_recv > max_acc) 
          max_acc = max_acc_recv;
        if (max_speed_recv > max_speed)
          max_speed = max_speed_recv;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Gatherv(par_per_proc, nums_per_proc, particle_mpi_t,
                particles, counts, displs, particle_mpi_t,
                0, MPI_COMM_WORLD);
  

    /* 2. Move task (only in root) */ 
    if(rank == 0){
      for(i = 0; i < nparticles; i++) {
        move_particle(&particles[i], dt);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // send new positions, forces, acc
    MPI_Bcast(particles, nparticles, particle_mpi_t, 0, MPI_COMM_WORLD);

    MPI_Bcast(&max_speed, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_acc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    /* Adjust dt based on maximum speed and acceleration--this
       simple rule tries to insure that no velocity will change
       by more than 10% */

    dt = 0.1*max_speed/max_acc;
    step++;

#if DISPLAY
    clear_display();
    draw_all_particles();
    flush_display();
#endif
  } 

  if (rank == 0) {
    t2 = MPI_Wtime();
    printf("t2 = %f\n", t2);
    duration = t2 - t1;
  }
  t2 = MPI_Wtime();
  duration = t2 - t1;

#ifdef DUMP_RESULT
  FILE* f_out = fopen("particles.log", "w");
  assert(f_out);
  print_all_particles(f_out);
  fclose(f_out);
#endif

  free(par_per_proc);
  free(particles);

  if (rank==0) {
    printf("-----------------------------\n");
    printf("nparticles: %d\n", nparticles);
    printf("T_FINAL: %f\n", T_FINAL);
    printf("-----------------------------\n");
    printf("Simulation took %lf s to complete\n", duration);
  }

#ifdef DISPLAY
  clear_display();
  draw_all_particles();
  flush_display();

  printf("Hit return to close the window.");

  getchar();
  /* Close the X window used to display the particles */
  XCloseDisplay(theDisplay);
#endif
  MPI_Finalize();
  return 0;
}
