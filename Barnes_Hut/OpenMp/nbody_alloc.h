#ifndef NBODY_ALLOC_H
#define NBODY_ALLOC_H

typedef struct Bloc {
  struct Bloc* suivant;
} Bloc;

struct memory_t {
  Bloc *debutListe;
  size_t block_size;
  unsigned nb_free;
};


void mem_init(struct memory_t *mem, size_t block_size, int nb_blocks);
void *mem_alloc(struct memory_t* mem);
void mem_free(struct memory_t* mem, void *ptr);

#endif
