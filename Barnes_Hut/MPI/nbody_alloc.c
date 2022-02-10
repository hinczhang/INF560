#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "nbody_alloc.h"

/**************************************************************************/
/* Routine d'initialisation de l'allocateur : constitue une liste chainee */
/* de nbMaxAlloc blocs de taille TAILLEBLOC, le premier bloc etant pointe */
/* par debutListe et le dernier pointant sur NULL                         */
/**************************************************************************/
void mem_init(struct memory_t *mem, size_t block_size, int nb_blocks)
{
  Bloc *ptr;
  int i;

  /* ATTENTION : vous n'avez droit de faire qu'un seul malloc de */
  /*             nbMaxAlloc*TAILLEBLOC ou bien un calloc !       */
  /* On alloue un unique bloc avec calloc. On y decoupera nos blocs */
  ptr = calloc(nb_blocks, block_size);
  assert(ptr != 0);

  /* Memorisation du debut de la liste chainee */
  mem->debutListe = ptr;
  mem->block_size = block_size;
  mem->nb_free = nb_blocks;

  /* Decoupage du bloc retourne par malloc en nbMaxBloc constituant une liste*/
  /* chainee                                                                 */
  char* block_ptr = (char*)ptr;
  Bloc*p = (Bloc*)block_ptr;
  for (i=0 ; i<(nb_blocks-1) ; i++) {
    p = (Bloc*) block_ptr;
    block_ptr += block_size;
    p->suivant = (Bloc*) block_ptr;
  }

  size_t alloc_size = nb_blocks*block_size;
  char* end_addr = (char*) mem->debutListe;
  end_addr += alloc_size;
  /* Le dernier bloc a son pointeur a NULL */
  p->suivant = NULL;
}

/**************************************************************************/
/* Fonction renvoyant un pointeur sur une zone memoire                    */
/* NB : on ne peut pas preciser la taille, puisque la taille est          */
/*      predefinie                                                        */
/**************************************************************************/
void *mem_alloc(struct memory_t*mem)
{
  Bloc *ptr;
  assert(mem->debutListe != NULL);

  ptr = mem->debutListe;
  mem->debutListe = mem->debutListe->suivant;
  memset(ptr, 0, mem->block_size);

  mem->nb_free--;
  return (void *)ptr;
}

/**************************************************************************/
/* Fonction liberant la zone memoire pointee par ptr                      */
/**************************************************************************/
void mem_free(struct memory_t* mem, void *ptr)
{
  Bloc *pBloc = (Bloc*)ptr;
  pBloc->suivant = mem->debutListe;
  mem->debutListe = pBloc;
  mem->nb_free++;
}
