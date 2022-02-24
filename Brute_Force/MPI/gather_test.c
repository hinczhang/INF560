/* Source: https://stackoverflow.com/questions/31890523/how-to-use-mpi-gatherv-for-collecting-strings-of-diiferent-length-from-different */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#define nstrings 5
const char *const strings[nstrings] = {"Hello","world!","Bonjour","le","monde!"};

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv); 

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Everyone gets a string */    

    int myStringNum = rank % nstrings;
    char *mystring = (char *)strings[myStringNum];
    int mylen = strlen(mystring);

    printf("Rank %d: %s\n", rank, mystring);

    /*
     * Now, we Gather the string lengths to the root process, 
     * so we can create the buffer into which we'll receive the strings
     */

    // const int root = 0;
    int *recvcounts = NULL;

    /* Only root has the received data */
    // if (rank == root)
        recvcounts = malloc( size * sizeof(int)) ;

    MPI_Allgather(&mylen, 1, MPI_INT,
               recvcounts, 1, MPI_INT,
               MPI_COMM_WORLD);

    /*
     * Figure out the total length of string, 
     * and displacements for each rank 
     */

    int totlen = 0;
    int *displs = NULL;
    char *totalstring = NULL;

    // if (rank == root) {
        displs = malloc( size * sizeof(int) );

        displs[0] = 0;
        totlen += recvcounts[0]+1;

        for (int i=1; i<size; i++) {
           totlen += recvcounts[i]+1;   /* plus one for space or \0 after words */
           displs[i] = displs[i-1] + recvcounts[i-1] + 1;
        }

        /* allocate string, pre-fill with spaces and null terminator */
        totalstring = malloc(totlen * sizeof(char));            
        for (int i=0; i<totlen-1; i++)
            totalstring[i] = ' ';
        totalstring[totlen-1] = '\0';
    // }

    /* 
     * Now we have the receive buffer, counts, and displacements, and 
     * can gather the strings 
     */

    MPI_Allgatherv(mystring, mylen, MPI_CHAR,
                totalstring, recvcounts, displs, MPI_CHAR,
                MPI_COMM_WORLD);


    // if (rank == root) {
        printf("%d: <%s>\n", rank, totalstring);
        free(totalstring);
        free(displs);
        free(recvcounts);
    // }

    MPI_Finalize();
    return 0;
}