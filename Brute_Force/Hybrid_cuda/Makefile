CC	= mpicc
NVCC = nvcc
CFLAGS	= -O0 -g -Wall -fopenmp
LDFLAGS = -lstdc++ -lcuda -L/usr/local/cuda-11.4.1/targets/x86_64-linux/lib -lcudart -lm -lpthread -lX11 -fopenmp 
NVCCFLAGS = -I.
VERBOSE	= 
TARGET	= nbody_brute_force 
OBJS	= ui.o xstuff.o nbody_tools.o nbody_alloc.o cuda_compute_force.o

#DISPLAY = -DDISPLAY
DISPLAY =

DUMP = -DDUMP_RESULT
#DUMP =

all: $(TARGET)

cuda_compute_force.o: cuda_compute_force.cu 
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

nbody_brute_force: nbody_brute_force.o $(OBJS)
	$(CC) $(VERBOSE) -o $@ $< $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< $(VERBOSE) $(DISPLAY) $(DUMP)


clean:
	rm -f *.o $(TARGET)
