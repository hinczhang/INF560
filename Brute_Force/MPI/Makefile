CC	= mpicc
CFLAGS	= -O0 -g -Wall
LDFLAGS = -lm -lpthread -lX11 -fopenmp
VERBOSE	=
TARGET	= nbody_brute_force
OBJS	= ui.o xstuff.o nbody_tools.o nbody_alloc.o

#DISPLAY = -DDISPLAY
DISPLAY =

DUMP = -DDUMP_RESULT
#DUMP =

all: $(TARGET)

nbody_brute_force: nbody_brute_force.o $(OBJS)
	$(CC) $(VERBOSE) -o $@ $< $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< $(VERBOSE) $(DISPLAY) $(DUMP)
clean:
	rm -f *.o $(TARGET)
