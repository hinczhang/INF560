CC	= nvcc
CFLAGS	= -O0 -g
LDFLAGS = -g -lm -lpthread -lX11
VERBOSE	= -I.
TARGET	= nbody_barnes_hut
OBJS	= ui.o xstuff.o nbody_tools.o nbody_alloc.o queue.o

#DISPLAY = -DDISPLAY
#DISPLAY =

#DUMP = -DDUMP_RESULT
DUMP =

all: $(TARGET)

nbody_brute_force: nbody_brute_force.o $(OBJS)
	$(CC) $(VERBOSE) -o $@ $< $(OBJS) $(LDFLAGS)

nbody_barnes_hut: nbody_barnes_hut.o $(OBJS)
	$(CC) $(VERBOSE) -o $@ $< $(OBJS)  $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< $(VERBOSE) $(DUMP) -dc
clean:
	rm -f *.o $(TARGET)
	rm -f nbody_alloc
	rm -f nbody_tools
	rm -f ui
	rm -f xstuff
	rm -f xstuff queue
