objects = forceComputation.o collisionWithWall.o binningAlgorithm.o gearsScheme.o box.o

all: $(objects)
	nvcc -g -arch=compute_35 -code=sm_35,compute_35 $(objects) -o dem

%.o: %.cu
	nvcc -x cu -g -arch=compute_35 -code=sm_35,compute_35 -I. -dc  $< -c -o $@

clean:
	rm -f *.o dem
