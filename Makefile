all:  parallel_IRAM

parallel_IRAM:	parallel_IRAM.c
	 mpicc -O3 -g -fopenmp parallel_IRAM.c mmio.c -lm -llapacke -lblas -o parallel_IRAM

clean:
	rm -rf *.o parallel_IRAM
