export PATH := /usr/local/cuda/bin:$(PATH)
export LD_LIBRARY_PATH := /usr/local/cuda/lib64:$(LD_LIBRARY_PATH)

all: mat_mul

run: mat_mul 
	./mat_mul

mat_mul: mat_mul.cu
	nvcc -o mat_mul mat_mul.cu

clean:
	rm -f mat_mul
