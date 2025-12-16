export PATH := /usr/local/cuda/bin:$(PATH)
export LD_LIBRARY_PATH := /usr/local/cuda/lib64:$(LD_LIBRARY_PATH)

#all: run clean 
all: vec_add

run: vec_add 
	./vec_add

vec_add: vec_add.cu
	nvcc -o vec_add vec_add.cu

clean:
	rm -f vec_add
