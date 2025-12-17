export PATH := /usr/local/cuda/bin:$(PATH)
export LD_LIBRARY_PATH := /usr/local/cuda/lib64:$(LD_LIBRARY_PATH)

all: dev_prop

run: dev_prop 
	./dev_prop

dev_prop: dev_prop.cu
	nvcc -o dev_prop dev_prop.cu

clean:
	rm -f dev_prop
