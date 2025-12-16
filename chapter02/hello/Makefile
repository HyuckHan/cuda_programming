export PATH := /usr/local/cuda/bin:$(PATH)
export LD_LIBRARY_PATH := /usr/local/cuda/lib64:$(LD_LIBRARY_PATH)

all: hello

run: hello 
	./hello

hello: hello.cu
	nvcc -o hello hello.cu

clean:
	rm -f hello
