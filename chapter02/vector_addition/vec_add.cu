#include <stdio.h>
#include <stdlib.h>
 
__global__ void vec_add(int* x, int* y, int* z, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if( i < N ) {
		z[i] = x[i] + y[i];
	}
}
 
int main() { 
	int N = 123397730;
	int i;
    int *x, *y, *z;
    int *x_d, *y_d, *z_d;
	int numThreadsPerBlock = 256;

	x = (int*) malloc( N*sizeof(int));
	y = (int*) malloc( N*sizeof(int));
	z = (int*) malloc( N*sizeof(int));

	for( i=0; i<N; i++) {
		x[i] = i+1;
		y[i] = i+1;
	}

    cudaMalloc((void**) &x_d, N*sizeof(int));
    cudaMalloc((void**) &y_d, N*sizeof(int));
    cudaMalloc((void**) &z_d, N*sizeof(int));

	cudaMemcpy(x_d, x, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, N*sizeof(int), cudaMemcpyHostToDevice);

    vec_add<<< (N + numThreadsPerBlock - 1)/numThreadsPerBlock, numThreadsPerBlock >>>(x_d, y_d, z_d, N);

	cudaMemcpy(z, z_d, N*sizeof(int), cudaMemcpyDeviceToHost);

	for( i=0; i<N; i++) {
		if( z[i] != 2*(i+1) ){
			printf("ERROR\n");
			return 0;
		}
	}

	printf("OK\n");
    
    return 0;
}
