#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 32

#define DEBUG_FLAG
 
inline unsigned int cdiv(unsigned int a, unsigned int b) { 
	return (a + b - 1) / b;
}

__global__ void mat_mul(long* x, long* y, long* z, int M, int N, int K) {
	__shared__ long A_s[TILE_DIM][TILE_DIM];
	__shared__ long B_s[TILE_DIM][TILE_DIM];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	long sum = 0;

	for(unsigned int tile = 0; tile < (N+TILE_DIM-1)/TILE_DIM; ++tile) {
		// Load tile to shared memory 
		if (row < M && (tile * TILE_DIM + threadIdx.x) < N) 
			A_s[threadIdx.y][threadIdx.x] = x[row*N + tile*TILE_DIM + threadIdx.x]; //x[row][tile*TILE_DIM + threadIdx.x]
		else
			A_s[threadIdx.y][threadIdx.x] = 0;

		if ((tile * TILE_DIM + threadIdx.y) < N && (col < K)) 
			B_s[threadIdx.y][threadIdx.x] = y[(tile*TILE_DIM + threadIdx.y)*K + col]; //y[tile*TILE_DIM + threadIdx.y][col]
		else
			B_s[threadIdx.y][threadIdx.x] = 0;

		__syncthreads();

		// Compute with tile
		for(unsigned int i = 0; i < TILE_DIM; ++i) {
			sum += A_s[threadIdx.y][i]*B_s[i][threadIdx.x];
		}

		__syncthreads();

	}

	if( row < M && col < K ) 
		z[row*K + col] = sum; //z[row][col]

}

void mat_mul_host(long* x, long* y, long* z, int M, int N, int K) {
	int col, row;
	for( row = 0; row < M; row++ ) {
		for( col = 0; col < K; col++) {
			long sum = 0;
			int i;

			for( i = 0; i < N ; i++ ){
#ifdef DEBUG_FLAG
				if(row==0 && col ==0)
					printf("%ld %ld\n", x[row*N+i],y[i*K+col]);
#endif
				sum += x[row*N+i]*y[i*K+col];
			}

			z[row*K+col] = sum;
		}
	}
}

void mat_print(long* x, int M, int K) {
	int col, row;
	for( row = 0; row < M; row++ ) {
		for( col = 0; col < K; col++) { 
			printf("%ld ", x[row*K+col]);
		} 
		printf("\n");
	}
}

int main() { 
#ifdef DEBUG_FLAG
	int M = 10;
	int N = 4;
	int K = 20;
#else
	int M = 1000;
	int N = 256;
	int K = 123;
#endif

	int i;
    long *x, *y, *z;
    long *x_d, *y_d, *z_d;

	dim3 dimBlock(TILE_DIM, TILE_DIM);
	dim3 dimGrid(cdiv(K, dimBlock.x), cdiv(M, dimBlock.y));

	x = (long*) malloc( M*N*sizeof(long) );
	y = (long*) malloc( N*K*sizeof(long) );
	z = (long*) malloc( M*K*sizeof(long) );

	for( i=0; i<M*N; i++) 
		x[i] = i+1;

	for( i=0; i<N*K; i++)
		y[i] = i+1;

    cudaMalloc((void**) &x_d, M*N*sizeof(long));
    cudaMalloc((void**) &y_d, N*K*sizeof(long));
    cudaMalloc((void**) &z_d, M*K*sizeof(long));

	cudaMemcpy(x_d, x, M*N*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, N*K*sizeof(long), cudaMemcpyHostToDevice);

#ifdef DEBUG_FLAG
    mat_mul_host(x, y, z, M, N, K);

	printf("\n");
    mat_print(x, M, N);
	printf("\n");
    mat_print(y, N, K);
	printf("\n");
    mat_print(z, M, K);
#endif

	for( i=0; i<M*K; i++) 
		z[i] = 0;
    mat_mul<<<dimGrid, dimBlock>>>(x_d, y_d, z_d, M, N, K);
	cudaMemcpy(z, z_d, M*K*sizeof(long), cudaMemcpyDeviceToHost);

#ifdef DEBUG_FLAG
	printf("\n");
    mat_print(z, M, K);
#endif

	printf("OK\n");
    
    return 0;
}
