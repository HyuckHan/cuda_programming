#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//#define DEBUG_FLAG
 
inline unsigned int cdiv(unsigned int a, unsigned int b) { 
	return (a + b - 1) / b;
}

__global__ void mat_mul(int* x, int* y, int* z, int M, int N, int K) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if( row < M && col < K ) {
		int sum = 0;
		int i;

		for( i = 0; i < N ; i++ ) 
			z[row*K+col] += x[row*N+i]*y[i*K+col];
			//sum += x[row*N+i]*y[i*K+col];

		//z[row*K+col] = sum;
	}

}

void mat_mul_host(int* x, int* y, int* z, int M, int N, int K) {
	int col, row;
	for( row = 0; row < M; row++ ) {
		for( col = 0; col < K; col++) {
			int sum = 0;
			int i;

			for( i = 0; i < N ; i++ ){
#ifdef DEBUG_FLAG
				if(row==0 && col ==0)
					printf("%d %d\n", x[row*N+i],y[i*K+col]);
#endif
				sum += x[row*N+i]*y[i*K+col];
			}

			z[row*K+col] = sum;
		}
	}
}

void mat_print(int* x, int M, int K) {
	int col, row;
	for( row = 0; row < M; row++ ) {
		for( col = 0; col < K; col++) { 
			printf("%d ", x[row*K+col]);
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
	int M = 16384;
	int N = 16384;
	int K = 16384;
#endif

	int i;
    int *x, *y, *z;
    int *x_d, *y_d, *z_d;

	dim3 dimBlock(16, 16);
	dim3 dimGrid(cdiv(K, dimBlock.x), cdiv(M, dimBlock.y));

	cudaEvent_t start, stop;
	clock_t start_host, stop_host;
	float milliseconds = 0.0;

	x = (int*) malloc( M*N*sizeof(int) );
	y = (int*) malloc( N*K*sizeof(int) );
	z = (int*) malloc( M*K*sizeof(int) );

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for( i=0; i<M*N; i++) 
		x[i] = i+1;

	for( i=0; i<N*K; i++)
		y[i] = i+1;

	cudaEventRecord(start);
	cudaMalloc((void**) &x_d, M*N*sizeof(int));
	cudaMalloc((void**) &y_d, N*K*sizeof(int));
	cudaMalloc((void**) &z_d, M*K*sizeof(int));

	cudaMemcpy(x_d, x, M*N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, N*K*sizeof(int), cudaMemcpyHostToDevice);

	mat_mul<<<dimGrid, dimBlock>>>(x_d, y_d, z_d, M, N, K);
	cudaMemcpy(z, z_d, M*K*sizeof(int), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);

	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("OK %fms\n", milliseconds);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

#ifdef DEBUG_FLAG
	printf("\n");
	mat_print(x, M, N);
	printf("\n");
	mat_print(y, N, K);
	printf("\n");
	mat_print(z, M, K);
#endif

	start_host = clock();
	for( i=0; i<M*K; i++) 
		z[i] = 0;
	mat_mul_host(x, y, z, M, N, K);
	stop_host = clock();
	printf("OK %lfms\n", ((double)stop_host - start_host)*1000 / CLOCKS_PER_SEC);

#ifdef DEBUG_FLAG
	printf("\n");
	mat_print(z, M, K);
	printf("\n");
#endif

	free(x);
	free(y);
	free(z);

	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(z_d);

	return 0;
}
