#include "common/book.h"

#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blockPerGrid = imin( 32, (N+threadsPerBlock-1) / threadsPerBlock);

__global__ void dot( float *a, float *b, float *c) {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;    
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

int main(void) {
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    partial_c = (float *)malloc(blockPerGrid * sizeof(float));

    //init
    for(int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    HANDLE_ERROR( cudaMallocManaged(&dev_a, N * sizeof(float)) );
    HANDLE_ERROR( cudaMallocManaged(&dev_b, N * sizeof(float)) );
    HANDLE_ERROR( cudaMallocManaged(&dev_partial_c, blockPerGrid * sizeof(float)) );

    HANDLE_ERROR( cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice));

    dot<<<blockPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    HANDLE_ERROR( cudaMemcpy(partial_c, dev_partial_c, blockPerGrid*sizeof(float), cudaMemcpyDeviceToHost));

    c = 0;

    for (int i = 0; i < blockPerGrid; i++) {
        c += partial_c[i];
    }

    #define sum_square(x) (x*(x+1)*(2*x+1)/6)
    printf( "Valor da GPU %.6g = %.6g\n", c, 2 * sum_square( (float) (N -1)));

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    free(a);
    free(b);
    free(partial_c);
}