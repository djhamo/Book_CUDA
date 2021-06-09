#include <windows.h>
#include <iostream>
#include <math.h>
//#include <sys/time.h>

void add(int n, float *x, float *y) {
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
void add2(int n, float *x, float *y) {
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

__global__
void add3(int n, float *x, float *y) {
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

__global__
void add4(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

#define BILLION  1000000000L
int main(void) {

    printf("Add sem CUDA\n");
    //Sem CUDA
    int N = 1<<20; // 1M elements

    //Inicializacao
    float *x = new float[N];
    float *y = new float[N];
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    SYSTEMTIME time;
    GetSystemTime(&time);
    LONG start = (time.wSecond * 1000) + time.wMilliseconds;
    // Run kernel on 1M elements on the CPU
    add(N, x, y);
    GetSystemTime(&time);
    LONG stop = (time.wSecond * 1000) + time.wMilliseconds;
    float accum = stop - start;
	printf("difference is %0.2lf miliseconds\n", accum);

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    delete [] x;
    delete [] y;

    // Com CUDA    
    //clock_gettime(CLOCK_REALTIME, &start);

    printf("Add2\n");
    // Allocate Unified Memory -- accessible from CPU or GPU
    N = 1<<20;
    float *_x, *_y;
    cudaMallocManaged(&_x, N*sizeof(float));
    cudaMallocManaged(&_y, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        _x[i] = 1.0f;
        _y[i] = 2.0f;
        //printf("%20d => %f %f \n", i, _x[i], _y[i]);
    }
    
    GetSystemTime(&time);
    start = (time.wSecond * 1000) + time.wMilliseconds;
    // Run kernel on 1M elements on the GPU
    add2<<<1, 1>>>(N, _x, _y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    GetSystemTime(&time);
    stop = (time.wSecond * 1000) + time.wMilliseconds;
    accum = stop - start;
	printf("difference is %0.2lf miliseconds\n", accum);

    //Tunning    
    // Check for errors (all values should be 3.0f)
    maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(_y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;


    printf("Add3\n");
    for (int i = 0; i < N; i++) {
        _x[i] = 1.0f;
        _y[i] = 2.0f;
        //printf("%20d => %f %f \n", i, _x[i], _y[i]);
    }
    
    GetSystemTime(&time);
    start = (time.wSecond * 1000) + time.wMilliseconds;
    // Run kernel on 1M elements on the GPU
    add3<<<1, 256>>>(N, _x, _y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    GetSystemTime(&time);
    stop = (time.wSecond * 1000) + time.wMilliseconds;
    accum = stop - start;
	printf("difference is %0.2lf miliseconds\n", accum);

    // Check for errors (all values should be 3.0f)
    maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(_y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;


    printf("Add4\n");
    for (int i = 0; i < N; i++) {
        _x[i] = 1.0f;
        _y[i] = 2.0f;
        //printf("%20d => %f %f \n", i, _x[i], _y[i]);
    }
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    GetSystemTime(&time);
    start = (time.wSecond * 1000) + time.wMilliseconds;
    add4<<<numBlocks, blockSize>>>(N, _x, _y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    GetSystemTime(&time);
    stop = (time.wSecond * 1000) + time.wMilliseconds;
    accum = stop - start;
	printf("difference is %0.2lf miliseconds\n", accum);

    // Check for errors (all values should be 3.0f)
    maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(_y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(_x);
    cudaFree(_y);

	//clock_gettime(CLOCK_REALTIME, &end);

    //accum = (( end.tv_sec - start.tv_sec ) * BILLION) + ( end.tv_nsec - start.tv_nsec );
	//printf("difference is %0.2lf nanoseconds\n", accum);

    return 0;
}