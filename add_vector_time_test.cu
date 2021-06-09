#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "common/book.h"

// include OS specific timing library
#ifdef _MSC_VER
// Windows
#include <Windows.h>
#else
// Linux
#include <time.h>
#endif

/// return a timestamp with sub-second precision
/** QueryPerformanceCounter and clock_gettime have an undefined starting point (null/zero)
and can wrap around, i.e. be nulled again. **/
double seconds() {
#ifdef _MSC_VER
    static LARGE_INTEGER frequency;
    if (frequency.QuadPart == 0)
    ::QueryPerformanceFrequency(&frequency);
    LARGE_INTEGER now;
    ::QueryPerformanceCounter(&now);
    return now.QuadPart / double(frequency.QuadPart);
#else
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + now.tv_nsec / 1000000000.0;
#endif
}
    
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

    double start, end;
    //Sem CUDA
    int N = 1<<20; // 1M elements

    //Inicializacao
    float *x = new float[N];
    float *y = new float[N];
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }


    start = seconds();
    // Run kernel on 1M elements on the CPU
    add(N, x, y);

	end =  seconds();
    printf("difference is %f nanoseconds\n", (end - start));

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    printf("Max error: %.6f\n", maxError);

    // Free memory
    delete [] x;
    delete [] y;

    printf("Add2\n");
    // Allocate Unified Memory -- accessible from CPU or GPU
    N = 1<<20;
    float *_x, *_y;
    HANDLE_ERROR( cudaMallocManaged(&_x, N*sizeof(float)) );
    HANDLE_ERROR( cudaMallocManaged(&_y, N*sizeof(float)) );

    for (int i = 0; i < N; i++) {
        _x[i] = 1.0f;
        _y[i] = 2.0f;
        //printf("%20d => %f %f \n", i, _x[i], _y[i]);
    }
    
    start = seconds();
    // Run kernel on 1M elements on the GPU
    add2<<<1, 1>>>(N, _x, _y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

	end =  seconds();
    printf("difference is %f nanoseconds\n", (end - start));
   
    //Tunning    
    // Check for errors (all values should be 3.0f)
    maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(_y[i]-3.0f));
    printf("Max error: %.6f\n", maxError);

    printf("Add3\n");
    for (int i = 0; i < N; i++) {
        _x[i] = 1.0f;
        _y[i] = 2.0f;
        //printf("%20d => %f %f \n", i, _x[i], _y[i]);
    }
    
    start = seconds();
    // Run kernel on 1M elements on the GPU
    add3<<<1, 256>>>(N, _x, _y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
	end =  seconds();
    printf("difference is %f nanoseconds\n", (end - start));

    // Check for errors (all values should be 3.0f)
    maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(_y[i]-3.0f));
    printf("Max error: %.6f\n", maxError);


    printf("Add4\n");
    for (int i = 0; i < N; i++) {
        _x[i] = 1.0f;
        _y[i] = 2.0f;
        //printf("%20d => %f %f \n", i, _x[i], _y[i]);
    }
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    start = seconds();
    add4<<<numBlocks, blockSize>>>(N, _x, _y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
	end =  seconds();
    printf("difference is %f nanoseconds\n", (end - start));

    // Check for errors (all values should be 3.0f)
    maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(_y[i]-3.0f));
    printf("Max error: %.6f\n", maxError);

    printf ("Teste de parametros\n");
    cudaDeviceProp prop;
    
    double diff;
    double MIN_TIME = 5000L;
    int BEST_BLOCK = 0;
    int BEST_THREAD = 0;

    HANDLE_ERROR (cudaGetDeviceProperties (&prop, 0));
    int MAX_BLOCk = prop.multiProcessorCount;

    MAX_BLOCk = (N + prop.maxThreadsPerBlock - 1) / prop.maxThreadsPerBlock;
    printf("Max Block %d - Max Thread %d \n", MAX_BLOCk, prop.maxThreadsPerBlock);
    
    for (int p = 0; p <= MAX_BLOCk; p++) {
        for (int t = 0; t <= prop.maxThreadsPerBlock; t++) {
            start = seconds();
            add4<<<p, t>>>(N, _x, _y);
            cudaDeviceSynchronize();
            end =  seconds();
            diff = (end - start);
            if (diff == MIN_TIME) {
                printf("Time %f - Block %d - Thread %d \n", diff, p, t);
            }
            if (diff < MIN_TIME) {
                MIN_TIME = diff;
                BEST_BLOCK = p;
                BEST_THREAD = t;
            }

        }
    }

    printf("Best Time %f - Block %d - Thread %d \n", MIN_TIME, BEST_BLOCK, BEST_THREAD);
    // Free memory
    cudaFree(_x);
    cudaFree(_y);

    return 0;
}