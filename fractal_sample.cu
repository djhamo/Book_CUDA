#include "common/book.h"
#include "common/cpu_bitmap.h"
#include <stdio.h>
#include <curand_kernel.h>
#include <time.h>

#define DIM 1024

struct cuComplex {
    float r;
    float i;
    __device__ cuComplex (float a, float b) : r(a), i(b) {}
    __device__ float magnitude2 (void) { 
        return r * r + i * i; 
    }
    __device__ cuComplex operator* (const cuComplex a) { 
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __device__ cuComplex operator+ (const cuComplex a) {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int julia( int x, int y, float SCALE) {
    float jx = SCALE * (float) (DIM/2 - x)/(DIM /2);
    float jy = SCALE * (float) (DIM/2 - y)/(DIM /2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    for (int i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__ void init(unsigned int seed, curandState_t* states) {

    curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

__global__ void kernel(unsigned char *ptr, float scale, curandState_t* states) {

    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    int r;
    int juliaValue = julia(x, y, scale);
    r = curand(&states[0]) % 256;
    ptr[offset*4 + 0] = r * juliaValue;
    r = curand(&states[0]) % 256;
    ptr[offset*4 + 1] = r * juliaValue;
    r = curand(&states[0]) % 256;
    ptr[offset*4 + 2] = r * juliaValue;
    ptr[offset*4 + 3] = 255;

}

int main(int argc, char* argv[]) {

    curandState_t* states;
    HANDLE_ERROR( cudaMallocManaged(&states, sizeof(curandState_t)) );
    init<<<1, 1>>>(time(NULL), states);

    float scale = 1.5f;
    for ( int i = 1; i < argc; i++ ) {
        if (!strcmp(argv[i], "-s"))
            scale = atof(argv[++i]);
    }

    CPUBitmap bitmap (DIM, DIM);

    unsigned char *dev_bitmap;

    HANDLE_ERROR( cudaMallocManaged(&dev_bitmap, bitmap.image_size()) );

    dim3 grid(DIM, DIM, 1);

    kernel<<<grid, 1>>>(dev_bitmap, scale, states);

    HANDLE_ERROR( cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost) );

    bitmap.display_and_exit();

    HANDLE_ERROR( cudaFree(states) );
    HANDLE_ERROR( cudaFree(dev_bitmap) );
    
}