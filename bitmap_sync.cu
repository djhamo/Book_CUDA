#include "cuda.h"
#include "common/book.h"
#include "common/cpu_bitmap.h"

#define DIM 1024
#define PI 3.1415926535897932f
#define THREAD_DIM 16

__global__ void kernel(unsigned char *ptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y *blockDim.x * gridDim.x;

    __shared__ float shared[THREAD_DIM][THREAD_DIM];

    const float period = 128.0f;

    shared[threadIdx.x][threadIdx.y] = 
        255 * (sinf(x * 2.0f * PI / period) + 1.0f) *
              (sinf(y * 2.0f * PI / period) + 1.0f) / 4.0f;

    __syncthreads();
              
    ptr[offset*4 + 0] = 0;          
    ptr[offset*4 + 1] = shared[15 - threadIdx.x][15 - threadIdx.y];          
    ptr[offset*4 + 2] = 0;          
    ptr[offset*4 + 3] = 255;          
}

int main(void) {
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;

    HANDLE_ERROR( cudaMallocManaged(&dev_bitmap, bitmap.image_size()) );

    dim3 grids(DIM/THREAD_DIM, DIM/THREAD_DIM);
    dim3 threads(THREAD_DIM, THREAD_DIM);

    kernel<<<grids, threads>>>(dev_bitmap);

    HANDLE_ERROR( cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost) );

    bitmap.display_and_exit();
    
    cudaFree(dev_bitmap);
    
}