#include "cuda.h"
#include "common/book.h"
#include "common/cpu_bitmap.h"

#define INF 2e10f
#define rnd( x ) (x * rand() / RAND_MAX)
#define SPHERES 35
#define DIM 1024
#define PI 3.1415926535897932f

__global__ void kernel(unsigned char *ptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = (x * blockDim.x * gridDim.x) + y;

    float ox = (x - DIM/2);
    float oy = (y - DIM/2);
    float raio = 200;

    if ((ox * ox) + (oy * oy) < (raio * raio)) {
        ptr[offset*4 + 0] = 0;
        ptr[offset*4 + 1] = 0;
        ptr[offset*4 + 2] = 255;
        ptr[offset*4 + 3] = 0;
    } else {
        ptr[offset*4 + 0] = 0;
        ptr[offset*4 + 1] = 0;
        ptr[offset*4 + 2] = 0;
        ptr[offset*4 + 3] = 255;
    }

}

int main( void ) {
    cudaEvent_t start, stop;

    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start, 0) );

    CPUBitmap bitmap(DIM, DIM);

    unsigned char *dev_bitmap;

    HANDLE_ERROR( cudaMallocManaged(&dev_bitmap, bitmap.image_size()) );
    
    dim3 grids(DIM/16, DIM/16);
    dim3 threads(16,16);

    kernel<<<grids, threads>>>(dev_bitmap);

	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	
	float elapsedTime;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
	printf( "Time to generate: %3.1f ms\n", elapsedTime);
	
	HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );
    
    HANDLE_ERROR( cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost) );

    bitmap.display_and_exit();

    cudaFree(dev_bitmap);
}

