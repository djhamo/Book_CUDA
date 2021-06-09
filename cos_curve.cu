#include "cuda.h"
#include "common/book.h"
#include "common/cpu_bitmap.h"
#include "common/cpu_anim.h"

#define DIM 1024
#define PI 3.1415926535897932f

struct DataBloc {
    unsigned char *dev_bitmap;
    CPUAnimBitmap *bitmap;
};

void cleanup(DataBloc *d) {
    cudaFree(d->dev_bitmap);
}

__global__ void kernel(unsigned char *ptr, int ticks) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = (x * blockDim.x * gridDim.x) + y;

    float ox = (x - DIM/2);
    float oy = (y );
    float raio = 120.0f;
    //float dis = 120.0;

    if ((ox / raio) <= (sin((oy / raio) + (ticks/16.0f)) )) {
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


void generate_frame(DataBloc *d, int ticks) {

    dim3 blocks(DIM/16, DIM/16);
    dim3 threads(16, 16);

    kernel<<<blocks, threads>>>(d->dev_bitmap, ticks);

    HANDLE_ERROR( cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost) );
}

int main( void ) {
    cudaEvent_t start, stop;

    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start, 0) );
/*
    CPUBitmap bitmap(DIM, DIM);

    unsigned char *dev_bitmap;

    HANDLE_ERROR( cudaMallocManaged(&dev_bitmap, bitmap.image_size()) );
    
    dim3 grids(DIM/16, DIM/16);
    dim3 threads(16,16);

    kernel<<<grids, threads>>>(dev_bitmap);
*/

    DataBloc data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;

    HANDLE_ERROR( cudaMallocManaged( &data.dev_bitmap, bitmap.image_size()) );

    bitmap.anim_and_exit( (void (*) (void*, int))generate_frame, (void (*)(void*))cleanup); 

	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	
	float elapsedTime;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
	printf( "Time to generate: %3.1f ms\n", elapsedTime);
	
	HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );
    /*
    HANDLE_ERROR( cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost) );

    bitmap.display_and_exit();

    cudaFree(dev_bitmap);
    */
}

