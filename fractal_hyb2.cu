/*
Fractal code for CS 4380 / CS 5351

Copyright (c) 2016, Texas State University. All rights reserved.

Redistribution in source or binary form, with or without modification,
is not permitted. Use in source and binary forms, with or without
modification, is only permitted for academic use in CS 4380 or CS 5351
at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
editor: Diego Loya
*/

#include <cstdlib>
#include <sys/time.h>
#include <cuda.h>
#include "cs43805351.h"

static const int ThreadsPerBlock = 512;

static const double Delta = 0.005491;
static const double xMid = 0.745796;
static const double yMid = 0.105089;

static __global__
void FractalKernel(const int from_frame, const int to_frame, const int width, unsigned char pic_d[])
{
  // kernel code goes in here; use the same parallelization approach as in the previous project{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
   if ((from_frame * width * width <= idx) &&(idx < to_frame * width * width)){ 
    const int col = idx % width;
    const int row = (idx / width) % width;
    const int frame = idx / (width * width);

    const double delta = Delta * pow(0.99,frame+1);          //removes loop-carried dependency
    //We do not need a loop since each thread runs only one pixel
    const double xMin = xMid - delta;
    const double yMin = yMid - delta;
    const double dw = 2.0 * delta / width;
    const double cx = -xMin - col * dw;
    const double cy = -yMin - row * dw;
    double x = cx;
    double y = cy;
    int depth = 256;
    double x2, y2;
    do {
       x2 = x * x;
       y2 = y * y;
       y = 2 * x * y + cy;
       x = x2 - y2 + cx;
       depth--;
    }while ((depth > 0) && ((x2 + y2) < 5.0));
    pic_d[idx] = (unsigned char)depth;		//**************** pic->pic_d
  }
}
unsigned char* GPU_Init(const int size)
{
  // allocate pic_d array on GPU and return pointer to it
  unsigned char* pic_d;
  cudaMalloc((void **)&pic_d, size);
  return pic_d;

}

void GPU_Exec(const int from_frame, const int to_frame, const int width, unsigned char pic_d[])
{
  // call the kernel (and do nothing else)
  FractalKernel<<<(((width*width*(to_frame-from_frame))+(ThreadsPerBlock-1))/ThreadsPerBlock),ThreadsPerBlock>>>(from_frame,to_frame,width,pic_d);
}

void GPU_Fini(const int size, unsigned char pic[], unsigned char pic_d[])
{
  // copy the pixel data to the CPU and deallocate the GPU array
  cudaMemcpy(pic, pic_d, size, cudaMemcpyDeviceToHost);
  cudaFree(pic_d);
}
