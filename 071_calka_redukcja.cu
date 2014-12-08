 
#include <stdio.h>
#include <cuda.h>
 
 
// Kernel wykonywane na "CUDA device"

__host__ __device__ float f(float x){
  return exp(x*x)*cos(x);
}


__global__ void oblicz_fx(float h, float a, float *w)
{

  // rozmiar bloku równy 64

  __shared__ float y[64]; 

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  y[threadIdx.x] = f(a+(i+1)*h);
  
  __syncthreads();
  
  
  if(threadIdx.x%2==0) y[threadIdx.x]+=y[threadIdx.x+1];
  __syncthreads();

  if(threadIdx.x%4==0) y[threadIdx.x]+=y[threadIdx.x+2];
  __syncthreads();

  if(threadIdx.x%8==0) y[threadIdx.x]+=y[threadIdx.x+4];
  __syncthreads();

  if(threadIdx.x%16==0) y[threadIdx.x]+=y[threadIdx.x+8];
  __syncthreads();

  if(threadIdx.x%32==0) y[threadIdx.x]+=y[threadIdx.x+16];
  __syncthreads();

  if(threadIdx.x==0) {
  
    y[threadIdx.x]+=y[threadIdx.x+32];
    w[blockIdx.x]=y[threadIdx.x];

  }

/*
  if(threadIdx.x==0){

     float x=0;
     for(int i=0;i<blockDim.x;i++)
        x+=y[i];

     w[blockIdx.x]=x;

  }
*/
} 
 
// program wykonywany na "host computer"
int main(void)
{
  float *w_h, *w_d;  // wskazniki do tablic na host i device 
  const int N = 4*65536+1;  // liczba elementow tablicy
  
  float a=0.0;
  float b=1.0;
  float h=(b-a)/N;
  
  int bsize=64;
  int gsize=(N-1)/bsize;
  
  size_t size = gsize * sizeof(float);
  w_h = (float *)malloc(size);        // alokacja tablicy na host
  cudaMalloc((void **) &w_d, size);   // alokacja tablicy na device
  

  // wykonanie obliczen na device

  oblicz_fx <<< gsize, bsize >>> (h,a,w_d);

  // skopiowanie wynikow z pamieci karty do pomieci hosta
  cudaMemcpy(w_h, w_d, sizeof(float)*gsize, cudaMemcpyDeviceToHost);

  // Print results
  
  float sum=0.5*(f(a)+f(b));
  
  for (int i=0; i<gsize; i++) 
     sum+=w_h[i];
  
  sum*=h;
  
  printf("calka=%f\n", sum);

  // zwolnienie pamieci
  free(w_h); 
  cudaFree(w_d);
}


