#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

//#include "cublas.h"
//#include "magma.h"
//#include "magmablas.h"

#define NN 4096


void host_mv(int n,float *a,int lda,float *x,float *y){

  int i,j;
  for(j=0;j<n;j++){
    for(i=0;i<n;i++){
      y[i]+=a[i+j*lda]*x[j];
    }
  }
}

__global__ void cuda_mv(int n,float *a,int lda,float *x,float *y){

  int j;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  float t = y[idx];
  
  for(j=0;j<n;j++){
    t+=a[idx+j*lda]*x[j];
  }
  
  y[idx]=t;
}

__global__ void cuda_xmv(int n,float *a,int lda,float *x,float *y){

  __shared__ float xc[128];
  __shared__ float yc[128];

  int j,k;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;

  yc[threadIdx.x] = y[idx];
  
  for(k=0;k<n/blockDim.x;k++) {
  
     xc[threadIdx.x]=x[k*blockDim.x+threadIdx.x];
     __syncthreads();
  
    for(j=0;j<blockDim.x;j++){
       yc[threadIdx.x]+=a[(j+k*blockDim.x)*lda+idx]*xc[j];
    }

    __syncthreads();
  }
  y[idx]=yc[threadIdx.x];
}


/* Main */
int main(int argc, char** argv)
{    
    float *h_a;
    float *h_b;
    float *h_c;

    float *d_a;
    float *d_b;
    float *d_c;


    int i,j,n;

	struct timeval t1, t2;
	double time;

    int devCount;
    cudaGetDeviceCount(&devCount);
    
    printf("Liczba GPU= %d\n",devCount);
    

    //cudaSetDevice(1);

    //scanf("%d",&n);
	n = NN;
    
    // alokacja tablic w pamięci hosta

    h_a = (float*)malloc(n*n * sizeof(*h_a));
    h_b = (float*)malloc(n * sizeof(*h_b));
    h_c = (float*)malloc(n * sizeof(*h_c));


    // inicjowanie tablic
    
    for (i = 0; i < n*n; i++) {
        h_a[i] = 2; // rand() / (float)RAND_MAX;
    
    }

    for (i = 0; i < n; i++) {
        h_b[i] = 2; // rand() / (float)RAND_MAX;
        h_c[i] = 2; // rand() / (float)RAND_MAX;
    
    }


    // alokacja tablic w pamięci karty


    cudaMalloc((void **) &d_a, n*n*sizeof(*d_a));
    cudaMalloc((void **) &d_b, n*sizeof(*d_b));
    cudaMalloc((void **) &d_c, n*sizeof(*d_c));

    // kopiowanie danych do pamięci karty


    cudaMemcpy(d_a, h_a, n*n*sizeof(*d_a), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n*sizeof(*d_b), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, n*sizeof(*d_c), cudaMemcpyHostToDevice);

	gettimeofday(&t1, 0); // start

    cuda_xmv<<<n/128,128>>>(n,d_a,n,d_b,d_c);

//    cuda_mv<<<n/128,128>>>(n,d_a,n,d_b,d_c);

    cudaThreadSynchronize();


//    host_mv(n,h_a,n,h_b,h_c);

	gettimeofday(&t2, 0); // stop
	time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
	//printf("Czas = %f\n",time);
	

    cudaMemcpy(h_c, d_c, sizeof(float)*n, cudaMemcpyDeviceToHost);
    

/*
    for(i=0;i<n;i++){
      printf("%f\n",h_c[i]);
    }

*/
    // zwolnienie pamięci
    
    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);



    printf("N=%d Czas=%le MFlops=%lf\n",n,
                            time,2.0*n*n/(1000000.0*time));


    return 0;
}
