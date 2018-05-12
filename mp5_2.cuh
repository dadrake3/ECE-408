// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this
#define SCAN 1024

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host

  __shared__ float temp[2 * BLOCK_SIZE];

  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int bd = blockDim.x;

  int i = (bx*bd) + tx;

  if(i < len){
    temp[tx] = input[i];
  }

  unsigned int s;
  for(s = 1; s <= BLOCK_SIZE; s = s * 2){
    __syncthreads();
    unsigned int j = (2 * s * (1 + tx)) - 1;
    if(j < len && j < SCAN){
      temp[j] = temp[j] + temp[j - s];
    }

    //printf(s)
  }

  for(s = BLOCK_SIZE / 2; s > 0; s = s / 2){
    __syncthreads();
    unsigned int j = (2 * s * (1 + tx)) - 1;
    if(j + s < len && j + s < SCAN){
      temp[j + s] = temp[j + s] + temp[j];
    }
    //printf(s)
  }
  __syncthreads();

  if(i < SCAN && i < len){
   // output[i] = output[i] + temp[tx];
    output[i] = temp[tx];

  }

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int gd = ceil(numElements/(BLOCK_SIZE * 1.0));
  dim3 grid(gd, 1, 1);
  dim3 block(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  for(int j = 0; j < gd; j++) {
        int t = j * BLOCK_SIZE;
        if(j != 0) {
                float temp = 0;
                cudaMemcpy(&temp, &deviceOutput[t - 1], sizeof(float), cudaMemcpyDeviceToHost);
                temp += hostInput[t];
                cudaMemcpy(&deviceInput[t], &temp, sizeof(float),cudaMemcpyHostToDevice);
        }
        scan<<<grid, block>>>(&deviceInput[t], &deviceOutput[t], numElements);
  }

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
