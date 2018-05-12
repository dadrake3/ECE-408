#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH   8
#define TILE_WIDTH_F   (TILE_WIDTH * 1.0)
#define MASK_WIDTH   3
#define BLOCK_WIDTH  (TILE_WIDTH + MASK_WIDTH - 1)
//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[MASK_WIDTH * MASK_WIDTH * MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {

  //@@ Insert kernel code here
  __shared__ float block_mem[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];

  //easy access to thread coords
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  //easy access to block coords
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;

  //easy access to overall matrix coords
  int mx = bx * TILE_WIDTH + tx;
  int my = by * TILE_WIDTH + ty;
  int mz = bz * TILE_WIDTH + tz;

  //since every block is larger by MASK_WIDTH from the actual tile this will give the approriate offset indicies in the matrix
  int x_offset = mx - (MASK_WIDTH / 2);
  int y_offset = my - (MASK_WIDTH / 2);
  int z_offset = mz - (MASK_WIDTH / 2);

  //check if values fall inside the range of the matrix bounds, if so load into shared mem
  if (
    (x_offset < x_size) && (x_offset >= 0) &&
    (y_offset < y_size) && (y_offset >= 0) &&
    (z_offset < z_size) && (z_offset >= 0)
  ){
    block_mem[tz][ty][tx] = input[(z_offset * (y_size * x_size)) + (y_offset * x_size) + x_offset];
  }
  //if not load 0 into shared mem
  else{
    block_mem[tz][ty][tx] = 0.0;
  }
  __syncthreads();

  float conv_val = 0.0;
  //check if thread index is within tile
    if(
      (tx < TILE_WIDTH) && (tx >= 0) &&
      (ty < TILE_WIDTH) && (ty >= 0) &&
      (tz < TILE_WIDTH) && (tz >= 0)
    ){
      //iterate across the kernel mask positions and calc appropriate indicies
      for(int zz = 0; zz < MASK_WIDTH; zz++){
        for(int yy = 0; yy < MASK_WIDTH; yy++){
          for(int xx = 0; xx < MASK_WIDTH; xx++){
            //calculate the convolution
            float temp = deviceKernel[(zz * (MASK_WIDTH * MASK_WIDTH)) + (yy * MASK_WIDTH) + xx];
            temp *= block_mem[tz + zz][ty + yy][tx + xx];
            conv_val += temp;
          }
        }
      }
      //make sure the calculated value corresponds to an actualy matrix value
      if(
       (mz < z_size) && (mz >= 0) &&
       (my < y_size) && (my >= 0) &&
       (mx < x_size) && (mx >= 0)
      )
        output[(mz * (y_size * x_size)) +(my * x_size) + mx] = conv_val;
    }
    __syncthreads();




}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**) &deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void**) &deviceOutput, (inputLength - 3) * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, &hostInput[3], (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");


  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 grid(ceil( x_size /TILE_WIDTH_F), ceil( y_size / TILE_WIDTH_F), ceil(z_size / TILE_WIDTH_F));
  dim3 block(BLOCK_WIDTH,BLOCK_WIDTH, BLOCK_WIDTH);
  //@@ Launch the GPU kernel here
  conv3d<<<grid, block>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)

  wbTime_stop(Copy, "Copying data from the GPU");
  cudaMemcpy(&hostOutput[3], deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
