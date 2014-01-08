#include "../headers/tiltshift.h"
#include "../headers/gpu.h"

void TiltshiftFilter::alloc_buffers() {
  #if CUDA
  VideoFilter::alloc_buffers();
  blur_filter.init(cols_in, rows_in, step_in, step_out, channels);
  blur_filter.alloc_buffers();
  #endif
}

void TiltshiftFilter::free_buffers() {
  #if CUDA
  VideoFilter::free_buffers();
  blur_filter.free_buffers();
  #endif
}

__global__ void tiltshift_kernel(uchar *bytes_in, uchar *bytes_out,
        uint cols, uint rows, uint channels, uint step,
        const uint BLOCK_SIZE) {

  uint row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  uint col = (blockIdx.x) / channels * BLOCK_SIZE + threadIdx.x;
  uint ch = blockIdx.x % channels;

  if (row < rows && col < cols) {

    float opacity = 1.0 - fabs(1.0 - (float) row / (float) (rows / 2));
    opacity = (opacity > 1.0) ? 1.0 : opacity;
    opacity *= opacity;
    opacity *= opacity;

    float opacity_alt = 1.0 - opacity;
    uint index = array_index(col, row, channels, step);

    uchar val = (uchar) (opacity * (float) bytes_in[index + ch] + opacity_alt * (float) bytes_out[index + ch]);
    val = (val > 0) ? val : 0;

    bytes_out[index + ch] = val;
  }
}

void TiltshiftFilter::process(uchar const* bytes_in_host, uchar* bytes_out_host,
        uint cols, uint rows, uint channels, uint step_in, uint step_out) {

  blur_filter.process(bytes_in_host, bytes_out_host, cols, rows, channels, step_in, step_out);
  blur_filter.process(bytes_out_host, bytes_out_host, cols, rows, channels, step_in, step_out);

  uint step = step_in;
  uint size = size_in();

  const uint BLOCK_SIZE = dev::get_block_size(threads, cols_in * rows_in);

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(((cols * channels + BLOCK_SIZE) / BLOCK_SIZE), (rows + BLOCK_SIZE) / BLOCK_SIZE);

  uchar *bytes_in_dev, *bytes_out_dev;
  bytes_in_dev = buffers[0];
  bytes_out_dev = buffers[1];

  host::cuda_host2dev(bytes_in_host, bytes_in_dev, size);
  host::cuda_host2dev(bytes_out_host, bytes_out_dev, size);

  float time = 1.0;
  cudaEvent_t start, stop;
  dev::test(cudaEventCreate(&start));
  dev::test(cudaEventCreate(&stop));

  dev::test(cudaEventRecord(start, NULL)); // stop time measurement

  tiltshift_kernel << < grid, block >> >(bytes_in_dev, bytes_out_dev, cols, rows, channels, step, BLOCK_SIZE);
  dev::test(cudaGetLastError());

  dev::test(cudaEventRecord(stop, NULL)); // stop time measurement
  dev::test(cudaEventSynchronize(stop));
  dev::test(cudaEventElapsedTime(&time, start, stop));

  dev::test(cudaEventDestroy(start));
  dev::test(cudaEventDestroy(stop));

  dev::cuda_dev2host(bytes_out_dev, bytes_out_host, size);
}
