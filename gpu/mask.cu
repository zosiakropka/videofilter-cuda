#include "../headers/mask.h"

void MaskFilter::alloc_buffers() {
  VideoFilter::alloc_buffers();
  dev::alloc_mem_int(&mask_weigths_dev, mask.size() * mask.size());
  host::cuda_host2dev(mask.weigths, mask_weigths_dev, mask.size() * mask.size(), sizeof (int));
}

void MaskFilter::free_buffers() {
  VideoFilter::free_buffers();
  dev::free_mem_int(&mask_weigths_dev);
}

__global__ void mask_kernel(uchar *bytes_in, uchar *bytes_out,
        uint cols, uint rows, uint channels, uint step,
        uint size,
        int *mask_weigths, int mask_center, int mask_total,
        const uint BLOCK_SIZE) {

  uint row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  uint col = ((blockIdx.x / channels) * BLOCK_SIZE) + threadIdx.x;
  uint ch = blockIdx.x % channels;

  if (row < rows && col < cols) {
    uint index = array_index(col, row, channels, step);

    int val = 0;
    int subcol, subrow;
    for (int x = (int) -mask_center; x <= (int) mask_center; x++) {
      for (int y = (int) -mask_center; y <= (int) mask_center; y++) {
        subcol = col + x;
        subrow = row + y;
        subcol = (subcol > cols) ? (cols - 1) : subcol;
        subrow = (subrow > rows) ? (rows - 1) : subrow;
        uint subindex = array_index(subcol, subrow, channels, step);
        val += mask_weight(mask_weigths, x, y, mask_center) * bytes_in[subindex + ch];
        //        val += mask_weight(mask_weigths, x, y, mask_center) * 256;
      }
    }
    bytes_out[index + ch] = (uchar) ((double) val / (double) mask_total);
  }
}

void MaskFilter::process(uchar const* bytes_in_host, uchar* bytes_out_host,
        uint cols, uint rows, uint channels, uint step_in, uint step_out) {

  memcpy(bytes_out_host, bytes_in_host, size_in() * sizeof (uchar));

  uint step = step_in;
  uint size = size_in();

  const uint BLOCK_SIZE = dev::get_block_size(threads, cols_in * rows_in);

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(((cols * channels + BLOCK_SIZE) / BLOCK_SIZE), (rows + BLOCK_SIZE) / BLOCK_SIZE);

  uchar *bytes_in_dev, *bytes_out_dev;
  bytes_in_dev = buffers[0];
  bytes_out_dev = buffers[1];

  host::cuda_host2dev(bytes_in_host, bytes_in_dev, size_in());
  host::cuda_host2dev(bytes_out_host, bytes_out_dev, size_out());

  float time = 1.0;
  cudaEvent_t start, stop;
  dev::test(cudaEventCreate(&start));
  dev::test(cudaEventCreate(&stop));

  dev::test(cudaEventRecord(start, NULL)); // stop time measurement

  mask_kernel << < grid, block >> >(bytes_in_dev, bytes_out_dev, cols, rows, channels, step, size, mask_weigths_dev, mask.center, mask.total, BLOCK_SIZE);

  dev::test(cudaGetLastError());

  dev::test(cudaEventRecord(stop, NULL)); // stop time measurement
  dev::test(cudaEventSynchronize(stop));

  dev::test(cudaEventElapsedTime(&time, start, stop));

  dev::test(cudaEventDestroy(start));
  dev::test(cudaEventDestroy(stop));

  dev::cuda_dev2host(bytes_out_dev, bytes_out_host, size_out());
}
