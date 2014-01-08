#include "../headers/resize.h"
#include "../headers/gpu.h"

__device__ double get_distance(int in, int out, double ratio) {
  return fabs(((double) in) * ratio - (((double) out)));
}

__global__ void resize_kernel(uchar *bytes_in, uchar *bytes_out,
        uint cols_in, uint rows_in, uint cols_out, uint rows_out, uint step_in, uint step_out,
        double ratio,
        const uint BLOCK_SIZE) {

  uint col_o = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  uint row_o = blockIdx.y * BLOCK_SIZE + threadIdx.y;


  if (col_o < cols_out && row_o < rows_out) {

    const int NEIGHBOURHOOD = 2; // hardcode neighbourhood
    const uint CHANNELS = 3; // just assume there are 3 channels

    double f_x[NEIGHBOURHOOD + 1][CHANNELS];
    uint id_in[NEIGHBOURHOOD + 1];

    uint id_o = array_index(col_o, row_o, CHANNELS, step_out);

    int col_i = floor(((double) col_o) / ratio);
    int row_i = floor(((double) row_o) / ratio);

    memset(f_x, 0.0, (NEIGHBOURHOOD + 1) * CHANNELS * sizeof (double));

    double d_x = get_distance(col_i, col_o, ratio);
    double d_y = get_distance(row_i, row_o, ratio);

    for (int y = 0; y <= NEIGHBOURHOOD; y++) {
      for (uint ch = 0; ch < CHANNELS; ch++) {
        for (int x = 0; x <= NEIGHBOURHOOD; x++) {
          int sub_col = col_i + x - 1;
          int sub_row = row_i + y - 1;
          sub_col = (sub_col < 0) ? 0 : sub_col;
          sub_col = (sub_col > cols_in) ? cols_in : sub_col;
          sub_row = (sub_row < 0) ? 0 : sub_row;
          sub_row = (sub_row > rows_in) ? rows_in : sub_row;
          id_in[x] = array_index(sub_col, sub_row, CHANNELS, step_in) + ch;
        }
        f_x[y][ch] = double (bytes_in[id_in[1]]) +
                double (bytes_in[id_in[2]] - bytes_in[id_in[0]])
                * d_x
                + double (bytes_in[id_in[0]] - 2 * bytes_in[id_in[1]] + bytes_in[id_in[2]])
                * d_x*d_x;
      }
    }
    for (uint ch = 0; ch < CHANNELS; ch++) {
      bytes_out[id_o + ch] = (f_x[1][ch] + (f_x[2][ch] - f_x[0][ch]) * d_y
              + (f_x[0][ch] - 2 * f_x[1][ch] + f_x[2][ch]) * d_y * d_y);
    }
  }
}

double ResizeFilter::get_distance(int in, int out) {
  return fabs(((double) in) * RATIO - (((double) out)));
}

void ResizeFilter::process(uchar const* bytes_in_host, uchar* bytes_out_host,
        uint cols_in, uint rows_in, uint channels, uint step_in, uint step_out) {

  const uint BLOCK_SIZE = dev::get_block_size(threads, cols_in * rows_in);

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(((cols_out() + BLOCK_SIZE) / BLOCK_SIZE), (rows_out() + BLOCK_SIZE) / BLOCK_SIZE);

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

  resize_kernel << < grid, block >> >(bytes_in_dev, bytes_out_dev, cols_in, rows_in, cols_out(), rows_out(), step_in, step_out, RATIO, BLOCK_SIZE);

  dev::test(cudaGetLastError());

  dev::test(cudaEventRecord(stop, NULL)); // stop time measurement
  dev::test(cudaEventSynchronize(stop));

  dev::test(cudaEventElapsedTime(&time, start, stop));

  dev::test(cudaEventDestroy(start));
  dev::test(cudaEventDestroy(stop));

  dev::cuda_dev2host(bytes_out_dev, bytes_out_host, size_out());

}
