#include "../headers/gpu.h"

namespace host {

  void alloc_mem_uchar(uchar** array, uint count) {
    (*array) = (uchar*) malloc(count * sizeof (uchar));
    memset((*array), 0, count * sizeof (uchar));
  }

  void alloc_mem_int(int** array, uint count) {
    (*array) = (int*) malloc(count * sizeof (int));
    memset((*array), 0, count * sizeof (int));
  }

  void free_mem_uchar(uchar** array) {
    free((*array));
    (*array) = NULL;
  }

  void free_mem_int(int** array) {
    free((*array));
    (*array) = NULL;
  }

  void cuda_host2dev(const void *host_array, void *dev_array, uint count, size_t size) {
    cudaMemcpy(dev_array, host_array, (count * size), cudaMemcpyHostToDevice);
  }

}
namespace dev {

  void alloc_mem_uchar(uchar** array, uint count) {
    test(cudaMalloc(array, count * sizeof (uchar)));
  }

  void alloc_mem_int(int** array, uint count) {
    test(cudaMalloc(array, count * sizeof (int)));
  }

  void free_mem_uchar(uchar** array) {
    test(cudaFree((*array)));
    (*array) = NULL;
  }

  void free_mem_int(int** array) {
    test(cudaFree((*array)));
    (*array) = NULL;
  }

  void cuda_dev2host(void* dev_array, void* host_array, uint count, size_t size) {
    test(cudaMemcpy(host_array, dev_array, (count * size), cudaMemcpyDeviceToHost));
  }

  void test(cudaError_t result) {
    if (result != cudaSuccess) {
      std::cerr << "Error: " << cudaGetErrorString(result) << std::endl;
      exit(1);
    }
  }

  cudaDeviceProp get_prop() {
    cudaDeviceProp properties;
    test(cudaGetDeviceProperties(&properties, 0));
    return properties;
  }

  uint get_max_block_size() {
    return sqrt(get_prop().maxThreadsPerBlock);
  }

  uint get_block_size(uint threads, uint problem) {

    uint max_size = dev::get_max_block_size();
    uint size = sqrt(problem / threads);

    if (size * size > max_size) {
      std::cerr << "Not enough threads." << std::endl;
      return max_size;
    } else if (!size) {
      std::cerr << "Too many threads." << std::endl;
      return max_size;
    }
    return size;
  }

}