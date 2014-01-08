/* 
 * File:   cuda_props.h
 * Author: Zosia
 *
 * Created on January 1, 2014, 4:05 AM
 */

#ifndef GPU_H
#define	GPU_H

#include <driver_types.h>
#include <iostream>
#include <cstdlib>
#include <math.h>

#include "filter.h"

#ifndef __device__
#define __device__
#endif

typedef unsigned char uchar;

namespace host {
  void alloc_mem_uchar(uchar** array, uint count);
  void alloc_mem_int(int** array, uint count);
  void free_mem_uchar(uchar** array);
  void free_mem_int(int** array);
  void cuda_host2dev(const void *host_array, void *dev_array, uint count, size_t size = sizeof (uchar));
}
namespace dev {
  void alloc_mem_uchar(uchar** array, uint count);
  void alloc_mem_int(int** array, uint count);
  void free_mem_uchar(uchar** array);
  void free_mem_int(int** array);
  void cuda_dev2host(void* dev_array, void* host_array, uint count, size_t size = sizeof (uchar));
  void test(cudaError_t result);
  cudaDeviceProp get_prop();
  uint get_max_block_size();
  uint get_block_size(uint threads, uint problem);
}

#endif	/* GPU_H */

