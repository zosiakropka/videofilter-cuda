/* 
 * File:   mask.h
 * Author: Zosia Sobocinska
 *
 * Created on December 30, 2013, 4:23 PM
 */

#ifndef MASK_FILTER_H
#define	MASK_FILTER_H

#include "filter.h"
#include "utils.h"

class MaskFilter : public VideoFilter {
public:

  void process(uchar const* bytes_in, uchar* bytes_out,
          uint cols, uint rows, uint channels, uint step_in, uint step_out);

  struct Mask {
    uint center;

    uint total;

    int *weigths;

    int size() {
      #define mask_size(center) (2 * center + 1)
      return mask_size(this->center);

    }

    int weight(int x, int y) {
      #define mask_weight(weigths, x, y, center) (weigths[(y+center) * mask_size(center) + (x+center)])
      x += this->center;
      y += this->center;
      return this->weigths[y * this->size() + x];
    }

  };

  void alloc_buffers();
  void free_buffers();

  void init(int cols_in, int rows_in, int step_in, int step_out, uint channels) {
    this->cols_in = cols_in;
    this->rows_in = rows_in;
    this->step_in = step_in;
    this->step_out = step_out;
    this->channels = channels;
  }


protected:

  Mask mask;
  #if CUDA
  int *mask_weigths_dev;
  #endif

};

#endif	/* MASK_FILTER_H */

