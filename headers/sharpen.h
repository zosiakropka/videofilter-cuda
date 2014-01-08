/* 
 * File:   blur.h
 * Author: Zosia Sobocinska
 *
 * Created on December 28, 2013, 3:33 PM
 */

#ifndef SHARPEN_FILTER_H
#define	SHARPEN_FILTER_H

#include "filter.h"
#include "mask.h"

class SharpenFilter : public MaskFilter {
  const static uint MASK_CENTER = 1;

  const static uint MASK_SIZE = MASK_CENTER * 2 + 1;

  const static uint MASK_TOTAL = 1;

  static int MASK[MASK_SIZE*MASK_SIZE];

public:

  SharpenFilter() : MaskFilter() {
    Mask* mask = &this->mask;
    mask->center = MASK_CENTER;
    mask->total = MASK_TOTAL;
    mask->weigths = MASK;

    strcpy(name, "sharpen");
  }

  void process(uchar const* bytes_in, uchar* bytes_out,
          uint cols, uint rows, uint channels, uint step_in, uint step_out) {
    
    MaskFilter::process(bytes_in, bytes_out, cols, rows, channels, step_in, step_out);
  }

};

#endif	/* SHARPEN_FILTER_H */
