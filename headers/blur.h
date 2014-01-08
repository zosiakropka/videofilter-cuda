/* 
 * File:   blur.h
 * Author: Zosia Sobocinska
 *
 * Created on December 28, 2013, 3:33 PM
 */

#ifndef BLUR_H
#define	BLUR_H

#include "mask.h"

class BlurFilter : public MaskFilter {
private:

  const static uint MASK_CENTER = 2;

  const static uint MASK_SIZE = MASK_CENTER * 2 + 1;

  const static uint MASK_TOTAL = 76;

  static int MASK[MASK_SIZE*MASK_SIZE];

public:

  BlurFilter() : MaskFilter() {
    Mask* mask = &this->mask;
    mask->center = MASK_CENTER;
    mask->total = MASK_TOTAL;
    mask->weigths = MASK;

    strcpy(name, "blur");
  }

  void process(uchar const* bytes_in, uchar* bytes_out,
          uint cols, uint rows, uint channels, uint step_in, uint step_out) {

    MaskFilter::process(bytes_in, bytes_out, cols, rows, channels, step_in, step_out);
  }

};

#endif	/* BLUR_H */

