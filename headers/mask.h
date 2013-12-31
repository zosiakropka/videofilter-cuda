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
protected:

    struct Mask {
        uint center;

        uint total;

        int const* weigths;

        int size() {
            return 2 * center + 1;
        }

        int weight(int x, int y) {
            x += this->center;
            y += this->center;
            return this->weigths[y * this->size() + x];
        }
    };

    Mask mask;
public:

    void process(uchar const* bytes_in, uchar* bytes_out,
            uint cols_o, uint rows_o, uint channels, uint step_in, uint step_out);
};

#endif	/* MASK_FILTER_H */

