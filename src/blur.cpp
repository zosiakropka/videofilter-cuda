#include "../headers/blur.h"

const int BlurFilter::MASK[BlurFilter::MASK_SIZE * BlurFilter::MASK_SIZE] = {
    0, 1, 2, 1, 0,
    1, 4, 8, 4, 1,
    2, 8, 16, 8, 2,
    1, 4, 8, 4, 1,
    0, 1, 2, 1, 0,
};
