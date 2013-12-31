#include "../headers/sharpen.h"

const int SharpenFilter::MASK[SharpenFilter::MASK_SIZE * SharpenFilter::MASK_SIZE] = {
    -1, -1, -1,
    -1, 9, -1,
    -1, -1, -1,
};
