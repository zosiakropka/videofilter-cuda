#include "../headers/resize.h"

void ResizeFilter::process(uchar const* bytes_in, uchar* bytes_out,
    uint cols_in, uint rows_in, uint channels, uint step_in, uint step_out) {

  double f_x[NEIGHBOURHOOD + 1][channels];
  uint id_in[NEIGHBOURHOOD + 1];

  for (uint row_o = 0; row_o < rows_out(); row_o++) {

    for (uint col_o = 0; col_o < cols_out(); col_o++) {

      uint id_o = get_index(col_o, row_o, channels, step_out);

      int col_i = floor(((double) col_o) / RATIO);
      int row_i = floor(((double) row_o) / RATIO);

      memset(f_x, 0.0, (NEIGHBOURHOOD + 1) * channels * sizeof (double));

      double d_x = get_distance(col_i, col_o);
      double d_y = get_distance(row_i, row_o);
      for (int y = 0; y <= NEIGHBOURHOOD; y++) {
        for (uint ch = 0; ch < channels; ch++) {
          for (int x = 0; x <= NEIGHBOURHOOD; x++) {

            id_in[x] = get_index(col_i + x - 1, row_i + y - 1, channels, step_in) + ch;
          }
          f_x[y][ch] = double (bytes_in[id_in[1]]) +
              double (bytes_in[id_in[2]] - bytes_in[id_in[0]])
              * d_x
              + double (bytes_in[id_in[0]] - 2 * bytes_in[id_in[1]] + bytes_in[id_in[2]])
              * d_x*d_x;
        }
      }
      for (uint ch = 0; ch < channels; ch++) {
        bytes_out[id_o + ch] = (f_x[1][ch] + (f_x[2][ch] - f_x[0][ch]) * d_y
            + (f_x[0][ch] - 2 * f_x[1][ch] + f_x[2][ch]) * d_y * d_y);

      }


    }

  }
}

double ResizeFilter::get_distance(int in, int out) {
  return fabs(((double) in) * RATIO - (((double) out)));
}
