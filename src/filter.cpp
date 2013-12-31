#include "../headers/filter.h"

void VideoFilter::filter(VideoCapture v_in, VideoWriter v_out) {

    Mat frame_in, frame_out;

    int cols_in = v_in.get(CV_CAP_PROP_FRAME_WIDTH);
    int rows_in = v_in.get(CV_CAP_PROP_FRAME_HEIGHT);

    int cols_out = (int) (RATIO * (double) cols_in);
    int rows_out = (int) (RATIO * (double) rows_in);

    while (true) {

        v_in >> frame_in;
        if (frame_in.empty()) {
            break;
        }
        frame_out.create(rows_out, cols_out, frame_in.type());

        uchar* bytes_in = (uchar*) (frame_in.data);
        uchar* bytes_out = (uchar*) (frame_out.data);

        process(bytes_in, bytes_out, frame_in.cols, frame_in.rows,
                frame_in.channels(), frame_in.step, frame_out.step);

        v_out << frame_out;

    }
}