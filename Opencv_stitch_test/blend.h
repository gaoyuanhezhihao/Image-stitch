#ifndef BLEND_H
#define BLEND_H
#include "opencv2\core\core.hpp"

bool blend_multi_band(const Mat & image1, const Mat & image2, Mat & blend_result, double sigma_first = 1.6, int pyramid_level = 4);
bool average_blend(const Mat & image1, const Mat & image2, Mat & blend_result);
#endif //BLEND_H