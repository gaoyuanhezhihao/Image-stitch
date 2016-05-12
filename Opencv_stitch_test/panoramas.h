#ifndef PANORAMAS_H
#define PANORAMAS_H
#include "opencv2\core\core.hpp"
#include <vector>
using namespace std;
using namespace cv;
Mat get_the_8_param_transform(vector< Point2f > & obj, vector< Point2f > & scene, int *sample_id, int sample_count);
Mat ransac_8_param(vector< Point2f > & obj, vector< Point2f > & scene, int max_iterate, \
	double error_thres, int min_sample_count);
bool average_warpPerspective(Mat & img_to_warp, Mat & result, Mat &H);
bool average_backwarpPerspective(Mat & img_to_warp, Mat & result, Mat &H);
#endif //PANORAMAS_H