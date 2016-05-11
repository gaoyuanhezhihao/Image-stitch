#include "panoramas.h"
#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include <vector>
#include <iostream>
#include <time.h>
using namespace std;
using namespace cv;
/*
* calculate the H transform matrix for the 8-param model.
* ------------------------------------------------------
* @match_pairs: <point of the image going to be transformed, point of the base image>
* @im_trsf: the matrix of image going to be transformed.
* @im_base: the matrix of the base image.
* @iterate_max_times: the maximum iteration can be done.
* @err_rate_thres: if the error rate is below this, the result can be returned.
*/
bool _get_H_from_p(Mat &p, Mat &H);
void create_random_diff_int(int data[], int n, int range);
double _calc_error(vector<Point2f> & obj, vector<Point2f> & scene, int * sample_id, int sample_count, Mat &H);


Mat ransac_8_param(vector< Point2f > & obj, vector< Point2f > & scene, int max_iterate, \
					double error_thres, int min_sample_count) {
	int i = 0;
	Mat H_best(3, 3, CV_64FC1);
	Mat H_tmp(3, 3, CV_64FC1);
	double error_min = 1000000;
	double error_tmp = 0;
	int sample_id[50] = { 0 };
	for (i = 0; i < max_iterate; ++i) {
		create_random_diff_int(sample_id, min_sample_count, obj.size());
		H_tmp = get_the_8_param_transform(obj, scene, sample_id, min_sample_count);
		error_tmp = _calc_error(obj, scene, sample_id, min_sample_count, H_tmp);
		if (error_tmp < error_min) {
			error_min = error_tmp;
			H_best = H_tmp;
		}
	}
	return H_best;
}

double _calc_error(vector<Point2f> & obj, vector<Point2f> & scene, int * sample_id, int sample_count, Mat &H) {
	int k = 0;
	int i = 0;
	Mat x_trans(1, 1, CV_64FC1);
	Mat y_trans(1, 1, CV_64FC1);
	Mat scale_trans(1, 1, CV_64FC1);
	Mat point_obj(3, 1, CV_64FC1);
	point_obj.at<double>(2, 0) = 1;
	double error = 0;
	for (k = 0; k < sample_count; ++k) {
		i = sample_id[k];
		point_obj.at<double>(0, 0) = obj[i].x;
		point_obj.at<double>(1, 0) = obj[i].y;
		x_trans = H.row(0) * point_obj;
		y_trans = H.row(1) * point_obj;
		scale_trans = H.row(2) * point_obj;
		x_trans.at<double>(0, 0) = x_trans.at<double>(0, 0) / scale_trans.at<double>(0, 0);
		y_trans.at<double>(0, 0) = y_trans.at<double>(0, 0) / scale_trans.at<double>(0, 0);
		error += abs(x_trans.at<double>(0, 0) - scene[i].x);
		error += abs(y_trans.at<double>(0, 0) - scene[i].y);
	}
	return error;
}

void create_random_diff_int(int data[], int n, int range)
{

	/* initialize random seed: */
	srand(time(NULL));

	/* generate secret number between 1 and 10: */
	int i = 0;
	int j = 0;
	bool uqiue_flag = false;
	for (i = 0; i< n; ++i)
	{
		uqiue_flag = true;
		do{
			data[i] = rand() % range;
			for (j = 0; j < i; ++j) {
				if (data[j] == data[i]) {
					uqiue_flag = false;
					break;
				}
			}
		} while (!uqiue_flag);
		//cout << data[i] <<endl;
	}
}

Mat get_the_8_param_transform(vector< Point2f > & obj, vector< Point2f > & scene, int *sample_id, int sample_count) {
	Mat p(8, 1, CV_64FC1);
	Mat H(3, 3, CV_64FC1);
	Mat A = Mat::zeros(2 * sample_count, 8, CV_64FC1);
	Mat B(2 * sample_count, 1, CV_64FC1);
	int k = 0;
	int i = 0;
	for (k = 0; k < sample_count; ++k) {
		i = sample_id[k];
		A.at<double>(k, 0) = obj[i].x;
		A.at<double>(k, 1) = obj[i].y;
		A.at<double>(k, 2) = 1;
		A.at<double>(k, 6) = -obj[i].x * scene[i].x;
		A.at<double>(k, 7) = -obj[i].y * scene[i].x;
		B.at<double>(k, 0) = scene[i].x;

		A.at<double>(k + sample_count, 3) = obj[i].x;
		A.at<double>(k + sample_count, 4) = obj[i].y;
		A.at<double>(k + sample_count, 5) = 1;
		A.at<double>(k + sample_count, 6) = -obj[i].x * scene[i].y;
		A.at<double>(k + sample_count, 7) = -obj[i].y * scene[i].y;
		B.at<double>(k + sample_count, 0) = scene[i].y;
	}
	solve(A, B, p, DECOMP_SVD);
	_get_H_from_p(p, H);
	return H;
}

bool _get_H_from_p(Mat &p, Mat &H) {
	H.at<double>(0, 0) = p.at<double>(0, 0);
	H.at<double>(0, 1) = p.at<double>(1, 0);
	H.at<double>(0, 2) = p.at<double>(2, 0);
	H.at<double>(1, 0) = p.at<double>(3, 0);
	H.at<double>(1, 1) = p.at<double>(4, 0);
	H.at<double>(1, 2) = p.at<double>(5, 0);
	H.at<double>(2, 0) = p.at<double>(6, 0);
	H.at<double>(2, 1) = p.at<double>(7, 0);
	H.at<double>(2, 2) = 1;
	return true;
}
