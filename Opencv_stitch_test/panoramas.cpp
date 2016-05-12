#include "panoramas.h"
#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include <vector>
#include <iostream>
#include <time.h>
#include <random>
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
bool _get_mat_from_row(Mat & H, Mat & H_in_row, int rows, int cols);
Mat get_the_rotation_param(vector< Point2f > & obj, vector< Point2f > & scene, int *sample_id, int sample_count);
bool average_warpPerspective(Mat & img_to_warp, Mat & result, Mat &H) {
	int r = 0;
	int c = 0;
	int r_warped, c_warped;
	Mat pt(3, 1, CV_64FC1);
	pt.at<double>(2, 0) = 1;
	Mat pt_warp(3, 1, CV_64FC1);
	for (r = 0; r < img_to_warp.rows; ++r) {
		for (c = 0; c<img_to_warp.cols;++c) {
			pt.at<double>(0, 0) = c;
			pt.at<double>(1, 0) = r;
			pt_warp = H * pt;
			//cout <<"r:" <<  r << ", c: " << c << endl;
			r_warped = pt_warp.at<double>(1, 0) / pt_warp.at<double>(2, 0);
			c_warped = pt_warp.at<double>(0, 0) / pt_warp.at<double>(2, 0);
			//cout << "r_warped: " << r_warped << " c_warped: " << c_warped << endl;
			if (r_warped < 0 || r_warped >= result.rows || c_warped >= result.cols || c_warped < 0) {
				continue;
			}
			else if (result.at<Vec3b>(r_warped, c_warped)[0] == 0 && result.at<Vec3b>(r_warped, c_warped)[1] == 0 && result.at<Vec3b>(r_warped, c_warped)[2] == 0) {
				result.at<Vec3b>(r_warped, c_warped) = img_to_warp.at<Vec3b>(r, c);
			}
			else {
				result.at<Vec3b>(r_warped, c_warped)[0] = (result.at<Vec3b>(r_warped, c_warped)[0] + img_to_warp.at<Vec3b>(r, c)[0]) / 2;
				result.at<Vec3b>(r_warped, c_warped)[1] = (result.at<Vec3b>(r_warped, c_warped)[1] + img_to_warp.at<Vec3b>(r, c)[1]) / 2;
				result.at<Vec3b>(r_warped, c_warped)[2] = (result.at<Vec3b>(r_warped, c_warped)[2] + img_to_warp.at<Vec3b>(r, c)[2]) / 2;
			}
		}
	}
	return true;
}
bool average_backwarpPerspective(Mat & img_to_warp, Mat & result, Mat &H) {
	int r = 0;
	int c = 0;
	int r_warped, c_warped;
	Mat pt(3, 1, CV_64FC1);
	Mat H_inv(3, 3, CV_64FC1);
	Mat pt_warp(3, 1, CV_64FC1);
	pt_warp.at<double>(2, 0) = 1;
	invert(H, H_inv);
	for (r_warped = 0; r_warped < result.rows; ++r_warped) {
		for (c_warped = 0; c_warped <result.cols; ++c_warped) {
				//cout << "r_warped: " << r_warped << " c_warped: " << c_warped << endl;
				pt_warp.at<double>(0, 0) = c_warped;
				pt_warp.at<double>(1, 0) = r_warped;
				pt = H_inv * pt_warp;
				r = pt.at<double>(1, 0) / pt.at<double>(2, 0);
				c = pt.at<double>(0, 0) / pt.at<double>(2, 0);
				//cout <<"r:" <<  r << ", c: " << c << endl;
				if (r < 0 || r >= img_to_warp.rows || c < 0 || c >= img_to_warp.cols){
					continue;
				}
				else if (result.at<Vec3b>(r_warped, c_warped)[0] == 0 && result.at<Vec3b>(r_warped, c_warped)[1] == 0 && result.at<Vec3b>(r_warped, c_warped)[2] == 0) {
					result.at<Vec3b>(r_warped, c_warped) = img_to_warp.at<Vec3b>(r, c);
				}
				else {
					result.at<Vec3b>(r_warped, c_warped)[0] = (result.at<Vec3b>(r_warped, c_warped)[0] + img_to_warp.at<Vec3b>(r, c)[0]) / 2;
					result.at<Vec3b>(r_warped, c_warped)[1] = (result.at<Vec3b>(r_warped, c_warped)[1] + img_to_warp.at<Vec3b>(r, c)[1]) / 2;
					result.at<Vec3b>(r_warped, c_warped)[2] = (result.at<Vec3b>(r_warped, c_warped)[2] + img_to_warp.at<Vec3b>(r, c)[2]) / 2;

				}
		}
	}
	return true;
}
Mat ransac_8_param(vector< Point2f > & obj, vector< Point2f > & scene, int max_iterate, \
					double error_thres, int min_sample_count) {
	int i = 0;
	Mat H_best(3, 3, CV_64FC1);
	Mat H_tmp(3, 3, CV_64FC1);
	double error_min = 1000000;
	double error_tmp = 0;
	int sample_id[50] = { 0 };
	for (i = 0; i < max_iterate; ++i) {
		//cout << i << "-th circle" << endl;
		create_random_diff_int(sample_id, min_sample_count, obj.size());
		//cout << "\t random idex generated" << endl;
		H_tmp = get_the_8_param_transform(obj, scene, sample_id, min_sample_count);
		//H_tmp = get_the_rotation_param(obj, scene, sample_id, min_sample_count);
		error_tmp = _calc_error(obj, scene, sample_id, min_sample_count, H_tmp);
		if (error_tmp < error_min) {
			error_min = error_tmp;
			H_best = H_tmp;
		}
		if (error_min < error_thres) {
			return H_best;
		}
	}
	cout << "min error: " << error_min << endl;
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
	std::random_device r;
	std::default_random_engine e1(r());
	std::uniform_int_distribution<int> uniform_dist(0, range-1);

	int i = 0;
	int j = 0;
	bool uqiue_flag = false;
	for (i = 0; i< n; ++i)
	{
		
		do{
			uqiue_flag = true;
			data[i] = uniform_dist(e1);
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
Mat get_the_rotation_param(vector< Point2f > & obj, vector< Point2f > & scene, int *sample_id, int sample_count) {
	Mat H_in_row(1, 9, CV_64FC1);
	Mat H(3, 3, CV_64FC1);
	Mat A = Mat::zeros(2 * sample_count, 9, CV_64FC1);
	Mat A_T = Mat::zeros(9, 2 * sample_count, CV_64FC1);
	Mat value_eigen, row_eigen_array;
	int k = 0;
	int i = 0;
	int min_eigen_value_id = 0;
	double min_eigen_value = 100000;
	for (k = 0; k < sample_count; ++k) {
		i = sample_id[k];
		A.at<double>(2*k, 0) = obj[i].x;
		A.at<double>(2*k, 1) = obj[i].y;
		A.at<double>(2*k, 2) = 1;
		A.at<double>(2*k, 6) = -obj[i].x * scene[i].x;
		A.at<double>(2*k, 7) = -obj[i].y * scene[i].x;
		A.at<double>(2*k, 8) = -scene[i].x;

		A.at<double>(2*k + 1, 3) = obj[i].x;
		A.at<double>(2*k + 1, 4) = obj[i].y;
		A.at<double>(2*k + 1, 5) = 1;
		A.at<double>(2*k + 1, 6) = -obj[i].x * scene[i].y;
		A.at<double>(2*k + 1, 7) = -obj[i].y * scene[i].y;
		A.at<double>(2*k + 1, 8) = -scene[i].y;
	}

	transpose(A, A_T);
	eigen(A_T*A, value_eigen, row_eigen_array);
	for (k = 0; k < 9; ++k) {
		if (abs(value_eigen.at<double>(k, 0)) < min_eigen_value) {
			min_eigen_value = abs(value_eigen.at<double>(k, 0));
			min_eigen_value_id = k;
		}
	}
	H_in_row = row_eigen_array.row(min_eigen_value_id);
	_get_mat_from_row(H, H_in_row, 3, 3);
	return H;
}
bool _get_mat_from_row(Mat & H, Mat & H_in_row, int rows, int cols) {
	int r = 0, c = 0;
	for (r = 0; r < rows; ++r) {
		for (c = 0; c < cols; ++c) {
			H.at<double>(r, c) = H_in_row.at<double>(0, r + c * rows);
		}
	}
	return true;
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
