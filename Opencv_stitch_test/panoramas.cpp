#include "panoramas.h"
#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "blend.h"
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
bool _interpolate_pixel(Mat & imag, double x, double y, Vec3b & interp_value);
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
	double r = 0;
	double c = 0;
	int r_warped, c_warped;
	Mat	result_solo = Mat::zeros(result.size(), result.type());
	Mat	result_chorus_ori = Mat::zeros(result.size(), result.type());
	Mat result_chorus_warp = Mat::zeros(result.size(), result.type());
	Mat average_blend_img;
	Mat multi_blend_img;
	Mat pt(3, 1, CV_64FC1);
	Mat H_inv(3, 3, CV_64FC1);
	Mat pt_warp(3, 1, CV_64FC1);
	Vec3b back_warped_pixel;
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
					result_solo.at<Vec3b>(r_warped, c_warped) = result.at<Vec3b>(r_warped, c_warped);
					continue;
				}
				else if (result.at<Vec3b>(r_warped, c_warped)[0] == 0 && result.at<Vec3b>(r_warped, c_warped)[1] == 0 && result.at<Vec3b>(r_warped, c_warped)[2] == 0) {
					_interpolate_pixel(img_to_warp, c, r, back_warped_pixel);
					result_solo.at<Vec3b>(r_warped, c_warped) = back_warped_pixel;
				}
				else {
					_interpolate_pixel(img_to_warp, c, r, back_warped_pixel);
					result_chorus_ori.at<Vec3b>(r_warped, c_warped) = result.at<Vec3b>(r_warped, c_warped);
					result_chorus_warp.at<Vec3b>(r_warped, c_warped) = back_warped_pixel;
					//result.at<Vec3b>(r_warped, c_warped)[0] = (result.at<Vec3b>(r_warped, c_warped)[0] + back_warped_pixel[0]) / 2;
					//result.at<Vec3b>(r_warped, c_warped)[1] = (result.at<Vec3b>(r_warped, c_warped)[1] + back_warped_pixel[1]) / 2;
					//result.at<Vec3b>(r_warped, c_warped)[2] = (result.at<Vec3b>(r_warped, c_warped)[2] + back_warped_pixel[2]) / 2;
				}
		}
	}
	blend_multi_band(result_chorus_ori, result_chorus_warp, multi_blend_img);
	average_blend(result_chorus_ori, result_chorus_warp, average_blend_img);
	result = multi_blend_img + result_solo;
	imwrite("multi_blend.jpg", result);
	imwrite("average_blend.jpg", average_blend_img + result_solo);
	//imshow("debug", result);
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
		//H_tmp = get_the_8_param_transform(obj, scene, sample_id, min_sample_count);
		H_tmp = get_the_rotation_param(obj, scene, sample_id, min_sample_count);
		error_tmp = _calc_error(obj, scene, sample_id, min_sample_count, H_tmp);
		if (error_tmp < error_min) {
			error_min = error_tmp;
			H_best = H_tmp;
		}
		//if (error_min < error_thres) {
		//	return H_best;
		//}
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
/* 
* get the interpolated value of point(x, y) of image. 
* ---------------------------------------------------
* @imag: the original image matrix.
* @x: the x coordinate of the point.
* @y: the y coordinate of the point.
* @interp_value: interpolated value.
* ----------------------------------------------------
* return: success or fail.
*/
bool _interpolate_pixel(Mat & imag, double x, double y, Vec3b & interp_value) {
	assert(x < imag.cols && y < imag.rows);
	assert(x >= 0 && y >= 0);
	//find the four neigbors.
	/*
	*  a   b;
	*    $  ;
	*  c   d;
	*/
	int a_x, a_y, b_x, b_y, c_x, c_y, d_x, d_y;
	a_x = cvRound(x - 0.5);
	a_x = a_x <  0? 0: a_x;
	a_y = cvRound(y - 0.5);
	a_y = a_y <	 0? 0: a_y;
	b_x = cvRound(x + 0.5);
	b_x = b_x >= imag.cols ? imag.cols-1 : b_x;
	b_y = cvRound(y - 0.5);
	b_y = b_y < 0 ? 0 : b_y;
	c_x = cvRound(x - 0.5);
	c_x = c_x < 0 ? 0 : c_x;
	c_y = cvRound(y + 0.5);
	c_y = c_y >= imag.rows ?  imag.rows-1: c_y;
	d_x = cvRound(x + 0.5);
	d_x = d_x >= imag.cols ? imag.cols-1: d_x;
	d_y = cvRound(y + 0.5);
	d_y = d_y >= imag.rows ? imag.rows-1: d_y;
	// interpolate the pixel in (x,y).
	int i = 0;
	for (i = 0; i < 3; ++i) {
		interp_value[i] = (imag.at<Vec3b>(a_y, a_x)[i] + imag.at<Vec3b>(b_y, b_x)[i] + imag.at<Vec3b>(c_y, c_x)[i] + imag.at<Vec3b>(d_y, d_x)[i]) / 4;
	}
	return true;
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
			H.at<double>(r, c) = H_in_row.at<double>(0, c + r * cols);
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
