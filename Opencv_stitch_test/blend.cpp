#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"

#include <vector>
#include <iostream>
#define _MY_DEBUG
using namespace cv;
using namespace std;


bool _build_gauss_pyr(Mat & original_img, vector<Mat> & gauss_pyr, int pyr_level, double sigma_first);
bool _build_laplacian_pyr(vector<Mat> &gauss_pyr, vector<Mat>& expand_gauss_pyr, vector<Mat>& laplacian_pyr, int pyr_levels);
bool _build_expand_pyr(vector<Mat> & gauss_pyr, vector<Mat> &expand_gauss_pyr, int pyr_level);
bool _merge_laplacian_pyr(vector<Mat> &lap_pyr_1, vector<Mat> &lap_pyr_2, vector<Mat> & gauss_pyr);
bool auto_subtract_mat(const Mat & subtractor, const Mat & minuend, Mat & result);
bool _reconstruct_img(vector<Mat> & lap_pry, Mat &blend_img);
bool auto_add_mat(const Mat &adder, const Mat &addor, Mat & result);
double _RGB_to_gray(Vec3s & rgb);
bool _merge_laplacian_img(Mat & img_master, Mat & img_slave, Mat & gauss_master);
/*
* blend the image using Multi-band Blending.
*--------------------------------------------------
* @image1: the first image to be blended. The blended image will be copyed to image1.
* @image2: the second image to be blended.
* @blend_im: blended result will be stored here.
* @sigma_first: the sigma of the gaussian kernel function in the first gaussian image. Its default is 1.6
* @pyramid_level: the level of the laplacian pyramid. Its default is 4.
*/
/*average_blend: blend the image use simple average method.
* ---------------------------------------------------------
* @image1: first image.
* @image2: second image.
* @blend_result: blended image will be stored here.
* ---------------------------------------------------------
* return: success or fail.
*/
bool average_blend(const Mat & image1, const Mat & image2, Mat & blend_result){
	blend_result = (image1 + image2) / 2;
	return true;
}
bool blend_multi_band(const Mat & image1, const Mat & image2, Mat & blend_result, double sigma_first = 1.6, int pyramid_level = 4) {
	vector<Mat> gauss_pyr1;
	vector<Mat> gauss_pyr2;
	vector<Mat> expand_gauss_pyr1;
	vector<Mat> expand_gauss_pyr2;
	vector<Mat> laplacian_pyr1;
	vector<Mat> laplacian_pyr2;
	Mat copy_base_im1;
	Mat debug_img;
	Mat blend_img;
	image1.convertTo(copy_base_im1, CV_16SC3);
	Mat copy_base_im2;
	image2.convertTo(copy_base_im2, CV_16SC3);
	gauss_pyr1.reserve(pyramid_level);
	gauss_pyr2.reserve(pyramid_level);
	expand_gauss_pyr1.reserve(pyramid_level - 1);
	expand_gauss_pyr2.reserve(pyramid_level - 1);
	// build gaussian pyramid.
	_build_gauss_pyr(copy_base_im1, gauss_pyr1, pyramid_level, sigma_first);
	_build_gauss_pyr(copy_base_im2, gauss_pyr2, pyramid_level, sigma_first);

	// build expand pyramid.
	_build_expand_pyr(gauss_pyr1, expand_gauss_pyr1, pyramid_level - 1);
	_build_expand_pyr(gauss_pyr2, expand_gauss_pyr2, pyramid_level - 1);
	// laplacian pyramid.
	_build_laplacian_pyr(gauss_pyr1, expand_gauss_pyr1, laplacian_pyr1, pyramid_level);
	_build_laplacian_pyr(gauss_pyr2, expand_gauss_pyr2, laplacian_pyr2, pyramid_level);
	// merge the laplacian pyramid.
	_merge_laplacian_pyr(laplacian_pyr1, laplacian_pyr2, gauss_pyr1);
	// reconstruct the image.
	_reconstruct_img(laplacian_pyr1, blend_img);
	blend_img.convertTo(blend_result, image1.type());
#ifdef _MY_DEBUG
	for (size_t k = 0; k < laplacian_pyr1.size(); ++k) {
		debug_img = laplacian_pyr1[k];
	}
	for (size_t k = 0; k < gauss_pyr1.size(); ++k) {
		debug_img = gauss_pyr1[k];
	}
	for (size_t k = 0; k < expand_gauss_pyr1.size(); ++k) {
		debug_img = expand_gauss_pyr1[k];
	}
#endif // _MY_DEBUG
	return true;
}
/* _reconstruct_img: reconstruct the image from the blended(merged) laplacian pyramid.
* ------------------------------------------------------------------------------------
* @lap_pyr: merged laplacian pyramid.
* @blend_img: blended image will stored in it.
* ------------------------------------------------------------------------------------
* return: success or fail.
*/
bool _reconstruct_img(vector<Mat> & lap_pry, Mat &blend_img) {
	Mat gauss_im_fake, expanded_im_fake;
	int id = lap_pry.size() - 1;
	gauss_im_fake = lap_pry[id];
	for (id = lap_pry.size() - 2; id >= 0; --id) {
		pyrUp(gauss_im_fake, expanded_im_fake, Size(gauss_im_fake.cols * 2, gauss_im_fake.rows * 2));
		auto_add_mat(lap_pry[id], expanded_im_fake, gauss_im_fake);
		//gauss_im_fake = expanded_im_fake + lap_pry[id];
	}
	blend_img = gauss_im_fake;

	return true;
}
/*
* _build_laplacian_pyr: build laplacian pyramid from gaussian pyramid and its expanded gaussian pyramid.
* ------------------------------------------------------------------------------------------------------
* @gauss_pyr: gaussian pyramid.
* @expand_gauss_pyr: expanded gaussian pyramid from gauss_pyr.
* @laplacian_pyr: laplacian pyramid.
* @pyr_level: count of the pyramid's levels.
* -------------------------------------------------------------------------------------------------------
*/
bool _build_laplacian_pyr(vector<Mat> &gauss_pyr, vector<Mat>& expand_gauss_pyr, vector<Mat>& laplacian_pyr, int pyr_levels) {
	int i = 0;
	Mat lap_img;
#ifdef _MY_DEBUG
	Mat debug_img;
#endif//  _MY_DEBUG
	for (i = 0; i < gauss_pyr.size() - 1; ++i) {
		//lap_img = gauss_pyr[i] - expand_gauss_pyr[i];
#ifdef  _MY_DEBUG
		debug_img = gauss_pyr[i];
		debug_img = expand_gauss_pyr[i];
#endif //  _MY_DEBUG
		auto_subtract_mat(gauss_pyr[i], expand_gauss_pyr[i], lap_img);
		laplacian_pyr.push_back(lap_img);
	}
	laplacian_pyr.push_back(gauss_pyr[i]);// last gauss image is last laplacian image.
	return true;
}
/* merge laplacian pyramid of two images. merged imaged will be saved in the first pyramid.
* ------------------------------------------
* @lap_pry_1: first laplacian pyramid.
* @lap_pry_2: second laplacian pyramid.
* @gauss_pyr: the gaussian pyramid of the first image.
* -------------------------------------------
* retrun: success or fail.
*/
bool _merge_laplacian_pyr(vector<Mat> &lap_pyr_1, vector<Mat> &lap_pyr_2, vector<Mat> & gauss_pyr) {
	assert(lap_pyr_1.size() > 0);
	assert(lap_pyr_1.size() == lap_pyr_2.size());
	auto it1 = lap_pyr_1.begin();
	auto it2 = lap_pyr_2.begin();
	auto it_gauss = gauss_pyr.begin();
	for (; it1 != lap_pyr_1.end(); ++it1, ++it2, ++it_gauss) {
		//*it1 += *it2;
		//*it1 /= 2;
		_merge_laplacian_img(*it1, *it2, *it_gauss);
	}
	return true;
}
bool _merge_laplacian_img(Mat & img_master, Mat & img_slave, Mat & gauss_master) {
	int r, c;
	double gauss_rate;
	for (r = 0; r < img_master.rows; ++r) {
		for (c = 0; c < img_master.cols; ++c) {
			gauss_rate = _RGB_to_gray(gauss_master.at<Vec3s>(r, c));
			img_master.at<Vec3s>(r, c) = gauss_rate * img_master.at<Vec3s>(r, c) + \
				(1 - gauss_rate) * img_slave.at<Vec3s>(r, c);
		}
	}
	return true;
}
double _RGB_to_gray(Vec3s & rgb) {
	double num = pow(rgb[0], 2) + pow(rgb[1], 2) + pow(rgb[2], 2);
	num = sqrt(num);
	double denom = pow(255, 2) * 3;
	denom = sqrt(denom);
	return num / denom;
}
/* auto_add_mat: result = adder + addor. Make add operation on matrix.
*	Sizes of adder and addor may not be same.
*	The size of the result is the same with adder's.
*	----------------------------------------------------------------------------------------
* @adder:
* @addor:
* @result:
* -----------------------------------------------------------------------------------------
* return: success for fail.
*/
bool auto_add_mat(const Mat &adder, const Mat &addor, Mat & result) {
	adder.copyTo(result);
	int col_count = adder.cols > addor.cols ? addor.cols : adder.cols;// choose the smaller one.
	int row_count = adder.rows > addor.rows ? addor.rows : adder.rows;// choose the smaller one.
	auto ptr_row_addor = addor.ptr < Vec3s>(0);
	auto ptr_row_result = result.ptr<Vec3s>(0);
	int r, c;
	for (r = 0; r < row_count; ++r){
		ptr_row_addor = addor.ptr<Vec3s>(r);
		ptr_row_result = result.ptr<Vec3s>(r);
		for (c = 0; c < col_count; ++c) {
			ptr_row_result[c] += ptr_row_addor[c];
		}
	}
	return true;
}


/* auto_subtract_mat: result = subtractor - minuend. Sizes of subtractor and minuend may not be same.
* -----------------------------------------------------
* @subtractor:
* @minuend:
* @result: result will be sorted here.
* -------------------------------------------------------
* return : success or fail.
*/
bool auto_subtract_mat(const Mat & subtractor, const Mat & minuend, Mat & result) {
	if (subtractor.size == minuend.size) {
		result = subtractor - minuend;
		return true;
	}
	else{
		auto ptr_row_main = subtractor.ptr < Vec3s>(0);
		auto ptr_row_slave = minuend.ptr<Vec3s>(0);
		int row_count = subtractor.rows <= minuend.rows ? subtractor.rows : minuend.rows;
		int col_count = subtractor.cols <= minuend.cols ? subtractor.cols : minuend.cols;
		result = Mat::zeros(row_count, col_count, subtractor.type());
		auto ptr_row_result = result.ptr<Vec3s>(0);
		int r = 0, c = 0;
		for (r = 0; r < row_count; ++r) {
			ptr_row_main = subtractor.ptr <Vec3s>(r);
			ptr_row_slave = minuend.ptr<Vec3s>(r);
			ptr_row_result = result.ptr<Vec3s>(r);
			for (c = 0; c < col_count; ++c){
				if (c > col_count * 3 / 4){
					;
				}
				ptr_row_result[c] = ptr_row_main[c] - ptr_row_slave[c];
			}
		}
	}
}
/*
* _build_expand_pyr: get the expanded gaussian pyramid level from its up-level pyramid level.
* @gauss_pyr: the pyramid of the gaussian blur.
* @expand_gauss_pyr: the expanded gaussian image pyramid.
* @pyr_level: count of levels of the expand_gauss_pyr.
*/
bool _build_expand_pyr(vector<Mat> & gauss_pyr, vector<Mat> &expand_gauss_pyr, int pyr_level) {
	int i = 0;
	Mat tmp_img;

	for (i = 0; i < pyr_level; ++i) {
		pyrUp(gauss_pyr[i + 1], tmp_img, Size(gauss_pyr[i + 1].cols * 2, gauss_pyr[i + 1].rows * 2));
		expand_gauss_pyr.push_back(tmp_img);
	}
	return true;
}
/*
* build the gaussian pyramid from the original image.
* ----------------------------------------------------
* @original_img: the original image.
* @gauss_pyr: vector containing the pyramid. gauss_pyr[0] is the base level of the pyramid.
* @pyr_level: the level of the pyramid.
* @sigma_first: sigma for the first gaussian blur.
*/
bool _build_gauss_pyr(Mat & original_img, vector<Mat> & gauss_pyr, int pyr_level, double sigma_first) {
	gauss_pyr.push_back(original_img);//base level. not gaussian blured.
	int i = 0;
	Mat tmp_img;
	Mat debug_img = gauss_pyr[0];
	double sigma = sigma_first;
	for (i = 1; i < pyr_level; ++i, sigma *= 2) {
		GaussianBlur(gauss_pyr[i - 1], tmp_img, Size(0, 0), sigma_first*pow(2, i - 1));
		pyrDown(tmp_img, tmp_img, Size(tmp_img.cols / 2, tmp_img.rows / 2));
		gauss_pyr.push_back(tmp_img);
		debug_img = gauss_pyr[i];
	}
	return true;
}