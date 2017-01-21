/* non-maximal values
 *
 * This is an adaptation of the code nonmaxsuppts.m by Peter Kovesi: http://www.peterkovesi.com/matlabfns/
 *
 *
 * Code by Jaime Lomeli-R.
 *	25th of May 2016
 *
 *
 *	Notes: src must be of type CV_32FC1
 *
 *
*/

#include <stdio.h>
#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <nms.h>

using namespace cv;

void nonMaximaSuppression(Mat& src, const int sz, std::vector<KeyPoint> &keypoints, const float thresh, int kernSize, const int featID) {

	if (src.type() != CV_32FC1)
		return;

	// initialise variables
	const int M = src.rows;
	const int N = src.cols;

	Mat dilIm;

	int sze = 2*sz+1;                							// calculate size of dilation mask.
    	dilate(src, dilIm, getStructuringElement(MORPH_RECT,Size(sze,sze),Point(sz,sz)));	// Grey-scale dilate.

	float* rowPtrD;
	float* rowPtrS;
	float  valD;
	float  valS;

	for (int row = sz; row < M-sz; row++) {
		rowPtrD = dilIm.ptr<float>(row);
		rowPtrS = src.ptr<float>(row);
		for (int col = sz; col < N-sz; col++) {
			valD = rowPtrD[col];
			valS = rowPtrS[col];
			if (valS == valD && valS > thresh) {
				keypoints.push_back(KeyPoint(col,row,(float)kernSize,-1,valS,0,featID));
			}
		}
	}
}
