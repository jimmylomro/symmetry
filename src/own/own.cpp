/*
    Copyright (C) 2016 Jaime Lomeli-R. Univesity of Southampton

    This file is part of OWN.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above copyright
         notice, this list of conditions and the following disclaimer in the
         documentation and/or other materials provided with the distribution.
       * Neither the name of the ASL nor the names of its contributors may be
         used to endorse or promote products derived from this software without
         specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <algorithm>
#include <own/own_params.h>
#include <own/own.h>
#include <own/nms.h>


//---------------------------------------------------------------------------------------------------------------------
//-----------------------------OwnFeatureMaps--------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------

// Constructors
own::OwnFeatureMaps::OwnFeatureMaps(float thresh, int M, int K, int kernSize, int ethresh) {

	this->thresh	= thresh;
	this->M			= M;
	this->K			= K;
	this->kernSize	= kernSize;
	this->ethresh	= ethresh;

	initCentreMat();
	initKernels();
}



// Methods
void own::OwnFeatureMaps::createFeatureMaps(const cv::Mat& image) {

	fillMagList(image);

	featureMaps.clear();
	featureMaps.resize(K);

	for (int k = 0; k < K; k++) {
		featureMaps[k] = createOneFeatureMap(k, image.rows, image.cols);
	}
}


void own::OwnFeatureMaps::detectKeypoints(std::vector<cv::KeyPoint>& keypoints, const cv::Mat& image) {

	if (featureMaps.empty()) {
		if (image.empty())
			throw std::runtime_error("Error: featureMaps is empty, no image loaded");
		else
			createFeatureMaps(image);
	}
	
	keypoints.clear();
	keypoints.reserve(own::MAX_EXPECTED_KEYPOINTS_PER_MAP * K);

	for (int k = 0; k < K; k++) {
		std::vector<cv::KeyPoint> keypointsInMap = detectKeypointsInMap(k);
		
		for (int i = 0; i < keypointsInMap.size(); i++)	
			keypoints.push_back(keypointsInMap[i]);
	}
}


cv::Mat own::OwnFeatureMaps::createOneFeatureMap(int centIdx, int rows, int cols) {

	if (magList.empty())
		throw std::runtime_error("Error: magList is empty");

	cv::Mat map = cv::Mat::zeros(rows, cols, CV_32FC1);

	float* mag_rowPtr;
	float* map_rowPtr;
	float* cen_rowPtr;
	float* centrePtr;
	
	float num;
	float den;
	float numDif;
	float denDif;
	float denAux;

	centrePtr = centres.ptr<float>(centIdx);
	for (int row = 0; row < rows; row++) {
		map_rowPtr = map.ptr<float>(row);
		for (int col = 0; col < cols; col++) {
			
			if (energyValid.at<char>(row*cols+col)) {

				mag_rowPtr = magList.ptr<float>(row*cols+col);

				num = 0;
				den = 0;
			
				// Calculate numerator
				for (int m = 0; m < M; m++) {
					numDif  = centrePtr[m] - mag_rowPtr[m];
					num    += numDif * numDif;
				}
				num = std::max(num,(float)0.000001);	// Handle division by 0
				num = 1/(num*num);

				// Calculate denominator
				for (int k = 0; k < K; k++) {
					cen_rowPtr = centres.ptr<float>(k);
					denAux = 0;
					for (int m = 0; m < M; m++) {
						denDif  = cen_rowPtr[m] - mag_rowPtr[m];
						denAux += denDif * denDif;
					}
					denAux = std::max(denAux,(float)0.000001);	// Handle division by 0
					den += 1/(denAux*denAux);
				}

				// Assign calculated membership to its position
				map_rowPtr[col] = num/den;
			}
			else {
				map_rowPtr[col] = 0;
			}
			
		}
	}
	return map;
}


std::vector<cv::KeyPoint> own::OwnFeatureMaps::detectKeypointsInMap(int centIdx) {

	if (featureMaps.empty())
		throw std::runtime_error("Error: featureMaps is empty, no image loaded");

	std::vector<cv::KeyPoint> toReturn;

	nonMaximaSuppression(featureMaps[centIdx], 2, toReturn, thresh, kernSize, centIdx);

	return toReturn;
}


void own::OwnFeatureMaps::fillMagList(const cv::Mat& image) {

	cv::Mat grayImage;

	if (image.empty()) return;

    	// Convert the image to gray scale 
    	switch (image.type()) {

	case CV_8UC1:
		image.convertTo(grayImage, CV_32FC1, 1/255.0);
		break;

    	case CV_8UC3:
        	cvtColor(image, grayImage, CV_BGR2GRAY);
		grayImage.convertTo(grayImage, CV_32FC1, 1/255.0);
        	break;

    	case CV_32FC3:
        	cvtColor(image, grayImage, CV_BGR2GRAY);
		grayImage.convertTo(grayImage, CV_32FC1, 1/255.0);
        	break;

    	default:
        	grayImage = image.clone();
        	break;
    	}

    	assert(grayImage.type() == CV_32FC1);
    	assert(grayImage.isContinuous());

	
	cv::Mat tempRe;
	cv::Mat tempIm;
	cv::Mat tempMag;
	cv::Mat tempMagList = cv::Mat::zeros(M, grayImage.rows*grayImage.cols, CV_32FC1);

	for (int m = 0; m < M; m++) {
		
		cv::filter2D(grayImage, tempRe, -1, re_filters[m]);
		cv::filter2D(grayImage, tempIm, -1, im_filters[m]);
		cv::magnitude(tempRe, tempIm, tempMag);
		assert(tempMag.type() == CV_32FC1);

		memcpy(tempMagList.ptr<float>(m), tempMag.data, grayImage.rows * grayImage.cols * sizeof(float));
	}

	cv::transpose(tempMagList, magList);
	
	energyValid = cv::Mat::zeros(magList.rows,1,CV_8UC1);
	
	float* rowPtr;	
	float  E;

	for (int row = 0; row < magList.rows; row++) {
		
		E = 0;
		rowPtr = magList.ptr<float>(row);
		for (int m = 0; m < M; m++)
			E += rowPtr[m];
		
		if (E > ethresh)
			energyValid.at<char>(row) = 1;
		else
			energyValid.at<char>(row) = 0;
		
		for (int m = 0; m < M; m++) {
			if (E > ethresh)
				rowPtr[m] *= 1/E;
			else
				rowPtr[m] = 0;
		}
	}
}


void own::OwnFeatureMaps::initCentreMat() {


	centres = cv::Mat::zeros(K, M, CV_32FC1);

	float* rowPtr;

	for (int row = 0; row < K; row++) {
		rowPtr = centres.ptr<float>(row);
		for (int col = 0; col < M; col++) {
			rowPtr[col] = own::params::centres[row][col];
		}
	}
}



void own::OwnFeatureMaps::initKernels() {
			
	if (own::params::MAX_M < M)
		throw std::runtime_error("Error: K and M must be smaller than MAX_K and MAX_M");

	re_filters.clear();
	im_filters.clear();
	re_filters.resize(M);
	im_filters.resize(M);
			
	for (int m = 0; m < M; m++) {
				
		cv::Mat origRe(own::params::MAX_KERN_SIZE, own::params::MAX_KERN_SIZE, CV_32FC1, own::params::filters_re[m]);
		cv::Mat origIm(own::params::MAX_KERN_SIZE, own::params::MAX_KERN_SIZE, CV_32FC1, own::params::filters_im[m]);
				
		cv::Mat newRe(kernSize, kernSize, CV_32FC1);
		cv::Mat newIm(kernSize, kernSize, CV_32FC1);

		cv::resize(origRe, newRe, newRe.size(), 0, 0, CV_INTER_CUBIC);
		cv::resize(origIm, newIm, newIm.size(), 0, 0, CV_INTER_CUBIC);
	
		re_filters[m] = newRe;
		im_filters[m] = newIm;
	}
}


void own::OwnFeatureMaps::getFeatureMaps(std::vector<cv::Mat>& toReturn) {

	toReturn = featureMaps;
}


//---------------------------------------------------------------------------------------------------------------------
//-----------------------------OwnFeatureDetector----------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------

// Constructor
own::OwnFeatureDetector::OwnFeatureDetector(float thresh, int M, int K, int kernSize) {

	this->thresh	= thresh;
	this->M 	= M;
	this->K		= K;
	this->kernSize	= kernSize;
	
	fm = new own::OwnFeatureMaps(thresh, M, K, kernSize);
}


// Methods
void own::OwnFeatureDetector::getFeatureMaps(std::vector<cv::Mat>& featureMaps) {

	fm->getFeatureMaps(featureMaps);
}


void own::OwnFeatureDetector::detectImpl(const cv::Mat& image,
					std::vector<cv::KeyPoint>& keypoints,
					const cv::Mat& mask) const {

	fm->detectKeypoints(keypoints, image);
}

own::OwnFeatureDetector::~OwnFeatureDetector(void) {

	delete fm;
}
