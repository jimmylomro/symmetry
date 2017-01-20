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

OwnFeatureMaps:
    There are two ways of creating the feature maps, you could either call detectKeypoints and send a valid image
    or you could first call createFeatureMaps. The first method will also detect the keypoints.
    If detectKeypoints is called without an image argument and createFeatureMaps has not been called (i.e. featureMaps is empty)
    the function will throw an error.
*/

#ifndef _OWN_H_
#define _OWN_H_


#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>


#ifndef M_PI
	#define M_PI 3.141592653589793
#endif


namespace own{

	const int MAX_EXPECTED_KEYPOINTS_PER_MAP = 1000;

	class CV_EXPORTS OwnFeatureMaps {
		public:
			OwnFeatureMaps(float thresh = 0.5, int M = 8, int K = 30, int kernSize = 31, int ethresh = 5);
			
			// this function creates the feature maps
			void createFeatureMaps(const cv::Mat& image);

			// this function detects the keypoints in all feature maps
			void detectKeypoints(std::vector<cv::KeyPoint>& keypoints, const cv::Mat& image);
			
			// accessor for the feature maps
			void getFeatureMaps(std::vector<cv::Mat>& toReturn);
	
		
		private:

			float	thresh;
			int	M, K, kernSize, ethresh;
			cv::Mat energyValid;

			// Parameter initialisation
			cv::Mat centres;
			std::vector<cv::Mat> re_filters;
			std::vector<cv::Mat> im_filters;

			// Mats containing the memberships for each centre
			std::vector<cv::Mat> featureMaps;
			// List of the magnitudes of the complex filters, this matrix has nPixels rows and M columns
			cv::Mat magList;

			void initCentreMat();
			void initKernels();
			void fillMagList(const cv::Mat& image);
			
			cv::Mat createOneFeatureMap(int centIdx, int rows, int cols);
			std::vector<cv::KeyPoint> detectKeypointsInMap(int centIdx);

	};


	class CV_EXPORTS OwnFeatureDetector : public cv::FeatureDetector {
		public:
			OwnFeatureDetector(float thresh = 0.5, int M = 8, int K = 30, int kernSize = 31);
			~OwnFeatureDetector(void);
			
			float thresh;
			int M;
			int K;
			int kernSize;

			// accesors
			void getFeatureMaps(std::vector<cv::Mat>& featureMaps);
		
		private:
			own::OwnFeatureMaps *fm;

		protected:
			// cv::FeatureDetector
			virtual void detectImpl(const cv::Mat& image,
					std::vector<cv::KeyPoint>& keypoints,
					const cv::Mat& mask=cv::Mat() ) const;		
	};

}

#endif // _OWN_H_
