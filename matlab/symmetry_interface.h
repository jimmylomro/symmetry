/*=================================================================
 *
 *  BRISK - Binary Robust Invariant Scalable Keypoints
 *  Reference implementation of
 *  [1] Stefan Leutenegger,Margarita Chli and Roland Siegwart, BRISK:
 *  	Binary Robust Invariant Scalable Keypoints, in Proceedings of
 *  	the IEEE International Conference on Computer Vision (ICCV2011).
 *
 * This file is part of BRISK.
 * 
 * Copyright (C) 2011  The Autonomous Systems Lab (ASL), ETH Zurich,
 * Stefan Leutenegger, Simon Lynen and Margarita Chli.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the ASL nor the names of its contributors may be 
 *       used to endorse or promote products derived from this software without 
 *       specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *=================================================================*/

#include <symbrisk/symbrisk.h>
#include <own/own.h>
#include <dbscan/dbscan.h>

#include <mex.h>
#include <vector>
#include <string.h>
#include <opencv2/opencv.hpp>


// test input args
bool mxIsScalarNonComplexDouble(const mxArray* arg);

class symmetryInterface{

public:

    symmetryInterface( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    ~symmetryInterface();
    
    inline void set(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    inline void loadImage( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    inline void detect(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    inline void describe(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    inline void knnMatch( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    inline void cluster( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    inline void image( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
        
        
private:

	// computer vision and machine learning tools
    cv::Ptr<own::OwnFeatureDetector>  detectorPtr;
	cv::Ptr<brisk::symBriskExtractor> descriptorPtr;
	cv::Ptr<cv::DescriptorMatcher>    matcherPtr;
	DBSCAN *dbscan;
            
	// variables
    cv::Mat img;								// temporary image stored with loadImage
    cv::Mat matchingMask;						// mask to match only keypoints of the same class
    std::vector<cv::KeyPoint> keypoints;		// temporary keypoint storage
    
    // own settings
    float own_threshold;
    int   own_kernSize;
    int   own_nMaps;

	// symbrisk settings
	bool  symbrisk_rotInv;
    bool  symbrisk_scaleInv;
    float symbrisk_patternScale;

	// matching settings
    int   matcher_dist;
    bool  matcher_maskFeats;
    
    // dbscan settings
    float dbscan_epsilon;
    int	  dbscan_minPts;
};
