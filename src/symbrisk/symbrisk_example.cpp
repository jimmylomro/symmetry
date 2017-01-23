#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <list>

#include <symbrisk.h>


int main(void) {


	// Read images and convert to gray scale
	cv::Mat imgRGB1 = cv::imread("../res/img1.jpg");
	cv::Mat imgRGB2 = imgRGB1.clone();
	cv::Mat imgGray1;
	cv::Mat imgGray2;
	cv::cvtColor(imgRGB1, imgGray1, CV_BGR2GRAY);
	cv::cvtColor(imgRGB2, imgGray2, CV_BGR2GRAY);


	// FAST Feature detection
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;
	cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(30,true);
	detector->detect(imgGray1,keypoints1);
	detector->detect(imgGray2,keypoints2);


	// get the symBrisk descriptors
	cv::Mat descriptors1;
	cv::Mat descriptors2;
	cv::Mat mirrDescriptors;
	cv::Ptr<brisk::symBriskExtractor> descriptorExtractor = brisk::symBriskExtractor::create();
	descriptorExtractor->compute(imgGray1,keypoints1,descriptors1, mirrDescriptors);
	descriptorExtractor->compute(imgGray2,keypoints2,descriptors2, mirrDescriptors);


	// matching
	std::vector<std::vector<cv::DMatch> > matches;
	cv::Ptr<cv::DescriptorMatcher> descriptorMatcher = new cv::BFMatcher(cv::NORM_HAMMING);
	descriptorMatcher->radiusMatch(descriptors2,descriptors1,matches,100.0);


	// Show the matching results
	cv::Mat outimg;
	drawMatches(imgRGB2, keypoints2, imgRGB1, keypoints1, matches, outimg, cv::Scalar(0,255,0), cv::Scalar(0,0,255), std::vector<std::vector<char> >(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("Matches");
	cv::imshow("Matches", outimg);
	cv::waitKey();

	return 0;
}
