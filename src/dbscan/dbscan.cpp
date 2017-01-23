/*
    Jaime Lomeli-R. Univesity of Southampton

    This file is part of SYMMETRY.

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


	This code implements the DBSCAN algorithm as presented by Ester et al. in
	"A density-based algorithm for discovering clusters in large spatial databases with noise"

	NOTE: THIS CODE IS NOT VERY SAFE AS IT DOES NOT CHECK FOR PARAMETERS OR INPUT MATRIX TO BE OF THE CORRECT TYPE OR
	FORMAT, BEHAVIOUR OF THE CODE WHEN THESE CONDITIONS ARE NOT MET IS UNDEFINED.

*/


#include <opencv2/opencv.hpp>
#include <math.h>

#include <dbscan.h>


DBSCAN::DBSCAN(double eps, int minPts) {

	this->epsilon = eps;
	this->minPts  = minPts;
}


void DBSCAN::pairWiseDist(const cv::Mat& src, cv::Mat& pwd) {

    pwd = cv::Mat(src.rows, src.rows, CV_64FC1);
    
    const double *rowPtr, *colPtr;
    double acum, aux;
    
    for (int row = 0; row < src.rows; row++) {
        rowPtr = src.ptr<double>(row);
        for (int col = 0; col < src.rows; col++) {
            colPtr = src.ptr<double>(col);
            acum   = 0;
            for (int dim = 0; dim < src.cols; dim++){
                aux   = rowPtr[dim] - colPtr[dim];
                acum += aux * aux;
            }
            pwd.at<double>(row,col) = sqrt(acum);
        }
    }
}


void DBSCAN::regionQuery(cv::Mat& pwd, int sampIdx, std::vector<int>& newNeighbors) {
    
    const double *rowPtr = pwd.ptr<double>(sampIdx);
    
    newNeighbors.clear();
    
    for (int col = 0; col < pwd.cols; col++) {
        if (rowPtr[col] <= epsilon)
            newNeighbors.push_back(col);
    }
}


void DBSCAN::expandCluster(cv::Mat &dst, cv::Mat &pwd, int sampIdx, int C, std::vector<int> &neighbors, cv::Mat &visited, cv::Mat &isnoise) {
    
    dst.at<char>(sampIdx) = C;
    
    int neighIdx = 0;
    int j;
    
    std::vector<int> newNeighbors;
    newNeighbors.reserve(500);
    
    while (neighIdx < neighbors.size()) {
        j = neighbors[neighIdx];
        
        if (visited.at<char>(j) == 0) {
            visited.at<char>(j) = 1;
            regionQuery(pwd, j, newNeighbors);
            if (newNeighbors.size() >= minPts) {
                    neighbors.insert(neighbors.end(), newNeighbors.begin(), newNeighbors.end());
            }
        }
        
        if (dst.at<char>(j) == 0) {
            dst.at<char>(j) = C;
        }
        
        neighIdx++;
    }
}


void DBSCAN::cluster(const cv::Mat& src, cv::Mat& dst) {
    
    cv::Mat pwd;
    pairWiseDist(src, pwd);
    
    int C = 0;
    int n = src.rows;
    
    dst = cv::Mat::zeros(n,1,CV_8UC1);
    
    cv::Mat visited = cv::Mat::zeros(n,1,CV_8UC1);
    cv::Mat isnoise = cv::Mat::zeros(n,1,CV_8UC1);
    
    std::vector<int> neighbors;
    neighbors.reserve(1000);
    
    for (int i = 0; i < n; i++){
        if (visited.at<char>(i) == 0) {
            visited.at<char>(i) = 1;
            
            regionQuery(pwd, i, neighbors);
            
            if (neighbors.size() < minPts) {
                isnoise.at<char>(i) = 1;
            }
            else {
                C++;
                expandCluster(dst, pwd, i, C, neighbors, visited, isnoise);
            }
        }
    }
}


DBSCAN::~DBSCAN(){
}
