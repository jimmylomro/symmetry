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

*/

#ifndef _DBSCAN_H_
#define _DBSCAN_H_


#include <opencv2/core/core.hpp>


class  DBSCAN {
public:

	DBSCAN(double eps = 0.05, int minPts = 50);
	~DBSCAN();

	void cluster(const cv::Mat& src, cv::Mat& dst);
	
	void setEpsilon(double eps) {epsilon = eps;}
	void setMinPts(int m) {minPts = m;}
	
private:

	double epsilon;
	int    minPts;

	void pairWiseDist(const cv::Mat& input, cv::Mat& pwd);
	void regionQuery(cv::Mat& pwd, int sampIdx, std::vector<int>& newNeighbors);
	void expandCluster(cv::Mat& dst, cv::Mat& pwd, int sampIdx, int C, std::vector<int>& neighbors, cv::Mat& visited, cv::Mat& isnoise);

};


#endif // _DBSCAN_H_
