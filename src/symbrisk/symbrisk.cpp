/*
    BRISK - Binary Robust Invariant Scalable Keypoints
    Reference implementation of
    [1] Stefan Leutenegger,Margarita Chli and Roland Siegwart, BRISK:
    	Binary Robust Invariant Scalable Keypoints, in Proceedings of
    	the IEEE International Conference on Computer Vision (ICCV2011).

    Copyright (C) 2011  The Autonomous Systems Lab (ASL), ETH Zurich,
    Stefan Leutenegger, Simon Lynen and Margarita Chli.

    This file is part of BRISK.

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

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <stdlib.h>
#include <tmmintrin.h>

#include <symbrisk.h>


namespace cv{
	// this is needed to avoid aliasing issues with the __m128i data type:
#ifdef __GNUC__
	typedef unsigned char __attribute__ ((__may_alias__)) UCHAR_ALIAS;
	typedef unsigned short __attribute__ ((__may_alias__)) UINT16_ALIAS;
	typedef unsigned int __attribute__ ((__may_alias__)) UINT32_ALIAS;
	typedef unsigned long int __attribute__ ((__may_alias__)) UINT64_ALIAS;
	typedef int __attribute__ ((__may_alias__)) INT32_ALIAS;
	typedef uint8_t __attribute__ ((__may_alias__)) U_INT8T_ALIAS;
#endif
#ifdef _MSC_VER
	// Todo: find the equivalent to may_alias
	#define UCHAR_ALIAS unsigned char //__declspec(noalias)
	#define UINT32_ALIAS unsigned int //__declspec(noalias)
	#define __inline__ __forceinline
#endif
}


using namespace cv;


// some helper structures for the Brisk pattern representation
struct BriskPatternPoint{
	float x;         // x coordinate relative to center
	float y;         // x coordinate relative to center
	float sigma;     // Gaussian smoothing sigma
};
struct BriskShortPair{
	unsigned int i;  // index of the first pattern point
	unsigned int j;  // index of other pattern point
};
struct BriskLongPair{
	unsigned int i;  // index of the first pattern point
	unsigned int j;  // index of other pattern point
	int weighted_dx; // 1024.0/dx
	int weighted_dy; // 1024.0/dy
};


//---------------------------------------------------------------------------------------------------------------------
//-----------------------------BriskDescriptorExtractor----------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------
// Implementation class

class CV_EXPORTS BriskDescriptorExtractor : public brisk::symBriskExtractor{
public:
	// create a descriptor with standard pattern
	BriskDescriptorExtractor(bool rotationInvariant=true, bool scaleInvariant=true, float patternScale=1.0f);

	~BriskDescriptorExtractor();

	// call this to generate the kernel:
	// circle of radius r (pixels), with n points;
	// short pairings with dMax, long pairings with dMin
	void generateKernel(std::vector<float> &radiusList,
		std::vector<int> &numberList, float dMax=5.85f, float dMin=8.2f,
		std::vector<int> indexChange=std::vector<int>());

	// TODO: implement read and write functions
	//virtual void read( const cv::FileNode& );
	//virtual void write( cv::FileStorage& ) const;

	int descriptorSize() const;
	int descriptorType() const;

	bool rotationInvariance;
	bool scaleInvariance;

	void computeAngles(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);

	// Wrapper compute function, this will call the protected computeImpl.
	void compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
		computeImpl(image,keypoints,descriptors);
	}
	

protected:

	// this is the subclass keypoint computation implementation:
	void computeImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
			cv::Mat& descriptors) const;

	__inline__ int smoothedIntensity(const cv::Mat& image,
			const cv::Mat& integral,const float key_x,
				const float key_y, const unsigned int scale,
				const unsigned int rot, const unsigned int point) const;
	// pattern properties
	BriskPatternPoint* patternPoints_; 	//[i][rotation][scale]
	unsigned int points_; 				// total number of collocation points
	float* scaleList_; 					// lists the scaling per scale index [scale]
	unsigned int* sizeList_; 			// lists the total pattern size per scale index [scale]
	static const unsigned int scales_;	// scales discretization
	static const float scalerange_; 	// span of sizes 40->4 Octaves - else, this needs to be adjusted...
	static const unsigned int n_rot_;	// discretization of the rotation look-up

	// pairs
	int strings_;						// number of uchars the descriptor consists of
	float dMax_; 						// short pair maximum distance
	float dMin_; 						// long pair maximum distance
	BriskShortPair* shortPairs_; 		// d<_dMax
	BriskLongPair* longPairs_; 			// d>_dMin
	unsigned int noShortPairs_; 		// number of shortParis
	unsigned int noLongPairs_; 			// number of longParis

	// general
	static const float basicSize_;
};


// Set constants
const float BriskDescriptorExtractor::basicSize_    =12.0;
const unsigned int BriskDescriptorExtractor::scales_=64;
const float BriskDescriptorExtractor::scalerange_   =30;        // 40->4 Octaves - else, this needs to be adjusted...
const unsigned int BriskDescriptorExtractor::n_rot_ =1024;	 // discretization of the rotation look-up


// Constructor
BriskDescriptorExtractor::BriskDescriptorExtractor(bool rotationInvariant,
		bool scaleInvariant, float patternScale){

	std::vector<float> rList;
	std::vector<int> nList;

	// this is the standard pattern found to be suitable also
	rList.resize(5);
	nList.resize(5);
	const double f=0.85*patternScale;


	rList[0]=f*10.8;
	rList[1]=f*7.4;
	rList[2]=f*4.9;
	rList[3]=f*2.9;
	rList[4]=f*0;

	nList[0]=20;
	nList[1]=14;
	nList[2]=12;
	nList[3]=10;
	nList[4]=1;

/*	rList[0]=f*0;
	rList[1]=f*2.9;
	rList[2]=f*4.9;
	rList[3]=f*7.4;
	rList[4]=f*10.8;

	nList[0]=1;
	nList[1]=10;
	nList[2]=14;
	nList[3]=15;
	nList[4]=20;
*/
	rotationInvariance=rotationInvariant;
	scaleInvariance=scaleInvariant; //5.85
	generateKernel(rList,nList,6.22*patternScale,8.2*patternScale);

}


void BriskDescriptorExtractor::generateKernel(std::vector<float> &radiusList,
			std::vector<int> &numberList, float dMax, float dMin,
			std::vector<int> indexChange){

	dMax_=dMax;
	dMin_=dMin;

	// get the total number of points
	const int rings=radiusList.size();
	assert(radiusList.size()!=0&&radiusList.size()==numberList.size());
	points_=0; // remember the total number of points
	for(int ring = 0; ring<rings; ring++){
		points_+=numberList[ring];
	}
	assert((points_%2) == 1);	// points is odd for symmetry
	// set up the patterns
	patternPoints_=new BriskPatternPoint[points_*scales_*n_rot_];
	BriskPatternPoint* patternIterator=patternPoints_;

	// define the scale discretization:
	static const float lb_scale=log(scalerange_)/log(2.0);
	static const float lb_scale_step = lb_scale/(scales_);

	scaleList_=new float[scales_];
	sizeList_=new unsigned int[scales_];

	const float sigma_scale=1.3;

	for(unsigned int scale = 0; scale <scales_; ++scale){
		scaleList_[scale]=pow((double)2.0,(double)(scale*lb_scale_step));
		sizeList_[scale]=0;

		// generate the pattern points look-up
		double alpha, half_d_alpha, theta;
		for(size_t rot=0; rot<n_rot_; ++rot){
			theta = double(rot)*2*M_PI/double(n_rot_); // this is the rotation of the feature
			int ring, num;	
			for(ring = 0; ring<rings; ring++){
				half_d_alpha = M_PI/double(numberList[ring]);
				for(num=0; num<ceil(numberList[ring]/2); num++){
					// the actual coordinates on the circle
					alpha = (double(num))*2*M_PI/double(numberList[ring]);
					patternIterator->x=scaleList_[scale]*radiusList[ring]*cos(alpha+half_d_alpha+theta); // feature rotation plus angle of the point
					patternIterator->y=scaleList_[scale]*radiusList[ring]*sin(alpha+half_d_alpha+theta);
					// and the gaussian kernel sigma
					if(ring == rings-1){
						patternIterator->sigma = sigma_scale*scaleList_[scale]*0.5;
					}
					else{
						patternIterator->sigma = sigma_scale*scaleList_[scale]*(double(radiusList[ring]))*sin(M_PI/numberList[ring]);
					}
					// adapt the sizeList if necessary
					const unsigned int size=ceil(((scaleList_[scale]*radiusList[ring])+patternIterator->sigma))+1;
					if(sizeList_[scale]<size){
						sizeList_[scale]=size;
					}

					// increment the iterator
					++patternIterator;
				}
			}

			
			for(ring = rings-1; ring>=0; ring--){
				half_d_alpha = M_PI/double(numberList[ring]);
				for(num = floor(numberList[ring]/2); num<numberList[ring]; num++){
					// the actual coordinates on the circle
					alpha = (double(num))*2*M_PI/double(numberList[ring]);
					patternIterator->x=scaleList_[scale]*radiusList[ring]*cos(alpha+half_d_alpha+theta); // feature rotation plus angle of the point
					patternIterator->y=scaleList_[scale]*radiusList[ring]*sin(alpha+half_d_alpha+theta);
					// and the gaussian kernel sigma
					if(ring == rings-1){
						patternIterator->sigma = sigma_scale*scaleList_[scale]*0.5;
					}
					else{
						patternIterator->sigma = sigma_scale*scaleList_[scale]*(double(radiusList[ring]))*sin(M_PI/numberList[ring]);
					}
					// adapt the sizeList if necessary
					const unsigned int size=ceil(((scaleList_[scale]*radiusList[ring])+patternIterator->sigma))+1;
					if(sizeList_[scale]<size){
						sizeList_[scale]=size;
					}

					// increment the iterator
					++patternIterator;
				}
			}
		}
	}

	// now also generate pairings
	shortPairs_ = new BriskShortPair[points_*(points_-1)/2];
	longPairs_ = new BriskLongPair[points_*(points_-1)/2];
	noShortPairs_=0;
	noLongPairs_=0;

	// fill indexChange with 0..n if empty
	unsigned int indSize=indexChange.size();
	if(indSize==0) {
		indexChange.resize(points_*(points_-1)/2);
		indSize=indexChange.size();
	}
	for(unsigned int i=0; i<indSize; i++){
		indexChange[i]=i;
	}
	const float dMin_sq =dMin_*dMin_;
	const float dMax_sq =dMax_*dMax_;
	
	for(unsigned int i= 0; i<(points_-1)/2; i++){
		for(unsigned int j= i+1; j<(points_-i); j++){ 				// find pairs of the first half
			// point pair distance:
			const float dx=patternPoints_[j].x-patternPoints_[i].x;
			const float dy=patternPoints_[j].y-patternPoints_[i].y;
			const float norm_sq=(dx*dx+dy*dy);
			if(norm_sq>dMin_sq){
				// save to long pairs
				BriskLongPair& longPair=longPairs_[noLongPairs_];
				longPair.weighted_dx=int((dx/(norm_sq))*2048.0+0.5);
				longPair.weighted_dy=int((dy/(norm_sq))*2048.0+0.5);
				longPair.i = i;
				longPair.j = j;
				++noLongPairs_;
			}
			else if (norm_sq<dMax_sq){
				// save to short pairs
				assert(noShortPairs_<indSize); // make sure the user passes something sensible
				BriskShortPair& shortPair = shortPairs_[indexChange[noShortPairs_]];
				shortPair.j = j;
				shortPair.i = i;
				++noShortPairs_;
			}
		}
	}

	
	shortPairs_[noShortPairs_].j   = (points_-1)/2;
	shortPairs_[noShortPairs_++].i = (points_-1)/2;
	shortPairs_[noShortPairs_].j   = (points_-1)/2;
	shortPairs_[noShortPairs_++].i = (points_-1)/2;


	for(unsigned int i=(points_+1)/2; i<points_; i++){
		for(unsigned int j= points_-i-1; j<i; j++){ 				// find pairs of the second half
			// point pair distance:
			const float dx=patternPoints_[j].x-patternPoints_[i].x;
			const float dy=patternPoints_[j].y-patternPoints_[i].y;
			const float norm_sq=(dx*dx+dy*dy);
			if(norm_sq>dMin_sq){
				// save to long pairs
				BriskLongPair& longPair=longPairs_[noLongPairs_];
				longPair.weighted_dx=int((dx/(norm_sq))*2048.0+0.5);
				longPair.weighted_dy=int((dy/(norm_sq))*2048.0+0.5);
				longPair.i = i;
				longPair.j = j;
				++noLongPairs_;
			}
			else if (norm_sq<dMax_sq){
				// save to short pairs
				assert(noShortPairs_<indSize); // make sure the user passes something sensible
				BriskShortPair& shortPair = shortPairs_[indexChange[noShortPairs_]];
				shortPair.j = j;
				shortPair.i = i;
				++noShortPairs_;
			}
		}
	}


	//for (unsigned int r = 0; r<noShortPairs_; r++) std::cout << shortPairs_[r].i << "," << shortPairs_[r].j << std::endl;
	//for (unsigned int i = 0; i<points_; i++) std::cout << patternPoints_[i].x << "," << patternPoints_[i].y << std::endl;

	// no bits:
	strings_=(int)ceil((float(noShortPairs_))/8.0);

}


// simple alternative:
__inline__ int BriskDescriptorExtractor::smoothedIntensity(const cv::Mat& image,
		const cv::Mat& integral,const float key_x,
			const float key_y, const unsigned int scale,
			const unsigned int rot, const unsigned int point) const{

	// get the float position
	const BriskPatternPoint& briskPoint = patternPoints_[scale*n_rot_*points_ + rot*points_ + point];
	const float xf=briskPoint.x+key_x;
	const float yf=briskPoint.y+key_y;
	const int x = int(xf);
	const int y = int(yf);
	const int& imagecols=image.cols;

	// get the sigma:
	const float sigma_half=briskPoint.sigma;
	const float area=4.0*sigma_half*sigma_half;

	// calculate output:
	int ret_val;
	if(sigma_half<0.5){
		//interpolation multipliers:
		const int r_x=(xf-x)*1024;
		const int r_y=(yf-y)*1024;
		const int r_x_1=(1024-r_x);
		const int r_y_1=(1024-r_y);
		uchar* ptr=image.data+x+y*imagecols;
		// just interpolate:
		ret_val=(r_x_1*r_y_1*int(*ptr));
		ptr++;
		ret_val+=(r_x*r_y_1*int(*ptr));
		ptr+=imagecols;
		ret_val+=(r_x*r_y*int(*ptr));
		ptr--;
		ret_val+=(r_x_1*r_y*int(*ptr));
		return (ret_val+512)/1024;
	}

	// this is the standard case (simple, not speed optimized yet):

	// scaling:
	const int scaling = 4194304.0/area;
	const int scaling2=float(scaling)*area/1024.0;

	// the integral image is larger:
	const int integralcols=imagecols+1;

	// calculate borders
	const float x_1=xf-sigma_half;
	const float x1=xf+sigma_half;
	const float y_1=yf-sigma_half;
	const float y1=yf+sigma_half;

	const int x_left=int(x_1+0.5);
	const int y_top=int(y_1+0.5);
	const int x_right=int(x1+0.5);
	const int y_bottom=int(y1+0.5);

	// overlap area - multiplication factors:
	const float r_x_1=float(x_left)-x_1+0.5;
	const float r_y_1=float(y_top)-y_1+0.5;
	const float r_x1=x1-float(x_right)+0.5;
	const float r_y1=y1-float(y_bottom)+0.5;
	const int dx=x_right-x_left-1;
	const int dy=y_bottom-y_top-1;
	const int A=(r_x_1*r_y_1)*scaling;
	const int B=(r_x1*r_y_1)*scaling;
	const int C=(r_x1*r_y1)*scaling;
	const int D=(r_x_1*r_y1)*scaling;
	const int r_x_1_i=r_x_1*scaling;
	const int r_y_1_i=r_y_1*scaling;
	const int r_x1_i=r_x1*scaling;
	const int r_y1_i=r_y1*scaling;

	if(dx+dy>2){
		// now the calculation:
		uchar* ptr=image.data+x_left+imagecols*y_top;
		// first the corners:
		ret_val=A*int(*ptr);
		ptr+=dx+1;
		ret_val+=B*int(*ptr);
		ptr+=dy*imagecols+1;
		ret_val+=C*int(*ptr);
		ptr-=dx+1;
		ret_val+=D*int(*ptr);

		// next the edges:
		int* ptr_integral=(int*)integral.data+x_left+integralcols*y_top+1;
		// find a simple path through the different surface corners
		const int tmp1=(*ptr_integral);
		ptr_integral+=dx;
		const int tmp2=(*ptr_integral);
		ptr_integral+=integralcols;
		const int tmp3=(*ptr_integral);
		ptr_integral++;
		const int tmp4=(*ptr_integral);
		ptr_integral+=dy*integralcols;
		const int tmp5=(*ptr_integral);
		ptr_integral--;
		const int tmp6=(*ptr_integral);
		ptr_integral+=integralcols;
		const int tmp7=(*ptr_integral);
		ptr_integral-=dx;
		const int tmp8=(*ptr_integral);
		ptr_integral-=integralcols;
		const int tmp9=(*ptr_integral);
		ptr_integral--;
		const int tmp10=(*ptr_integral);
		ptr_integral-=dy*integralcols;
		const int tmp11=(*ptr_integral);
		ptr_integral++;
		const int tmp12=(*ptr_integral);

		// assign the weighted surface integrals:
		const int upper=(tmp3-tmp2+tmp1-tmp12)*r_y_1_i;
		const int middle=(tmp6-tmp3+tmp12-tmp9)*scaling;
		const int left=(tmp9-tmp12+tmp11-tmp10)*r_x_1_i;
		const int right=(tmp5-tmp4+tmp3-tmp6)*r_x1_i;
		const int bottom=(tmp7-tmp6+tmp9-tmp8)*r_y1_i;

		return (ret_val+upper+middle+left+right+bottom+scaling2/2)/scaling2;
	}

	// now the calculation:
	uchar* ptr=image.data+x_left+imagecols*y_top;
	// first row:
	ret_val=A*int(*ptr);
	ptr++;
	const uchar* end1 = ptr+dx;
	for(; ptr<end1; ptr++){
		ret_val+=r_y_1_i*int(*ptr);
	}
	ret_val+=B*int(*ptr);
	// middle ones:
	ptr+=imagecols-dx-1;
	uchar* end_j=ptr+dy*imagecols;
	for(; ptr<end_j; ptr+=imagecols-dx-1){
		ret_val+=r_x_1_i*int(*ptr);
		ptr++;
		const uchar* end2 = ptr+dx;
		for(; ptr<end2; ptr++){
			ret_val+=int(*ptr)*scaling;
		}
		ret_val+=r_x1_i*int(*ptr);
	}
	// last row:
	ret_val+=D*int(*ptr);
	ptr++;
	const uchar* end3 = ptr+dx;
	for(; ptr<end3; ptr++){
		ret_val+=r_y1_i*int(*ptr);
	}
	ret_val+=C*int(*ptr);

	return (ret_val+scaling2/2)/scaling2;
}


bool RoiPredicate(const float minX, const float minY,
		const float maxX, const float maxY, const KeyPoint& keyPt){
	const Point2f& pt = keyPt.pt;
	return (pt.x < minX) || (pt.x >= maxX) || (pt.y < minY) || (pt.y >= maxY);
}


void BriskDescriptorExtractor::computeAngles(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints) {
	//Remove keypoints very close to the border
	size_t ksize=keypoints.size();
	std::vector<int> kscales; // remember the scale per keypoint
	kscales.resize(ksize);
	static const float log2 = 0.693147180559945;
	static const float lb_scalerange = log(scalerange_)/(log2);
	std::vector<cv::KeyPoint>::iterator beginning = keypoints.begin();
	std::vector<int>::iterator beginningkscales = kscales.begin();
	static const float basicSize06=basicSize_*0.6;
	unsigned int basicscale=0;
	if(!scaleInvariance)
		basicscale=std::max((int)(scales_/lb_scalerange*(log(1.45*basicSize_/(basicSize06))/log2)+0.5),0);
	for(size_t k=0; k<ksize; k++){
		unsigned int scale;
		if(scaleInvariance){
			scale=std::max((int)(scales_/lb_scalerange*(log(keypoints[k].size/(basicSize06))/log2)+0.5),0);
			// saturate
			if(scale>=scales_) scale = scales_-1;
			kscales[k]=scale;
		}
		else{
			scale = basicscale;
			kscales[k]=scale;
		}
		const int border = sizeList_[scale];
		const int border_x=image.cols-border;
		const int border_y=image.rows-border;
		if(RoiPredicate(border, border,border_x,border_y,keypoints[k])){
			keypoints.erase(beginning+k);
			kscales.erase(beginningkscales+k);
			if(k==0){
				beginning=keypoints.begin();
				beginningkscales = kscales.begin();
			}
			ksize--;
			k--;
		}
	}

	// first, calculate the integral image over the whole image:
	// current integral image
	cv::Mat _integral; // the integral image
	cv::integral(image, _integral);

	int* _values=new int[points_]; // for temporary use

	// now do the extraction for all keypoints:

	// temporary variables containing gray values at sample points:
	int t1;
	int t2;

	// the feature orientation
	int direction0;
	int direction1;

	for(size_t k=0; k<ksize; k++){
		int theta;
		cv::KeyPoint& kp=keypoints[k];
		const int& scale=kscales[k];
		int shifter=0;
		int* pvalues =_values;
		const float& x=kp.pt.x;
		const float& y=kp.pt.y;
		if(true/*kp.angle==-1*/){
			if (!rotationInvariance){
				// don't compute the gradient direction, just assign a rotation of 0°
				theta=0;
			}
			else{
				// get the gray values in the unrotated pattern
				for(unsigned int i = 0; i<points_; i++){
					*(pvalues++)=smoothedIntensity(image, _integral, x,
							y, scale, 0, i);
				}

				direction0=0;
				direction1=0;
				// now iterate through the long pairings
				const BriskLongPair* max=longPairs_+noLongPairs_;
				for(BriskLongPair* iter=longPairs_; iter<max; ++iter){
					t1=*(_values+iter->i);
					t2=*(_values+iter->j);
					const int delta_t=(t1-t2);
					// update the direction:
					const int tmp0=delta_t*(iter->weighted_dx)/1024;
					const int tmp1=delta_t*(iter->weighted_dy)/1024;
					direction0+=tmp0;
					direction1+=tmp1;
				}
				kp.angle=atan2((float)direction1,(float)direction0)/M_PI*180.0;
				theta=int((n_rot_*kp.angle)/(360.0)+0.5);
				if(theta<0)
					theta+=n_rot_;
				if(theta>=int(n_rot_))
					theta-=n_rot_;
			}
		}
	}
}


// computes the descriptor
void BriskDescriptorExtractor::computeImpl(const Mat& image,
		std::vector<KeyPoint>& keypoints, Mat& descriptors) const{

	//Remove keypoints very close to the border
	size_t ksize=keypoints.size();
	std::vector<int> kscales; // remember the scale per keypoint
	kscales.resize(ksize);
	static const float log2 = 0.693147180559945;
	static const float lb_scalerange = log(scalerange_)/(log2);
	std::vector<cv::KeyPoint>::iterator beginning = keypoints.begin();
	std::vector<int>::iterator beginningkscales = kscales.begin();
	static const float basicSize06=basicSize_*0.6;
	unsigned int basicscale=0;
	if(!scaleInvariance)
		basicscale=std::max((int)(scales_/lb_scalerange*(log(1.45*basicSize_/(basicSize06))/log2)+0.5),0);
	for(size_t k=0; k<ksize; k++){
		unsigned int scale;
		if(scaleInvariance){
			scale=std::max((int)(scales_/lb_scalerange*(log(keypoints[k].size/(basicSize06))/log2)+0.5),0);
			// saturate
			if(scale>=scales_) scale = scales_-1;
			kscales[k]=scale;
		}
		else{
			scale = basicscale;
			kscales[k]=scale;
		}
		const int border = sizeList_[scale];
		const int border_x=image.cols-border;
		const int border_y=image.rows-border;
		if(RoiPredicate(border, border,border_x,border_y,keypoints[k])){
			keypoints.erase(beginning+k);
			kscales.erase(beginningkscales+k);
			if(k==0){
				beginning=keypoints.begin();
				beginningkscales = kscales.begin();
			}
			ksize--;
			k--;
		}
	}

	// first, calculate the integral image over the whole image:
	// current integral image
	cv::Mat _integral; // the integral image
	cv::integral(image, _integral);

	int* _values=new int[points_]; // for temporary use

	// resize the descriptors:
	descriptors=cv::Mat::zeros(ksize,strings_, CV_8U);

	// now do the extraction for all keypoints:

	// temporary variables containing gray values at sample points:
	int t1;
	int t2;

	// the feature orientation
	int direction0;
	int direction1;

	uchar* ptr = descriptors.data;
	for(size_t k=0; k<ksize; k++){
		int theta;
		cv::KeyPoint& kp=keypoints[k];
		const int& scale=kscales[k];
		int shifter=0;
		int* pvalues =_values;
		const float& x=kp.pt.x;
		const float& y=kp.pt.y;
		if(true/*kp.angle==-1*/){
			if (!rotationInvariance){
				// don't compute the gradient direction, just assign a rotation of 0°
				theta=0;
			}
			else{
				// get the gray values in the unrotated pattern
				for(unsigned int i = 0; i<points_; i++){
					*(pvalues++)=smoothedIntensity(image, _integral, x,
							y, scale, 0, i);
				}

				direction0=0;
				direction1=0;
				// now iterate through the long pairings
				const BriskLongPair* max=longPairs_+noLongPairs_;
				for(BriskLongPair* iter=longPairs_; iter<max; ++iter){
					t1=*(_values+iter->i);
					t2=*(_values+iter->j);
					const int delta_t=(t1-t2);
					// update the direction:
					const int tmp0=delta_t*(iter->weighted_dx)/1024;
					const int tmp1=delta_t*(iter->weighted_dy)/1024;
					direction0+=tmp0;
					direction1+=tmp1;
				}
				kp.angle=atan2((float)direction1,(float)direction0)/M_PI*180.0;
				theta=int((n_rot_*kp.angle)/(360.0)+0.5);
				if(theta<0)
					theta+=n_rot_;
				if(theta>=int(n_rot_))
					theta-=n_rot_;
			}
		}
		else{
			// figure out the direction:
			//int theta=rotationInvariance*round((_n_rot*atan2(direction.at<int>(0,0),direction.at<int>(1,0)))/(2*M_PI));
			if(!rotationInvariance){
				theta=0;
			}
			else{
				theta=(int)(n_rot_*(kp.angle/(360.0))+0.5);
				if(theta<0)
					theta+=n_rot_;
				if(theta>=int(n_rot_))
					theta-=n_rot_;
			}
		}

		// now also extract the stuff for the actual direction:
		// let us compute the smoothed values
		shifter=0;

		//unsigned int mean=0;
		pvalues =_values;
		// get the gray values in the rotated pattern
		for(unsigned int i = 0; i<points_; i++){
			*(pvalues++)=smoothedIntensity(image, _integral, x,
					y, scale, theta, i);
		}

		// now iterate through all the pairings
		UINT32_ALIAS* ptr2=(UINT32_ALIAS*)ptr;
		const BriskShortPair* max=shortPairs_+noShortPairs_;
		for(BriskShortPair* iter=shortPairs_; iter<max;++iter){
			t1=*(_values+iter->i);
			t2=*(_values+iter->j);
			if(t1>t2){
				*ptr2|=((1)<<shifter);

			} // else already initialized with zero
			// take care of the iterators:
			++shifter;
			if(shifter==32){
				shifter=0;
				++ptr2;
			}
		}

		ptr+=strings_;
	}

	// clean-up
	_integral.release();
	delete [] _values;
}


// Getters
int BriskDescriptorExtractor::descriptorSize() const{
	return strings_;
}


int BriskDescriptorExtractor::descriptorType() const{
	return CV_8U;
}


// Destructor
BriskDescriptorExtractor::~BriskDescriptorExtractor(){
	delete [] patternPoints_;
	delete [] shortPairs_;
	delete [] longPairs_;
	delete [] scaleList_;
	delete [] sizeList_;
}




//---------------------------------------------------------------------------------------------------------------------
//-----------------------------symBriskExtractor-----------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------
// This is the wrapper class


cv::Ptr<brisk::symBriskExtractor> brisk::symBriskExtractor::create(bool rotationInvariant, bool scaleInvariant, float patternScale) {

    return cv::makePtr<BriskDescriptorExtractor>(rotationInvariant, scaleInvariant, patternScale);
}

