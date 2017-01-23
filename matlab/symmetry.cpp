/*=================================================================
 *
 * BRISK.C  .MEX interface to the BRISK C++ library
 *	    Detects, extracts and matches BRISK features
 *          Implementation according to 
 *
 *      [1] Stefan Leutenegger, 
 *          Margarita Chli and Roland Siegwart, BRISK: Binary 
 *          Robust Invariant Scalable Keypoints, in Proceedings of 
 *          the IEEE International Conference on Computer Vision 
 *          (ICCV) 2011.
 *
 * The calling syntax is:
 *
 *	varargout = brisk(subfunction, morevarargin)
 *
 *      where subfunction is to be used in order:
 *
 *      'init'        Initialize brisk. Optionally pass arguments to 
 *                    set properties (see below). 
 *                    Attention: this will create the pattern look-up table,
 *                    so this may take some fraction of a second. 
 *                    Do not rerun!
 *
 *      'set'         Set properties. The following may be set:
 *                    '-threshold'    FAST/AGAST detection threshold.
 *                                  The default value is 60.
 *                    '-kernSize'      No. kernSize for the detection.
 *                                  The default value is 4.
 *                    '-patternScale' Scale factor for the BRISK pattern.
 *                                  The default value is 1.0.
 *                    '-type'         BRISK special type 'S', 'U', 'SU'.
 *                                  By default, the standard BRISK is used.
 *                                    See [1] for explanations on this.
 *                    Attention: if the patternScale or the type is reset, 
 *                    the pattern will be regenerated, which is time-
 *                    consuming!
 *
 *      'loadImage'   Load an image either from Matlab workspace by passing
 *                    a UINT8 Matrix as a second argument, or by specifying 
 *                    a path to an image:
 *                        brisk('loadImage',imread('path/to/image'));
 *                        brisk('loadImage','path/to/image');
 *
 *      'detect'      Detect the keypoints. Optionally get the points back:
 *                      brisk('detect');
 *                      keyPoints=brisk('detect');
 *
 *      'describe'    Get the descriptors and the corresponding keypoints
 *                      [keyPoints,descriptors]=brisk('detect');
 *
 *      'radiusMatch' Radius match.
 *                      [indicesOfSecondKeyPoints]=brisk('radiusMatch',...
 *                          firstKeypoints,secondKeyPoints);
 * 
 *      'knnMatch'    k-nearest neighbor match.
 *                      [indicesOfSecondKeyPoints]=brisk('knnMatch',...
 *                          firstKeypoints,secondKeyPoints,k);
 *
 *      'image'       Returns the currently used gray-scale image
 *                      image=brisk('image');
 *
 *      'terminate'   Free the memory.
 *
 *
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

#include "brisk_interface.h"
#include <cstring>
#include <string>


// test input args
bool mxIsScalarNonComplexDouble(const mxArray* arg){
    mwSize mrows,ncols;
    mrows = mxGetM(arg);
    ncols = mxGetN(arg);
    if( !mxIsDouble(arg) || mxIsComplex(arg) ||
      !(mrows==1 && ncols==1) )
        return false;
    return true;
}


// constructor
symmetryInterface::symmetryInterface( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ){

    // init default parameters
    own_threshold   = 0.5;
    own_kernSize    = 31;
    own_nMaps       = 10;
    
	symbrisk_rotInv		  = true;
    symbrisk_scaleInv	  = true;
    symbrisk_patternScale = 1.0f;
    
    matcher_maskFeats = false;

	// create the matcher object
	matcherPtr = new cv::BFMatcher(cv::NORM_HAMMING);
	dbscan 	   = new DBSCAN();
    
    // set parameters and create objects
    set(nlhs, plhs, nrhs, prhs);
}


symmetryInterface::~symmetryInterface(){

    if(detectorPtr != NULL)
        detectorPtr.release();

    if(descriptorPtr != NULL)
        descriptorPtr.release();
        
    if(matcherPtr != NULL)
        matcherPtr.release();
}


// set parameters parsing inputs
inline void symmetryInterface::set(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    if((nrhs-1)%2!=0) mexErrMsgTxt("Bad input.");
    
    // remember what to re-initialize
    bool initDetector	= false;
    bool initDescriptor	= false;
    bool scaleSet		= false;
    bool initDbscan		= false;
    
    for(int i=1; i<nrhs; i+=2){
    
        // parse option
        char* str2=mxArrayToString(prhs[i]);
        
        if(!(mxGetClassID(prhs[i])==mxCHAR_CLASS))
            mexErrMsgTxt("Bad input.");
            
        if (strcmp(str2,"own_threshold")==0){
            if(!mxIsScalarNonComplexDouble(prhs[i+1]))
                mexErrMsgTxt("Bad input.");
            double* x=mxGetPr(prhs[i+1]);
            // bound
            if(*x<0) own_threshold =0;
            else if(*x>1) own_threshold = 1;
            else own_threshold = float(*x);
            initDetector = true;
        }
        
        else if (strcmp(str2,"own_nMaps")==0){
            if(!mxIsScalarNonComplexDouble(prhs[i+1]))
                mexErrMsgTxt("Bad input.");
            double* x=mxGetPr(prhs[i+1]);
            if(*x<0) own_nMaps = 0;
            else if(*x>10) own_nMaps = 10;
            else own_nMaps = int(*x);
            initDetector = true;
        }
        
        else if (strcmp(str2,"own_kernSize")==0){
            if(!mxIsScalarNonComplexDouble(prhs[i+1]))
                mexErrMsgTxt("Bad input.");
            double* x=mxGetPr(prhs[i+1]);
            if(*x<0) own_kernSize = 0;
            else if(*x>128) own_kernSize = 128;
            else own_kernSize=int(*x);
            initDetector = true;
        }
        
        else if (strcmp(str2,"symbrisk_patternScale")==0){
            if(!mxIsScalarNonComplexDouble(prhs[i+1]))
                mexErrMsgTxt("Bad input.");
            double* x=mxGetPr(prhs[i+1]);
            if(*x<0) symbrisk_patternScale = 1.0;
            else symbrisk_patternScale = *x;
            scaleSet = true;
            initDescriptor = true;
        }
        
        else if (strcmp(str2,"matcher_maskFeats")==0) {
            if(!(mxGetClassID(prhs[i+1])==mxLOGICAL_CLASS))
                mexErrMsgTxt("Bad input.");
            mxLogical* data=mxGetLogicals(prhs[i+1]);
            matcher_maskFeats = (bool)(*data);
            initDetector=true;
        }
        
        else if (strcmp(str2,"dbscan_epsilon")==0){
            if(!mxIsScalarNonComplexDouble(prhs[i+1]))
                mexErrMsgTxt("Bad input.");
            double* x=mxGetPr(prhs[i+1]);
            // bound
            if(*x<0) dbscan_epsilon =0;
            else if(*x>1) dbscan_epsilon = 1;
            else dbscan_epsilon = float(*x);
            initDbscan = true;
        }
        
        else if (strcmp(str2,"dbscan_minPts")==0){
            if(!mxIsScalarNonComplexDouble(prhs[i+1]))
                mexErrMsgTxt("Bad input.");
            double* x=mxGetPr(prhs[i+1]);
            if(*x<0) own_kernSize = 0;
            else own_kernSize=int(*x);
            initDbscan = true;
        }
        
        else mexErrMsgTxt("Unrecognized input option.");
    }
    
    // reset detector and descriptor
    if(initDetector || detectorPtr == NULL) {
        if(detectorPtr != NULL)
            detectorPtr.release();
        
        detector = own::OwnFeatureDetector::create(own_threshold, 8, own_nMaps, own_kernSize);
    }
    
    if(initDescriptor || descriptorPtr == NULL) {
        if(descriptorPtr != NULL)
            descriptorPtr.release();
            
        descriptorPtr = brisk::symBriskExtractor::create(symbrisk_rotInv, symbrisk_scaleInv, symbrisk_patternScale);
    }
    
    if(initDbscan){
    	dbscan->setEpsilon(dbscan_epsilon);
    	dbscan->setMinPts(dbscan_minPts);
    }
}


// load an image
inline void symmetryInterface::loadImage( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {

    if(nrhs<2) 
        mexErrMsgTxt("No image passed.");
    if((mxGetClassID(prhs[1])==mxUINT8_CLASS)){
        // image dimensions
        int M=mxGetM(prhs[1]);
        int N=mxGetN(prhs[1]);
        mwSize dim=mxGetNumberOfDimensions(prhs[1]);
        if(dim==3){
            // this means we need to merge the channels.
            uchar *data = (uchar*) mxGetData(prhs[1]);
            std::vector<cv::Mat> BGR;
            BGR.push_back(cv::Mat(N/3, M, CV_8U, data+2*N*M/3));
            BGR.push_back(cv::Mat(N/3, M, CV_8U, data+N*M/3));
            BGR.push_back(cv::Mat(N/3, M, CV_8U, data));

            // merge into one BGR matrix
            cv::Mat imageBGR;
            cv::merge(BGR,imageBGR);
            // color conversion
            cv::cvtColor(imageBGR,img, CV_BGR2GRAY);
            
            // transpose
            img=img.t();
        }
        else if(dim==2){// cast image to a cv::Mat
            uchar* data = (uchar*) mxGetData(prhs[1]); 
            img=cv::Mat(N, M, CV_8U, data);
            
            // transpose
            img=img.t();
        }
        else mexErrMsgTxt("Image dimension must be 2 or 3.");
    }
    else if((mxGetClassID(prhs[1])==mxCHAR_CLASS)){
        char* fname = mxArrayToString(prhs[1]);
        img=cv::imread(fname,0); // forcing gray
        if(img.data==0){
            mexPrintf("%s ",fname);
            mexErrMsgTxt("Image could not be loaded.");
        }
        //mexMakeMemoryPersistent(&img);
    }
    else
        mexErrMsgTxt("Pass an UINT8_T image matrix or a path.");
}


// detection
inline void symmetryInterface::detect( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {

    if(img.empty()) 
            mexErrMsgTxt("Currently no image loaded.");
        
    // actual detection step
    assert(detectorPtr == NULL);
    detectorPtr->detect(img, keypoints);


	int N = keypoints.size();
    if (maskFeats) {
		matchingMask = cv::Mat::zeros(N,N,CV_8UC1);
		
        int y0      = 0;
        int past_id = keypoints[y0].class_id;
        for (int y1 = 0; y1 < N; y1++) {
            if (keypoints[y1].class_id != past_id) {
                cv::rectangle(matchingMask, cv::Point(y0,y0), cv::Point(y1-1,y1-1), cv::Scalar(255), CV_FILLED);
                y0 = y1;
                past_id = keypoints[y1].class_id;
            }
        }
        // draw last rectangle
        cv::rectangle(matchingMask, cv::Point(y0,y0), cv::Point(N-1,N-1), cv::Scalar(255), CV_FILLED);
    }
    else {
    	matchingMask = cv::Mat::ones(N,N,CV_8UC1) * 255;
    }


    // send the keypoints to the user, if he wants it
    // allocate plhs
    if(nlhs>=1){
        const int keypoint_size=keypoints.size();
        mxArray* tmp;
        tmp=mxCreateDoubleMatrix(6,keypoint_size,mxREAL);
        double *ptr=mxGetPr(tmp);
        // fill it - attention: in Matlab, memory is transposed...
        for(int k=0; k<keypoint_size; k++){
            const int k6=6*k;
            ptr[k6]=keypoints[k].pt.x;
            ptr[k6+1]=keypoints[k].pt.y;
            ptr[k6+2]=keypoints[k].size;
            ptr[k6+3]=-1;
            ptr[k6+4]=keypoints[k].class_id;
            ptr[k6+5]=keypoints[k].response;
        }
        
        // finally, re-transpose for better readibility:
        mexCallMATLAB(1, plhs, 1, &tmp, "transpose");
    }
}


// description
inline void symmetryInterface::describe(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

    // in this case, the user is forced to pass two lhs args
    if(nlhs!=3) 
        mexErrMsgTxt("Three left-hand side arguments must be passed.");
    if(img.empty())
        mexErrMsgTxt("No image loaded.");

    // check the keypoints
    if(keypoints.size()==0)
        mexErrMsgTxt("Keypoints empty. Run detect.");

    // now we can extract the descriptors
    cv::Mat descriptors, mirrorDescriptors;
    assert(descriptorPtr == NULL);
    descriptorPtr->compute(img, keypoints, descriptors, mirrorDescriptors);

    // allocate the lhs mirror descriptor matrix
    int dim[2];
    dim[0] = descriptorPtr->descriptorSize();
    dim[1] = keypoints.size();
    mxArray* tmp2=mxCreateNumericArray(2,dim,mxUINT8_CLASS,mxREAL);
    uchar* data = (uchar*) mxGetData(tmp2); 
    // copy - kind of dumb, but necessary due to the matlab memory 
    // management
    memcpy(data,mirrorDescriptors.data,dim[0]*dim[1]);
    // transpose for better readibility
    mexCallMATLAB(1, &plhs[2], 1, &tmp2, "transpose");

    // allocate the lhs descriptor matrix
    mxArray* tmp1=mxCreateNumericArray(2,dim,mxUINT8_CLASS,mxREAL);
    data = (uchar*) mxGetData(tmp1); 
    // copy - kind of dumb, but necessary due to the matlab memory 
    // management
    memcpy(data,descriptors.data,dim[0]*dim[1]);
    // transpose for better readibility
    mexCallMATLAB(1, &plhs[1], 1, &tmp1, "transpose");

    // also write the keypoints
    const int keypoint_size=keypoints.size();
    mxArray* tmp;
    tmp=mxCreateDoubleMatrix(6,keypoint_size,mxREAL);
    double *ptr=mxGetPr(tmp);
    // fill it - attention: in Matlab, memory is transposed...
    for(int k=0; k<keypoint_size; k++){
        const int k6=6*k;
        ptr[k6]=keypoints[k].pt.x;
        ptr[k6+1]=keypoints[k].pt.y;
        ptr[k6+2]=keypoints[k].size;
        ptr[k6+3]=keypoints[k].angle;
        ptr[k6+4]=keypoints[k].class_id;
        ptr[k6+5]=keypoints[k].response;
    }
    
    // finally, re-transpose for better readibility:
    mexCallMATLAB(1, plhs, 1, &tmp, "transpose");
}


// matching
inline void symmetryInterface::knnMatch( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ){

    // ensure correct arguments
    if(nrhs<3)
        mexErrMsgTxt("Two descriptors must be passed.");
    if(nlhs!=1)
        mexErrMsgTxt("Specify one output argument.");
    if(mxGetClassID(prhs[1])!=mxUINT8_CLASS || mxGetClassID(prhs[2])!=mxUINT8_CLASS)
        mexErrMsgTxt("Wrong descriptor type.");

    // get the two input descriptors
    mxArray *d1, *d2;
    mexCallMATLAB(1, &d1, 1, (mxArray**)&prhs[1], "transpose");
    mexCallMATLAB(1, &d2, 1, (mxArray**)&prhs[2], "transpose");
    // cast to cv::Mat
    const int N1 = mxGetN(d1);
    const int N2 = mxGetN(d2);
    const int M = mxGetM(d1);
    if(M!=mxGetM(d2))
        mexErrMsgTxt("Incompatible descriptors (wrong no. bytes).");
    uchar* data1 = (uchar*)mxGetData(d1);
    uchar* data2 = (uchar*)mxGetData(d2);
    cv::Mat d1m(N1,M,CV_8U,data1);
    cv::Mat d2m(N2,M,CV_8U,data2);

    // get the number of nearest neighbors if provided
    int k=1;
    if(nrhs>3){
        if(!mxIsScalarNonComplexDouble(prhs[3]))
            mexErrMsgTxt("Wrong type for no. nearest neighbors.");
        double* kd=mxGetPr(prhs[3]);
        if (*kd<1.0) *kd=1.0;
        k=int(*kd);
    }
    
    int r = 100;
    if(nrhs>4){
        if(!mxIsScalarNonComplexDouble(prhs[4]))
            mexErrMsgTxt("Wrong type of radius.");
        double* kd=mxGetPr(prhs[4]);
        if (*kd<1.0) *kd=1.0;
        r=int(*kd);
    }

    // perform the match
    std::vector<std::vector<cv::DMatch> > matches;
    if (maskFeats)
        matcherPtr->knnMatch(d1m,d2m,matches,k,matchingMask);
    else
        matcherPtr->knnMatch(d1m,d2m,matches,k);

    // assign the output - first determine the matrix size
    const unsigned int msize=matches.size();
    const unsigned int maxMatches=k;

    // allocate memory
    plhs[0]=mxCreateDoubleMatrix(msize,maxMatches,mxREAL);
    double *data=mxGetPr(plhs[0]);

    // fill
    for(int m=0; m<msize; m++){
        const unsigned int size=matches[m].size();
        for(int s=0; s<size; s++){
            if (matches[m][s].distance < r)
                data[m+s*msize]=matches[m][s].trainIdx+1;
        }
    }
}


inline void symmetryInterface::cluster(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

    // in this case, the user is forced to pass two lhs args
    if(nlhs!=1) 
        mexErrMsgTxt("One left-hand side argument must be passed.");

    if(nrhs != 1) 
        mexErrMsgTxt("Parameter space points must be passed.");
        
    // get the input matrix
    mxArray *desc;
    mexCallMATLAB(1, &desc, 1, (mxArray**)&prhs[0], "transpose");

    // cast to cv::Mat
    const int N = mxGetN(desc);
    const int M = mxGetM(desc);

    double* data = (double*) mxGetData(desc);
    
    cv::Mat src(N,M,CV_64FC1,data);
    cv::Mat dst;
    
    dbscan->cluster(src, dst);
    
    // allocate the lhs result matrix
    int dim[2] = {dst.cols, dst.rows};

    mxArray* tmp = mxCreateNumericArray(2,dim,mxUINT8_CLASS,mxREAL);
    uchar* data = (uchar*) mxGetData(tmp); 
    // copy - kind of dumb, but necessary due to the matlab memory management
    memcpy(data,dst.data,dim[0]*dim[1]);
    
    mexCallMATLAB(1, &plhs[0], 1, &tmp, "transpose");
}


// grayImage access
inline void symmetryInterface::image( int nlhs, mxArray *plhs[], 
        int nrhs, const mxArray *prhs[] ){
    if(nlhs!=1)
        mexErrMsgTxt("No output variable specified.");
    if(nrhs!=1)
        mexErrMsgTxt("bad input.");
    if(img.empty())
        mexErrMsgTxt("No image loaded.");
    int dim[2];

    // depending on whether or not the image was imported from Matlab 
    // workspace, it needs to be transposed or not
    // must be transposed
    dim[0]=img.cols;
    dim[1]=img.rows;
    mxArray* tmp=mxCreateNumericArray(2,dim,mxUINT8_CLASS,mxREAL);
    uchar* dst=(uchar*)mxGetData(tmp);
    memcpy(dst,img.data,img.cols*img.rows);
    mexCallMATLAB(1, plhs, 1, &tmp, "transpose");
}
    


// ---------------------------------------------------------------------
// the interface object
BriskInterface* p_briskInterface=0;

// this is the actual (single) entry point:
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )     
{
    // user must provide at least one right-hand argument:
	if(nrhs < 1) mexErrMsgTxt("No input specified.");
    // and this must be a string
    if(!(mxGetClassID(prhs[0])==mxCHAR_CLASS)) mexErrMsgTxt("Bad input.");
    // parse the first input argument:
    char* str=mxArrayToString(prhs[0]);
    
    if(strcmp(str,"init")==0) {
        if(!p_briskInterface) {
            p_briskInterface=new BriskInterface(nlhs, plhs, nrhs, prhs);
            // make sure the memory requested persists. 
            //mexMakeMemoryPersistent(p_briskInterface);
        }
        else{
            mexErrMsgTxt("Brisk is already initialized.");
        }
    }
    else if(strcmp(str,"set")==0){
        p_briskInterface->set(nlhs, plhs, nrhs, prhs);
    }
    else if(strcmp(str,"loadImage")==0){
        // init if necessary
        if(!p_briskInterface) {
            p_briskInterface = new BriskInterface(nlhs, plhs, 1, prhs);
            //mexMakeMemoryPersistent(p_briskInterface);
        }
        p_briskInterface->loadImage(nlhs, plhs, nrhs, prhs);
    }
    else if(strcmp(str,"detect")==0){
        // force initialized
        if(!p_briskInterface) {
            mexErrMsgTxt("Not initialized, no image loaded.");
        }
        p_briskInterface->detect(nlhs, plhs, nrhs, prhs);
    }
    else if(strcmp(str,"describe")==0) {
        // force initialized
        if(!p_briskInterface) {
            mexErrMsgTxt("Not initialized, no image loaded.");
        }
        p_briskInterface->describe(nlhs, plhs, nrhs, prhs);
    }
    else if(strcmp(str,"knnMatch")==0) {
        // init if necessary
        if(!p_briskInterface) {
            p_briskInterface = new BriskInterface(nlhs, plhs, 1, prhs);
            //mexMakeMemoryPersistent(p_briskInterface);
        }
        p_briskInterface->knnMatch(nlhs,plhs,nrhs,prhs);
    }
    else if(strcmp(str,"cluster")==0){
        // force initialized
        if(!p_briskInterface) {
            mexErrMsgTxt("Not initialized, no image loaded.");
        }
        p_briskInterface->cluster(nlhs, plhs, nrhs, prhs);
    }
    else if(strcmp(str,"image")==0) {
        // init if necessary
        if(!p_briskInterface) {
           mexErrMsgTxt("Not initialized, no image loaded.");
        }
        p_briskInterface->image(nlhs,plhs,nrhs,prhs);
    }
    else if(strcmp(str,"terminate")==0) {
        if(p_briskInterface) {
            delete p_briskInterface;
            p_briskInterface=0;
        }
        else{
            mexErrMsgTxt("Brisk was not initialized anyways.");
        }
    }
    else{
        mexErrMsgTxt("Unrecognized input.");
    }
}
