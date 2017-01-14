//
//  Procrustes.h
//  Procrustes
//
//  Created by Saburo Okita on 07/04/14. Modified by Abhishek Upperwal
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#ifndef __AAM__procrustes__
#define __AAM__procrustes__

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class GProcrustesAnalysis {
public:
    Mat translation;    /* Translation involved to obtain Yprime */
    Mat rotation;       /* Rotation involved to obtain Yprime */
    Mat Yprime;         /* The transformed point from Y to X */
    float scale;            /* Scaling invovled to obtain Yprime */
    float error;            /* Squared error of final result */
    bool scaling;
    bool bestReflection;
    
    GProcrustesAnalysis();
    GProcrustesAnalysis(bool use_scaling, bool best_reflection);
    ~GProcrustesAnalysis();
    
    vector<Point2f> yPrimeAsVector();
    float procrustes( vector<Point2f>& X, vector<Point2f>& Y );
    float procrustes( const Mat& X, const Mat& Y );
    
    vector<Mat> generalizedProcrustes( vector<vector<Point2f>>& X, vector<Point2f>& mean_shape, const int itol = 1000, const float ftol = 1e-6 );
    vector<Mat> generalizedProcrustes( vector<Mat>& X, Mat& mean_shape, const int itol = 1000, const float ftol = 1e-6 );
    
protected:
    static inline float sumSquared( const cv::Mat& mat );
    vector<Mat> recenter( const vector<Mat>& X );
    vector<Mat> normalize( const vector<Mat>& X );
    vector<Mat> align( const vector<Mat>& X, Mat& mean_shape );
    
};

#endif /* defined(__AAM__procrustes__) */
