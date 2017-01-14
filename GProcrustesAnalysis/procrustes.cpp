//
//  Procrustes.cpp
//  Procrustes
//
//  Created by Saburo Okita on 07/04/14.  Modified by Abhishek Upperwal
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#include "procrustes.h"

#include <random>

using namespace std;
using namespace cv;


GProcrustesAnalysis::GProcrustesAnalysis():
scale(1.0f), error(0.0f), scaling( true ), bestReflection( true ) {
}

GProcrustesAnalysis::GProcrustesAnalysis( bool use_scaling, bool best_reflection ) : scale(1.0f), error(0.0f), scaling( use_scaling ), bestReflection( best_reflection ) {
    
}

GProcrustesAnalysis::~GProcrustesAnalysis(){
}

/**
 * Convert YPrime back from Mat to vector<Point2f>
 * convenience method.
 */
vector<Point2f> GProcrustesAnalysis::yPrimeAsVector() {
    vector<Point2f> result;
    Yprime.copyTo( result );
    return result;
}

/**
 * Sum of squared of the 2 channels matrix
 */
float GProcrustesAnalysis::sumSquared( const Mat& mat ) {
    Mat temp;
    pow( mat, 2.0, temp );
    Scalar temp_scalar = sum( temp );
    return temp_scalar[0] + temp_scalar[1];
}

/**
 * Wrapper for vector of points
 */
float GProcrustesAnalysis::procrustes(vector<Point2f> &X, vector<Point2f> &Y) {
    return procrustes( Mat(X), Mat(Y));
}

/**
 * Perform procrustes analysis / superimpose (?)
 */
float GProcrustesAnalysis::procrustes( const Mat& X, const Mat& Y ){
    /* Recenter the points based on their mean ... */
    Scalar mu_x = cv::mean(X);
    Mat X0      = X - Mat(X.size(), X.type(), mu_x);
    
    Scalar mu_y = cv::mean(Y);
    Mat Y0      = Y - Mat(Y.size(), Y.type(), mu_y);
    
    
    /* ... and normalize them */
    float ss_X      = sumSquared( X0 );
    float norm_X    = sqrt( ss_X );
    X0              /= norm_X;
    
    float ss_Y      = sumSquared( Y0 );
    float norm_Y    = sqrt( ss_Y );
    Y0              /= norm_Y;
    
    
    /* Pad with zeros is Y has less points than X */
    if( Y.rows < X.rows )
        vconcat( Y0, Mat::zeros( X.rows - Y.rows, 1, Y.type()), Y0 );
    
    
    /* Perform SVD */
    Mat A = X0.reshape(1).t() * Y0.reshape(1);
    Mat U, s, Vt;
    SVDecomp( A, s, U, Vt );
    
    /* Since in USV, U and V represents the rotation, and S is the scaling... */
    Mat V           = Vt.t();
    this->rotation  = V * U.t();
    
    
    if( !bestReflection ) {
        bool have_reflection = determinant( this->rotation ) < 0;
        if( bestReflection != have_reflection ) {
            V.colRange( V.cols-1, V.cols) *= -1;
            s.rowRange( s.rows-1, s.rows) *= -1;
            this->rotation = V * U.t();
        }
    }
    
    /* Rotate Y0 first */
    Mat rotated_Y0;
    cv::transform( Y0, rotated_Y0, rotation.t() );
    
    /* Trace of eigenvalues is basically the scale */
    float trace_TA = sum( s )[0];
    
    if( scaling ) {
        scale   = trace_TA * norm_X / norm_Y;
        error   = 1 - trace_TA * trace_TA;
        Yprime  = norm_X * trace_TA * rotated_Y0 + mu_x;
    }
    else {
        error   = 1 + ss_Y / ss_X - 2 * trace_TA * norm_X / norm_Y;
        Yprime  = norm_Y * rotated_Y0 + mu_x;
    }
    
    if( Y.rows < X.rows )
        rotation = rotation.rowRange(0, Y.rows );
    
    translation = Mat(1, 1, CV_32FC2, mu_x).reshape(1) - scale * Mat(1, 1, CV_32FC2, mu_y).reshape(1) * rotation;
    
    return error;
}

/**
 * Recenter each matrix / set of points, by subtracting them with the mean / centroid
 */
vector<Mat> GProcrustesAnalysis::recenter( const vector<Mat>& X ) {
    vector<Mat> result;
    for( Mat x : X ) {
        Scalar mu_x = cv::mean(x);
        Mat temp    = x - Mat(x.size(), x.type(), mu_x);
        result.push_back( temp );
    }
    return result;
}

/**
 * Normalize each set of points
 */
vector<Mat> GProcrustesAnalysis::normalize( const vector<Mat>& X ) {
    vector<Mat> result;
    for( Mat x : X )
        result.push_back( x / cv::norm( x ) );
    return result;
}

/**
 * Perform rotation alignment between a set of points / shapes and a choosen mean shape
 * Points and mean shape should already be centered and normalize prior to the call of this function
 */
vector<Mat> GProcrustesAnalysis::align( const vector<Mat>& X, Mat& mean_shape ) {
    cout<<(mean_shape.reshape(1)).size();
    vector<Mat> result;
    Mat w, u, vt;
    for( Mat x : X ){
        SVDecomp( mean_shape.reshape(1).t() * x.reshape(1) , w, u, vt);
        result.push_back( (x.reshape(1) * vt.t()) * u.t() );
    }
    return result;
}


/**
 * Wrapper for vector<vector<Point2f>>
 */
vector<Mat> GProcrustesAnalysis::generalizedProcrustes( vector<vector<Point2f>>& X, vector<Point2f>& mean_shape, const int itol, const float ftol ) {
    vector<Mat> temp(X.size());
    for( int i = 0; i < temp.size(); i++ ) {
        //cout<<X[i][0]<<" "<<X[i][1];
        temp[i] = Mat( X[i] );
        cout<<temp[i].channels()<<"\n";
    }
    
    Mat mean_shape_mat;
    cout<<temp[0].size();
    temp = generalizedProcrustes( temp, mean_shape_mat, itol, ftol );
    
    /* Copy the mean shape back as vector of Point2f */
    mean_shape.clear();
    mean_shape_mat.reshape(2).copyTo( mean_shape );
    cout<<mean_shape[0];
    
    /* Copy the transformed X as vector of vector of Point2f, very circuitous way of doing thing*/
    /*vector<vector<Point2f>> result;
    for( Mat mat : temp ){
        vector<Point2f> vec;
        mat.reshape(2).copyTo( vec );
        result.push_back( vec );
    }*/
    //cout<<temp[0].size(); [68x2]
    return temp;
}

/**
 * Perform a generalized Procrustes analysis to find the mean shape
 **/
vector<Mat> GProcrustesAnalysis::generalizedProcrustes( std::vector<cv::Mat>& X, Mat& mean_shape, const int itol, const float ftol ) {
    /* Arbitrarily choose the first set of points as our mean shape */
    cout<<X[0].size();
    mean_shape = X[0].reshape( 1 );
    cout<<X[0].channels();
    
    
    int counter = 0;
    while( true ) {
        /* recenter, normalize, align */
        X = recenter( X );
        cout<<X[0].size();
        X = normalize( X );
        cout<<X[0].size();
        X = align( X, mean_shape );
        cout<<X[0].size();
        
        
        /* Find a new mean shape from all the set of points */
        Mat new_mean = Mat::zeros( mean_shape.size(), mean_shape.type() );
        accumulate( X.begin(), X.end(), new_mean );
        
        new_mean = new_mean / X.size();
        new_mean = new_mean / norm( new_mean );
        
        /* Perform the loop until convergence */
        float diff = norm( new_mean, mean_shape );
        if( counter++ > itol || diff <= ftol  )
            break;
        
        mean_shape = new_mean;
    }
    cout<<"ki: "<<X[0].at<float>(0, 0)<<" "<<X[0].at<float>(0, 1)<<" ";
    cout<<X[0].channels()<<" "<<X[0].size();
    return X;
}