//
//  PCA.cpp
//  AAM
//
//  Created by Abhishek on 20/02/15.
//  Copyright (c) 2015 Abhishek. All rights reserved.
//

#include "PCA.h"

CustomPCA::CustomPCA() {
}

CustomPCA::CustomPCA(const vector<Mat> &data, Mat meanShape, double variance, int majorType):PCA(asRowMatrix(data), Mat(), majorType, variance) {
    _formattedForPCA = asRowMatrix(data);
}

CustomPCA::CustomPCA(const vector<Mat> &data, Mat meanShape, int reducedDim, int majorType):PCA(asRowMatrix(data), Mat(), majorType, reducedDim) {
    _formattedForPCA = asRowMatrix(data);
}

Mat CustomPCA::asColMatrix(const vector<Mat>& src, int rtype, double alpha, double beta) {
    // Number of samples:
    size_t n = src.size();
    // Return empty matrix if no matrices given:
    if(n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src[0].total();
    // Create resulting data matrix:
    Mat data((int)d, (int)n, rtype);
    // Now copy data:
    for(int i = 0; i < n; i++) {
        //
        if(src[i].empty()) {
            string error_message = format("Image number %d was empty, please check your input data.", i);
            CV_Error(Error::StsBadArg, error_message);
        }
        // Make sure data can be reshaped, throw a meaningful exception if not!
        if(src[i].total() != d) {
            string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
            CV_Error(Error::StsBadArg, error_message);
        }
        // Get a hold of the current row:
        Mat xi = data.col(i);
        // Make reshape happy by cloning for non-continuous matrices:
        if(src[i].isContinuous()) {
            src[i].reshape(2, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src[i].clone().reshape(2, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

Mat CustomPCA::asRowMatrix(const vector<Mat>& src, int rtype, double alpha, double beta) {
    // Number of samples:
    size_t n = src.size();
    // Return empty matrix if no matrices given:
    if(n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src[0].total();
    cout<<d;
    // Create resulting data matrix:
    Mat data((int)n, (int)d, rtype);
    // Now copy data:
    for(int i = 0; i < n; i++) {
        //
        if(src[i].empty()) {
            string error_message = format("Image number %d was empty, please check your input data.", i);
            CV_Error(Error::StsBadArg, error_message);
        }
        // Make sure data can be reshaped, throw a meaningful exception if not!
        if(src[i].total() != d) {
            string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
            CV_Error(Error::StsBadArg, error_message);
        }
        // Get a hold of the current row:
        Mat xi = data.row(i);
        // Make reshape happy by cloning for non-continuous matrices:
        if(src[i].isContinuous()) {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

Mat CustomPCA::projectRow(int index) {
    return project(_formattedForPCA.row(index));
}

Mat CustomPCA::projectCol(int index) {
    return project(_formattedForPCA.col(index));
}

Mat CustomPCA::projectData(Mat data) {
    return project(data);
}