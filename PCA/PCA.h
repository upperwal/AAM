//
//  PCA.h
//  AAM
//
//  Created by Abhishek on 20/02/15.
//  Copyright (c) 2015 Abhishek. All rights reserved.
//

#ifndef __AAM__PCA__
#define __AAM__PCA__

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class CustomPCA: public PCA {
private:
    Mat _data;
    Mat _formattedForPCA;

public:
    CustomPCA();
    CustomPCA(const vector<Mat> &, Mat, double, int majorType = PCA::DATA_AS_ROW);
    CustomPCA(const vector<Mat> &, Mat, int, int majorType = PCA::DATA_AS_ROW);
    Mat asRowMatrix(const vector<Mat>&, int rytpe = CV_32FC1, double alpha = 1, double beta = 0);
    Mat asColMatrix(const vector<Mat>&, int rytpe = CV_32FC1, double alpha = 1, double beta = 0);
    Mat projectRow(int index);
    Mat projectCol(int index);
    Mat projectData(Mat);
};

#endif /* defined(__AAM__PCA__) */
