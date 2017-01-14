//
//  Filesystem.h
//  AAM
//
//  Created by Abhishek on 16/03/15.
//  Copyright (c) 2015 Abhishek. All rights reserved.
//

#ifndef __AAM__Filesystem__
#define __AAM__Filesystem__

#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

class Filesystem {
private:
    FileStorage _storage;
public:
    enum {READ, WRITE};
    Filesystem(String, int);
    void release();
    void writeMat(String, Mat);
    Mat loadMat(String);
    void writeVectorPoint2f(String, vector<Point2f>);
    vector<Point2f> loadVectorPoint2f(String);
    void writeVectorVectorPoint2f(String, vector<vector<Point2f> >);
    vector<vector<Point2f> > loadVectorVectorPoint2f(String);
};

#endif /* defined(__AAM__Filesystem__) */
