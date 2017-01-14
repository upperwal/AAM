//
//  coordinateSystem.h
//  AAM
//
//  Created by Abhishek on 23/02/15.
//  Copyright (c) 2015 Abhishek. All rights reserved.
//

#ifndef __AAM__coordinateSystem__
#define __AAM__coordinateSystem__

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Coordinate {
public:
    int _scale = 700;
    cv::Scalar _translation = cv::Scalar(200, 150);
    cv::Size _coordinateSize = cv::Size(400, 400);
    double minX, maxX, minY, maxY;
    
    void setCoordinateSize(vector<Point2f>);
    Scalar getMinXY();
};

#endif /* defined(__AAM__coordinateSystem__) */
