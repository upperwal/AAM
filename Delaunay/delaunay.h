//
//  delaunay.h
//  AAM
//
//  Created by Abhishek on 23/02/15.
//  Copyright (c) 2015 Abhishek. All rights reserved.
//

#ifndef __AAM__delaunay__
#define __AAM__delaunay__

#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class Delaunay {
private:
    Subdiv2D _subdiv;
    Mat _displayMat;
    vector<Vec6f> _triangleList;

public:
    Delaunay(Size size);
    void insertPoints(vector<Point2f>);
    void drawSubDiv(Mat img = Mat());
    void display();
    vector<vector<Point2f> > getTriangleList();
    void getusefullPoints();
};

#endif /* defined(__AAM__delaunay__) */
