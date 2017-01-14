//
//  annotate.h
//  AAM
//
//  Created by Abhishek on 19/02/15.
//  Copyright (c) 2015 Abhishek. All rights reserved.
//

#ifndef __AAM__annotate__
#define __AAM__annotate__

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Annotation {
private:
    string _folder;
    string _imagePrefix;
    string _imagePostfix;
    int _noImages;
    static vector<Point2f> _singleImagePoints;
    vector<vector<Point2f> > _result;
    
public:
    Annotation(int noImages, string folder, string imagePrefix, string imagePostfix);
    static void handleMouseClicks(int event, int x, int y, int flags, void *userdata) {
        if (event == EVENT_LBUTTONDOWN) {
            Point2f point(x, y);
            Annotation::_singleImagePoints.push_back(point);
        }
    }
    vector<vector<Point2f> > annotateImage();
    void savePoints();
    vector<vector<Point2f> > loadPoints();
    void displayPoints();
private:
    Mat loadImage(int);
};

#endif /* defined(__AAM__annotate__) */
