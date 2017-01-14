//
//  annotate.cpp
//  AAM
//
//  Created by Abhishek on 19/02/15.
//  Copyright (c) 2015 Abhishek. All rights reserved.
//

#include "annotate.h"
#include <string>

using namespace std;
using namespace cv;

vector<Point2f> Annotation::_singleImagePoints;

Annotation::Annotation(int noImages, string folder, string imagePrefix, string imagePostfix) {
    this->_folder = folder;
    this->_imagePrefix = imagePrefix;
    this->_imagePostfix = imagePostfix;
    this->_noImages = noImages;
}

Mat Annotation::loadImage(int index) {
    string sourceImagePath;
    sourceImagePath = _folder+"/"+_imagePrefix+to_string(index)+"."+_imagePostfix;
    return imread(sourceImagePath, IMREAD_GRAYSCALE);
}

vector<vector<Point2f> > Annotation::annotateImage() {
    namedWindow("Annotation");
    setMouseCallback("Annotation", handleMouseClicks);
    vector<vector<Point2f> > result;
    int imageIndex = 0;
    Mat loadedImage = loadImage(++imageIndex);
    
    while(true) {
        for (int i=0; i<_singleImagePoints.size(); i++) {
            circle(loadedImage, _singleImagePoints[i], 1, Scalar(255,255,0), 2);
        }
        imshow("Annotation", loadedImage);
        
        int ch = waitKey(10);
        if (ch == 'n') {
            result.push_back(_singleImagePoints);
            _singleImagePoints.clear();
            loadedImage = loadImage(++imageIndex);
            if (imageIndex>_noImages) {
                break;
            }
        }
    }
    _result = result;
    return result;
}

void Annotation::savePoints() {
    if(_result.empty()) {
        cout<<"First run annotateImage()\n";
        return;
    }
    for(int i=0;i<_noImages;i++) {
        string sourceImagePath;
        sourceImagePath = _folder+"/"+_imagePrefix+to_string(i+1)+".pts";
        FileStorage landmarksFile(sourceImagePath, FileStorage::WRITE);
        write(landmarksFile, "size", (int)_result[i].size());
        write(landmarksFile, "landmark", _result[i]);
        landmarksFile.release();
    }
}

vector<vector<Point2f> > Annotation::loadPoints() {
    _result.clear();
    for(int i=0;i<_noImages;i++) {
        string sourceImagePath;
        sourceImagePath = _folder+"/"+_imagePrefix+to_string(i+1)+".pts";
        FileStorage read(sourceImagePath, FileStorage::READ);
        if(!read.isOpened()) {
            cout<<"Could not open landmarks.yml";
        }
        //int numberOfImagesInFile = read["size"];
        FileNode temp = read["landmark"];
        vector<Point2f> tempPoint;
        cv::read(temp, tempPoint);
        _result.push_back(tempPoint);
    }
    return _result;
}

void Annotation::displayPoints() {
    namedWindow("Annotated Points");
    
    for (int i=0; i<_result.size(); i++) {
        Mat display(500, 500, CV_32FC3, Scalar::all(255));
        for(Point2f point : _result[i]) {
            circle(display, point, 1, Scalar(0,0,0), 2, -1);
        }
        imshow("Annotated Points", display);
        waitKey(500);
    }
    
}