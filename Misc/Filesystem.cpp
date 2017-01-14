//
//  Filesystem.cpp
//  AAM
//
//  Created by Abhishek on 16/03/15.
//  Copyright (c) 2015 Abhishek. All rights reserved.
//

#include "Filesystem.h"

Filesystem::Filesystem(String file, int type) {
    if (type == READ) {
        FileStorage temp(file, FileStorage::READ);
        _storage = temp;
    }
    else if(type == WRITE) {
        FileStorage temp(file, FileStorage::WRITE);
        _storage = temp;
    }
}

void Filesystem::release() {
    _storage.release();
}

void Filesystem::writeMat(String tag, Mat matrix) {
    _storage << tag << matrix;
}

Mat Filesystem::loadMat(String file) {
    Mat dataMat;
    _storage[file] >> dataMat;
    return dataMat;
}

void Filesystem::writeVectorPoint2f(String tag, vector<Point2f> data) {
    _storage << tag << data;
}

vector<Point2f> Filesystem::loadVectorPoint2f(String tag) {
    vector<Point2f> data;
    _storage[tag] >> data;
    return data;
}

void Filesystem::writeVectorVectorPoint2f(String tag, vector<vector<Point2f> > data) {
    Mat temp((int)data.size(),2*(int)data[0].size(),CV_64FC1);
    for (int i=0; i<(int)data.size(); i++) {
        for (int j=0,k=0; j<2*(int)data[0].size(); j+=2,k++) {
            temp.at<double>(i,j) = data[i][k].x;
            temp.at<double>(i,j+1) = data[i][k].y;
        }
    }
    _storage << tag << temp;
}

vector<vector<Point2f> > Filesystem::loadVectorVectorPoint2f(String tag) {
    vector<vector<Point2f> > data;
    Mat temp;
    _storage[tag] >> temp;
    for (int i=0; i<temp.rows; i++) {
        vector<Point2f> vectPoint;
        for (int j=0; j<temp.cols; j+=2) {
            Point2f point;
            point.x = temp.at<double>(i,j);
            point.y = temp.at<double>(i,j+1);
            vectPoint.push_back(point);
        }
        data.push_back(vectPoint);
    }
    return data;
}