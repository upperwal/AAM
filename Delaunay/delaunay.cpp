//
//  delaunay.cpp
//  AAM
//
//  Created by Abhishek on 23/02/15.
//  Copyright (c) 2015 Abhishek. All rights reserved.
//

#include <iostream>
#include "delaunay.h"

using namespace std;
using namespace cv;

Delaunay::Delaunay(Size size) {
    Rect rect(0, 0, size.width, size.height);
    _displayMat = Mat(rect.size(), CV_8UC3);
    _displayMat = Scalar::all(255);
    Subdiv2D temp_subdiv(rect);
    _subdiv = temp_subdiv;
}

void Delaunay::insertPoints(vector<Point2f> points) {
    for (int i=0; i<points.size(); i++) {
        cout<<"\n"<<points[i];
        _subdiv.insert(points[i]);
    }
}

void Delaunay::drawSubDiv(Mat img) {
    if (!img.empty()) {
        img.copyTo(_displayMat);
        cvtColor(_displayMat, _displayMat, COLOR_GRAY2BGR);
    }
    
    bool draw;
    
    vector<Vec6f> tempList = _triangleList;
    //_subdiv.getTriangleList(tempList);
    vector<Point2f> pt(3);
    for (size_t i = 0; i<tempList.size(); i++) {
        Vec6f t = tempList[i];
        pt[0] = Point2f(t[0], t[1]);
        pt[1] = Point2f(t[2], t[3]);
        pt[2] = Point2f(t[4], t[5]);
        
        draw=true;
        
        for(int i=0;i<3;i++){
            if(pt[i].x>_displayMat.cols||pt[i].y>_displayMat.rows||pt[i].x<0||pt[i].y<0)
                draw=false;
        }
        if (draw){
            //_triangleList.push_back(t);
            line(_displayMat, pt[0], pt[1], Scalar(255,0,255), 1);
            line(_displayMat, pt[1], pt[2], Scalar(255,0,255), 1);
            line(_displayMat, pt[2], pt[0], Scalar(255,0,255), 1);
        }
    }
    display();
}

void Delaunay::display() {
    namedWindow("Delaunay Triangle");
    imshow("Delaunay Triangle", _displayMat);
}

void Delaunay::getusefullPoints() {
    bool draw;
    _triangleList.clear();
    vector<Vec6f> tempList;
    _subdiv.getTriangleList(tempList);
    vector<Point2f> pt(3);
    for (size_t i = 0; i<tempList.size(); i++) {
        Vec6f t = tempList[i];
        pt[0] = Point2f(t[0], t[1]);
        pt[1] = Point2f(t[2], t[3]);
        pt[2] = Point2f(t[4], t[5]);
        
        draw=true;
        
        for(int i=0;i<3;i++){
            if(pt[i].x>_displayMat.cols||pt[i].y>_displayMat.rows||pt[i].x<0||pt[i].y<0)
                draw=false;
        }
        if (draw){
            _triangleList.push_back(t);
        }
    }
    drawSubDiv();
}

vector<vector<Point2f> > Delaunay::getTriangleList() {
    getusefullPoints();
    vector<vector<Point2f> > trianglePointVector;
    vector<Vec6f> tempList;
    tempList = _triangleList;
    //_subdiv.getTriangleList(tempList);
    for (int i=0; i<tempList.size(); i++) {
        vector<Point2f> trianglePointArray;
        trianglePointArray.push_back(Point2f(tempList[i][0], tempList[i][1]));
        trianglePointArray.push_back(Point2f(tempList[i][2], tempList[i][3]));
        trianglePointArray.push_back(Point2f(tempList[i][4], tempList[i][5]));
        trianglePointVector.push_back(trianglePointArray);
    }
    return trianglePointVector;
}