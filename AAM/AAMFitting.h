//
//  AAMFitting.h
//  AAM
//
//  Created by Abhishek on 07/03/15.
//  Copyright (c) 2015 Abhishek. All rights reserved.
//

#ifndef __AAM__AAMFitting__
#define __AAM__AAMFitting__

#include <iostream>
#include <opencv2/highgui.hpp>
#include "AAM.h"
#include "ASM.h"

using namespace cv;
using namespace std;

class AAMFitting {
private:
    Mat _templateImage, _gradientX, _gradientY;
    AAM _aam;
    ASM _asm;
    vector<Point2f> getAllTrianglesAtPoint(Point2f, vector<vector<Point2f> >);
    
    vector<Point2f> _fittingShape;
    Rect _faceRegion;
    Mat _inputImage;
    int _noIterations;
    Mat _Wx_dp, _Wy_dp;
    Mat _imageJacobian;
    Mat _similarityEigen, Q;
    vector<vector<Point2f> > triangleIndexOfAllPoints;
    vector<vector<vector<int> > > pointIndexingOfEachTriangleAroundPoint;
    
public:
    AAMFitting();
    AAMFitting(AAM&);
    void loadModelData(String);
    void startFitting(int);
    void preCompute();
    void initialize();
    void iterate();
    Mat project(Mat, Mat, Mat);
    Mat backProject(Mat, Mat, Mat);
    void loadTemplateImage(Mat);
    void findTemplateGradient(Mat);
    void imageJacobian();
    void createWarpJacobian();
    Mat createSimilarityEigens();
    Mat orthonoram(Mat);
    void setIndexOfPointOfEachTriangleAroundPoint();
    void detectFaceRegionAndAlignFittingShape();
    void displayThesePoints(String, Mat, vector<Point2f>);
    Mat wrapAffineToMean();
    int indexInMean(Point2f);
    void computeWarpUpdate(Mat);
    /*void computeSteepestDescentImages();
    void project_onto_jacobian_ECC(const Mat &,const Mat &, Mat &);*/
};

#endif /* defined(__AAM__AAMFitting__) */