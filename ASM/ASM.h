//
//  ASM.h
//  AAM
//
//  Created by Abhishek on 20/02/15.
//  Copyright (c) 2015 Abhishek. All rights reserved.
//

#ifndef __AAM__ASM__
#define __AAM__ASM__

#include <opencv2/opencv.hpp>
#include "procrustes.h"
#include "PCA.h"
#include "delaunay.h"
#include "coordinateSystem.h"
#include "Filesystem.h"

using namespace std;
using namespace cv;

class ASM {
    
private:
    vector<vector <Point2f> > _annotatedPoints;
    vector<Point2f> _meanShape;
    vector<Mat> _procrustesResultMat;
    Coordinate _coordinate;
    Mat _shapeEigenvector, _shapeEigenvalues;
    //Mat _shapeEVecVal;
    const String _shapeModelFileName = "shapemodel.yaml";
    
public:
    ASM();
    ASM(vector<vector <Point2f> > annotatedPoints);
    vector<vector<Point2f> > _annotatedProcustes;
    void generateASM();
    void displayMean();
    void convertProcustesToPoint2f();
    void displayProcrustesResult();
    void transposeProcrustesResult();
    vector<vector<Point2f> > findMeanDelaunayTriangles();
    void saveShapeModel(String);
    void standadizeData();
    /*Getters*/
    vector<Point2f> getMeanShapeVector();
    Mat getEigenvector();
    Coordinate getCoordinate();
    /*Mat getshapeEVecValConcat();*/
    
    /*End Getters */
    void loadShapeModel(String);

private:
    void computePCA();
    Mat formatForPCA(const vector<Mat>);
};

#endif /* defined(__AAM__ASM__) */
