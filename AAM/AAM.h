//
//  AAM.h
//  AAM
//
//  Created by Abhishek on 20/02/15.
//  Copyright (c) 2015 Abhishek. All rights reserved.
//

#ifndef __AAM__AAM__
#define __AAM__AAM__

#include <opencv2/opencv.hpp>
#include "PCA.h"
#include "delaunay.h"
#include "coordinateSystem.h"
#include "Filesystem.h"

using namespace std;
using namespace cv;

class AAM {
    
private:
    vector<Mat> _trainingSet;
    vector<Mat> _shapeFreeImages;
    vector<vector<Point2f> > _annotatedPoints;
    vector<vector<Point2f> > _annotatedProPoints;
    Mat _meanAppearance;
    int _noImages;
public:
    string _folder;
    vector<Point2f> _meanShapeVector;
    Mat _maskForFace, _maskForFaceWOBoundry;
    CustomPCA _pca;
    Mat _appEigenVector, _appEigenValue, _meanAppearanceRaw;
private:
    string _imagePrefix;
    string _imagePostfix;
    Mat _shapeWithTriangleIndex;
    vector<vector<Point2f> > _meanShapeTriangleList;
    const String _modelFileName = "appearancemodel.yaml";
    
public:
    AAM();
    AAM(Coordinate, vector<Point2f>, vector<vector<Point2f> >, vector<vector<Point2f> >, int, string folder="images", string imagePrefix="image_", string imagePostfix="jpg");
    void saveActiveModel(String);
    void loadActiveModel(String);
    Coordinate _coordinate;
    AAM& operator=(AAM&);
    void loadImages();
    Mat norm_0_255(const Mat& );
    void computeTexture();
    void wrapAffineToMean(vector<vector<Point2f> >);
    int indexInMean(Point2f);
    Mat getMeanAppearance();
    Mat getMaskForFace();
    Mat maskForFaceWOBoundry();
    Mat createShapeWithTriangleIndex();
    vector<Point2f> getMeanShape();
    vector<vector<Point2f> > getMeanShapeTriangles();
};

#endif /* defined(__AAM__AAM__) */
