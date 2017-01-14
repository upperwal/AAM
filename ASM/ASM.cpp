//
//  ASM.cpp
//  AAM
//
//  Created by Abhishek on 20/02/15.
//  Copyright (c) 2015 Abhishek. All rights reserved.
//

#include "ASM.h"

ASM::ASM() {
    
}

ASM::ASM(vector<vector <Point2f> > annotatedPoints) {
    _annotatedPoints = annotatedPoints;
}

void ASM::generateASM() {
    GProcrustesAnalysis procrustes;
    _procrustesResultMat = procrustes.generalizedProcrustes(_annotatedPoints, _meanShape);
    transposeProcrustesResult();
    standadizeData();
    computePCA();
}

void ASM::convertProcustesToPoint2f() {
    _annotatedProcustes.clear();
    vector<Mat> result = _procrustesResultMat;
    for (int i=0; i<result.size(); i++) {
        vector<Point2f> resultShape;
        result[i].reshape(2, (int)_annotatedPoints[0].size()).copyTo(resultShape);
        cout<<resultShape[0];
        _annotatedProcustes.push_back(resultShape);
    }
}

void ASM::loadShapeModel(String path) {
    Filesystem shapeModel(path + "/" + _shapeModelFileName, Filesystem::READ);
    _meanShape = shapeModel.loadVectorPoint2f("meanshape");
    _shapeEigenvector = shapeModel.loadMat("shapeEigenvector");
    //_shapeEVecVal = shapeModel.loadMat("shapeEVecVal");
    _coordinate.setCoordinateSize(_meanShape);
}

Mat ASM::getEigenvector() {
    return _shapeEigenvector;
}

Coordinate ASM::getCoordinate() {
    return _coordinate;
}

/*Mat ASM::getshapeEVecValConcat() {
    return _shapeEVecVal;
}*/

void ASM::standadizeData() {
    Mat meanMat(_meanShape);
    meanMat *= _coordinate._scale;
    meanMat += _coordinate._translation;
    _coordinate.setCoordinateSize(_meanShape);
    Scalar size = _coordinate.getMinXY();
    meanMat -= size;
    for (int i=0; i<_procrustesResultMat.size(); i++) {
        Mat proMat(_procrustesResultMat[i]);
        proMat *= _coordinate._scale;
        proMat += _coordinate._translation;
        proMat -= size;
    }
    convertProcustesToPoint2f();
}

void ASM::saveShapeModel(String path) {
    Filesystem storeShape(path + "/" + _shapeModelFileName, Filesystem::WRITE);
    storeShape.writeVectorPoint2f("meanshape", _meanShape);
    storeShape.writeMat("shapeEigenvector", _shapeEigenvector);
    //storeShape.writeMat("shapeEVecVal", _shapeEVecVal);
    storeShape.release();
}

void ASM::displayMean() {
    Mat display(_coordinate._coordinateSize, CV_32FC3, Scalar::all(255));
     for(Point2f point : _meanShape) {
         Scalar color(255,0,255);
         circle(display, point, 1, color, 2, -1);
     }
    namedWindow("Mean");
    imshow("Mean", display);
}

void ASM::transposeProcrustesResult() {
    for (int i=0; i<_procrustesResultMat.size(); i++) {
        _procrustesResultMat[i] = _procrustesResultMat[i].t();
        cout<<_procrustesResultMat[i].size();
    }
}

void ASM::displayProcrustesResult() {
    vector<Mat> result = _procrustesResultMat;
    Mat display(_coordinate._coordinateSize, CV_32FC3, Scalar::all(255));
    int k=0;
    for (int i=0; i<result.size(); i++) {
        vector<Point2f> resultShape;
        result[i].reshape(2).copyTo(resultShape);
        const vector<Scalar> colors = {
            Scalar( 255, 0, 0 ),
            Scalar( 0, 255, 0 )
        };
        for(Point2f point : resultShape) {
            circle(display, point, 1, colors[k], 2);
        }
        k++;
    }
    namedWindow("Procrustes Result");
    imshow("Procrustes Result", display);
}

void ASM::computePCA() {
    cout<<"Computing PCA\n";
    Mat mean;
    mean = Mat(_meanShape);
    cout<<mean.size();
    CustomPCA pca(_procrustesResultMat, mean, 0.95);
    //cout<<_meanShape.size()<<" "<<_procrustesResultMat.size();
    exit(1);
    
    _shapeEigenvalues = pca.eigenvalues;
    _shapeEigenvector = pca.eigenvectors.t();
    _shapeEigenvector.convertTo(_shapeEigenvector, CV_64FC1);
    
    /*Mat eVDia = Mat::diag(_shapeEigenvalues);
    hconcat(_shapeEigenvector, eVDia, _shapeEVecVal);
     */
    
    Mat proj = pca.projectRow(1);
    Mat back = pca.backProject(proj);
    
    vector<Point2f> pointVec;
    Mat backReshape = back.reshape(1, 2);
    //back.reshape(2, 2).reshape(2).copyTo(point);
    for (int i=0; i<backReshape.cols; i++) {
        Point2f pt(backReshape.col(i));
        pointVec.push_back(pt);
    }
    namedWindow("BackP");
    Mat displayy(_coordinate._coordinateSize, CV_32FC3, Scalar::all(255));
    for(Point2f point : pointVec) {
        Scalar color(255,0,255);
        circle(displayy, point, 1, color, 2, -1);
    }
    imshow("BackP", displayy);
    
    Mat display(_coordinate._coordinateSize, CV_32FC3, Scalar::all(255));
    int k=0;
    for (int i=0; i<_procrustesResultMat.size(); i++) {
        Mat res = pca.projectRow(i);
        //res.size() is 4 X 1.
        Mat reProj = pca.backProject(res);
        vector<Point2f> resultShape;
        Mat reProjReshape = reProj.reshape(1, 2);
        //back.reshape(2, 2).reshape(2).copyTo(point);
        for (int i=0; i<reProjReshape.cols; i++) {
            Point2f pt(reProjReshape.col(i));
            resultShape.push_back(pt);
        }
        const vector<Scalar> colors = {
            Scalar( 255, 0, 0 ),
            Scalar( 0, 255, 0 )
        };
        for(Point2f point : resultShape) {
            circle(display, point, 1, colors[k], 2);
        }
        k++;
    }

    namedWindow("Shape Back Projection");
    imshow("Shape Back Projection", display);
}

vector<vector<Point2f> > ASM::findMeanDelaunayTriangles() {
    Delaunay delaunay(_coordinate._coordinateSize);
    delaunay.insertPoints(_meanShape);
    delaunay.drawSubDiv(Mat());
    return delaunay.getTriangleList();
}

vector<Point2f> ASM::getMeanShapeVector() {
    return _meanShape;
}