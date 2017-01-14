//
//  AAM.cpp
//  AAM
//
//  Created by Abhishek on 20/02/15.
//  Copyright (c) 2015 Abhishek. All rights reserved.
//

#include "AAM.h"

AAM::AAM() {
}

AAM::AAM(Coordinate coord, vector<Point2f> meanShape, vector<vector<Point2f> > annotatedPoints, vector<vector<Point2f> > annotatedProPoints, int noImages, string folder, string imagePrefix, string imagePostfix) {
    _annotatedPoints = annotatedPoints;
    _annotatedProPoints = annotatedProPoints;
    _meanShapeVector = meanShape;
    _noImages = noImages;
    _folder = folder;
    _imagePrefix = imagePrefix;
    _imagePostfix = imagePostfix;
    _coordinate = coord;
}

AAM& AAM::operator=(AAM &aam) {
    _trainingSet = aam._trainingSet;
    _shapeFreeImages = aam._shapeFreeImages;
    _annotatedPoints = aam._annotatedPoints;
    _annotatedProPoints = aam._annotatedProPoints;
    _meanShapeVector = aam._meanShapeVector;
    _meanAppearance = aam._meanAppearance;
    _noImages = aam._noImages;
    _folder = aam._folder;
    _imagePrefix = aam._imagePrefix;
    _imagePostfix = aam._imagePostfix;
    _shapeWithTriangleIndex = aam._shapeWithTriangleIndex;
    _meanShapeTriangleList = aam._meanShapeTriangleList;
    _coordinate = aam._coordinate;
    
    return *this;
}

void AAM::saveActiveModel(String path) {
    Filesystem storeAppearance(path + "/" + _modelFileName, Filesystem::WRITE);
    storeAppearance.writeMat("meanappearance", _meanAppearance);
    storeAppearance.writeVectorPoint2f("meanshapevector", _meanShapeVector);
    storeAppearance.writeMat("shapewithtriangleindex", _shapeWithTriangleIndex);
    storeAppearance.writeVectorVectorPoint2f("meanshapetrianglelist", _meanShapeTriangleList);
    storeAppearance.writeVectorVectorPoint2f("annotatedProPoints", _annotatedProPoints);
    storeAppearance.writeMat("maskForFace", _maskForFace);
    storeAppearance.writeMat("maskForFaceWOBoundry", _maskForFaceWOBoundry);
    storeAppearance.writeMat("appEigenVector", _appEigenVector);
    storeAppearance.writeMat("appEigenValue", _appEigenValue);
    storeAppearance.writeMat("meanAppearanceRaw", _meanAppearanceRaw);
}

void AAM::loadActiveModel(String path) {
    Filesystem appearanceFile(path + "/" + _modelFileName, Filesystem::READ);
    _meanAppearance = appearanceFile.loadMat("meanappearance");
    _meanShapeVector = appearanceFile.loadVectorPoint2f("meanshapevector");
    _shapeWithTriangleIndex = appearanceFile.loadMat("shapewithtriangleindex");
    _meanShapeTriangleList = appearanceFile.loadVectorVectorPoint2f("meanshapetrianglelist");
    _maskForFace = appearanceFile.loadMat("maskForFace");
    _maskForFaceWOBoundry = appearanceFile.loadMat("maskForFaceWOBoundry");
    _coordinate.setCoordinateSize(_meanShapeVector);
    _appEigenVector = appearanceFile.loadMat("appEigenVector");
    _appEigenValue = appearanceFile.loadMat("appEigenValue");
    _meanAppearanceRaw = appearanceFile.loadMat("meanAppearanceRaw");
    _annotatedProPoints = appearanceFile.loadVectorVectorPoint2f("annotatedProPoints");
}

Mat AAM::getMeanAppearance() {
    return _meanAppearance;
}

Mat AAM::getMaskForFace() {
    return _maskForFace;
}

Mat AAM::maskForFaceWOBoundry() {
    return _maskForFaceWOBoundry;
}

void AAM::loadImages() {
    for (int i=1; i<=_noImages; i++) {
        string sourceImagePath;
        sourceImagePath = _folder+"/"+_imagePrefix+to_string(i)+"."+_imagePostfix;
        _trainingSet.push_back(imread(sourceImagePath, IMREAD_GRAYSCALE));
    }
}

Mat AAM::norm_0_255(const Mat& src) {
    Mat dst;
    switch(src.channels()) {
        case 1:
            cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
            break;
        case 3:
            cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
            break;
        default:
            src.copyTo(dst);
            break;
    }
    return dst;
}

void AAM::computeTexture() {
    if (_shapeFreeImages.empty()) {
        cout<<"First call loadImages().";
        return;
    }
    Mat meanPCA;
    CustomPCA pca(_shapeFreeImages, meanPCA, 0.97);
    _pca = pca;
    
    Mat projection = pca.projectRow(0);
    /*cout<<"Pro Size: "<<projection.size();
    cout<<"\nProjec: "<<projection;
    projection = pca.projectRow(0);
    cout<<"\nProjec: "<<projection;
    projection = pca.projectRow(1);
    cout<<"\nProjec: "<<projection;
    projection = pca.projectRow(2);
    cout<<"\nProjec: "<<projection;
    projection = pca.projectRow(3);
    cout<<"\nProjec: "<<projection;*/
    //projection.at<float>(0,1) = -1500.3456;
    
    int pr1,pr2,pr3,pr4;
    pr1 = -projection.at<float>(0,0);
    pr2 = -projection.at<float>(0,1);
    pr3 = -projection.at<float>(0,2);
    pr4 = -projection.at<float>(0,3);
    
    //namedWindow("backproject");
    createTrackbar("Value 1", "backproject", &pr1, 2500);
    createTrackbar("Value 2", "backproject", &pr2, 2500);
    createTrackbar("Value 3", "backproject", &pr3, 2500);
    createTrackbar("Value 4", "backproject", &pr4, 2500);
    
    Mat backProjection;
    backProjection = pca.backProject(projection);
    /*while (true) {
        projection.at<float>(0,0) = (float)pr1;
        projection.at<float>(0,1) = (float)pr2;
        projection.at<float>(0,2) = (float)pr3;
        projection.at<float>(0,3) = (float)pr4;
        cout<<"\n\nProjection: "<<projection;
        backProjection = pca.backProject(projection);
        imshow("backproject", norm_0_255(backProjection.reshape(1, _shapeFreeImages[0].rows)));
        waitKey(100);
    }*/
    
    
    Mat mean = pca.mean.clone();
    Mat eigenvalues = pca.eigenvalues.clone();
    Mat eigenvectors = pca.eigenvectors.clone();
    
    _appEigenVector = eigenvectors.clone();
    _appEigenValue = eigenvalues.clone();
    _meanAppearanceRaw = mean.clone();
    
    namedWindow("avg");
    namedWindow("pc1");
    namedWindow("pc2");
    namedWindow("pc3");
    
    //_appEigenVector = norm_0_255(_appEigenVector);
    
    
    
    //Mat maskEigen = _maskForFace;
    //repeat(maskEigen.reshape(1, 1), _appEigenVector.rows, 1, maskEigen);
    //multiply(_appEigenVector, maskEigen, _appEigenVector);
    
    for (int l=0; l<_appEigenVector.rows; l++) {
        imwrite("model/eigen/ei_"+ to_string(l) +".jpg", _appEigenVector.row(l).reshape(1, _shapeFreeImages[0].rows));
    }
    
    
    imwrite("model/eigen/mean.jpg", norm_0_255(mean.reshape(1, _shapeFreeImages[0].rows)));
    
    //imshow("backproject", norm_0_255(backProjection.reshape(1, _shapeFreeImages[0].rows)));
    // The mean face:
    _meanAppearance = norm_0_255(mean.reshape(1, _shapeFreeImages[0].rows));
    imshow("avg", _meanAppearance);
    
    
    // The first three eigenfaces:
    
    imshow("pc1", norm_0_255(_appEigenVector.row(0)).reshape(1, _shapeFreeImages[0].rows));
    imshow("pc2", norm_0_255(_appEigenVector.row(1)).reshape(1, _shapeFreeImages[0].rows));
    imshow("pc3", norm_0_255(_appEigenVector.row(2)).reshape(1, _shapeFreeImages[0].rows));
    
    
}

int AAM::indexInMean(Point2f point) {
    for (int i=0; i< _meanShapeVector.size(); i++) {
        if(point == _meanShapeVector[i]) {
            return i;
        }
    }
    return -1;
}

void AAM::wrapAffineToMean(vector<vector<Point2f> > meanShapeTriangleList) {
    _meanShapeTriangleList = meanShapeTriangleList;
    vector<vector<Point2f> > trainingSetImageTriangleList;
    
    namedWindow("Warp");
    //namedWindow("Warp Data");
    
    for (int image=0; image<_noImages; image++) {
        trainingSetImageTriangleList.clear();
        for (int i=0; i<meanShapeTriangleList.size(); i++) {
            vector<Point2f> tempTriPoints;
            for (int j=0; j<3; j++) {
                tempTriPoints.push_back(_annotatedProPoints[image][indexInMean(meanShapeTriangleList[i][j])]);
            }
            trainingSetImageTriangleList.push_back(tempTriPoints);
        }
        
        
        Mat warp_final(_coordinate._coordinateSize, CV_8UC1);
        warp_final = Scalar::all(0);
        
        for(int i=0; i<trainingSetImageTriangleList.size(); i++) {
            Mat warp_mat( 2, 3, CV_32FC1 );
            warp_mat = getAffineTransform(trainingSetImageTriangleList[i], meanShapeTriangleList[i]);
            
            Mat dst(_coordinate._coordinateSize, CV_8UC1);
            Mat mask(_coordinate._coordinateSize, CV_8UC1);
            mask = Scalar::all(0);
            
            warpAffine(_trainingSet[image], dst, warp_mat, dst.size());
            
            Point meanTri[3];
            meanTri[0] = meanShapeTriangleList[i][0];
            meanTri[1] = meanShapeTriangleList[i][1];
            meanTri[2] = meanShapeTriangleList[i][2];
            
            fillConvexPoly(mask, meanTri, 3, Scalar(255,255,255));
            
            dst.copyTo(warp_final, mask);
            //imshow("Warp", warp_final);
            //waitKey(100);
        }
        _shapeFreeImages.push_back(warp_final);
    }
    _maskForFace = (_shapeFreeImages[0]>0)/255;
    erode(_maskForFace, _maskForFace, Mat(3,3, CV_8UC1));
    erode(_maskForFace, _maskForFaceWOBoundry, Mat(3,3, CV_8UC1));
    //imshow("Warp", _maskForFace*255);
    //imshow("Warp Data", _maskForFaceWOBoundry*255);
    /*for (int i=0; i<6; i++) {
        imshow("Warp Data", _shapeFreeImages[i]);
        //waitKey(1500);
    }*/
}

Mat AAM::createShapeWithTriangleIndex() {
    if (_meanShapeTriangleList.empty()) {
        cout<<"\nError: Load mean shape triangle list.\n";
        waitKey();
    }
    _shapeWithTriangleIndex.create(_coordinate._coordinateSize, CV_8UC1);
    _shapeWithTriangleIndex = Scalar::all(0);
    _shapeWithTriangleIndex.convertTo(_shapeWithTriangleIndex, CV_8UC1);
    for (int i=0; i<_meanShapeTriangleList.size(); i++) {
        Point meanTri[3];
        meanTri[0] = _meanShapeTriangleList[i][0];
        meanTri[1] = _meanShapeTriangleList[i][1];
        meanTri[2] = _meanShapeTriangleList[i][2];
        
        Mat mask(_coordinate._coordinateSize, CV_8UC1);
        mask = Scalar::all(0);
        fillConvexPoly(mask, meanTri, 3, Scalar(255,255,255));
        threshold(mask, mask, 0, 1, THRESH_BINARY);
        erode(mask, mask, Mat(3,3,CV_8UC1));
        _shapeWithTriangleIndex =  _shapeWithTriangleIndex + (mask * (i+1));
    }
    return _shapeWithTriangleIndex;
}

vector<Point2f> AAM::getMeanShape() {
    return _meanShapeVector;
}

vector<vector<Point2f> > AAM::getMeanShapeTriangles() {
    return _meanShapeTriangleList;
}

