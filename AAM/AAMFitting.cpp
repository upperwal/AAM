//
//  AAMFitting.cpp
//  AAM
//
//  Created by Abhishek on 07/03/15.
//  Copyright (c) 2015 Abhishek. All rights reserved.
//

#include "AAMFitting.h"

using namespace cv;

AAMFitting::AAMFitting() {
    
}

AAMFitting::AAMFitting(AAM &aam) {
    _aam = aam;
}

void AAMFitting::startFitting(int noIterations) {
    //namedWindow("FinalPts");
    namedWindow("Backp");
    _noIterations = noIterations;
    preCompute();//1
    initialize();//2
    detectFaceRegionAndAlignFittingShape();//3
    iterate();
}

void AAMFitting::preCompute() {
    createWarpJacobian();
}

void AAMFitting::initialize() {
    _fittingShape = _aam.getMeanShape();
    //cout<<
}

void AAMFitting::iterate() {
    Mat eigenvectors = _aam._appEigenVector;
    eigenvectors.convertTo(eigenvectors, CV_64FC1);
    Mat eigenvalues = _aam._appEigenValue;
    Mat mean = _aam._meanAppearanceRaw;
    mean.convertTo(mean, CV_64FC1);
    /*mean = _aam.norm_0_255(mean);
    mean = mean.reshape(1, _aam._coordinate._coordinateSize.height);
    cout<<"Type: "<<mean;
    imshow("Backp", mean);*/
    //eigenvectors.convertTo(eigenvectors, CV_64FC1);
    Mat c, dc, backProjectData, backProjectDataAsCol, Jfsic, Hfsic, inv_Hfsic, dqp, errorImage;
    for (int i=0; i<_noIterations; i++) {
        Mat warpAff = wrapAffineToMean();
        warpAff = warpAff.reshape(1, 1);
        
        if (i==0) {
            c = project(mean, eigenvectors, warpAff);
            cout<<c;
        }
        else {
            //c.convertTo(c, CV_64FC1);
            c = c + dc.t();
        }
        backProjectData = backProject(mean, eigenvectors, c);
        
        backProjectData.copyTo(backProjectDataAsCol);
        backProjectDataAsCol.convertTo(backProjectDataAsCol, CV_64FC1);
        backProjectData = backProjectData.reshape(1, _aam._coordinate._coordinateSize.height);
        //backProjectData = _aam.norm_0_255(backProjectData);
        findTemplateGradient(backProjectData);
        imshow("Backp", _aam.norm_0_255(backProjectData));
        imageJacobian();
        
        //eigenvectors.convertTo(eigenvectors, CV_64FC1);
        Jfsic = _imageJacobian - eigenvectors.t() * (eigenvectors*_imageJacobian);
        
        Hfsic = Jfsic.t() * Jfsic;
        inv_Hfsic = Hfsic.inv();
        
        warpAff.convertTo(warpAff, CV_64FC1);
        mean.convertTo(mean, CV_64FC1);
        errorImage = warpAff - mean;
        
        Mat bb = Jfsic.t() * errorImage.t();
        dqp = inv_Hfsic * bb;
        
        dc = eigenvectors * (warpAff.t() - backProjectDataAsCol.t() - (_imageJacobian * dqp));
        
        computeWarpUpdate(dqp);
        
        cout<<"Jacob: "<<dqp.size();
    }
}

void AAMFitting::computeWarpUpdate(Mat dqp) {
    vector<Point2f> meanShape = _aam.getMeanShape();
    Mat meanMat(meanShape), dr, dp, ds0, S, Q, t, r, p, s, s_new(meanMat.rows, 2, CV_64FC1, Scalar::all(0.00));
    dr = -dqp.rowRange(0, 3);
    dp = -dqp.rowRange(4, dqp.rows);
    Q = _similarityEigen.colRange(0, 3);
    S = _similarityEigen.colRange(4, dqp.rows);
    ds0 = S * dp + Q * dr;
    meanMat.convertTo(meanMat, CV_64FC1);
    cout<<"\n"<<ds0;;
    t = ds0 + meanMat.reshape(1, meanMat.rows*2);
    t = t.reshape(1, meanMat.rows);
    cout<<"T: "<<t.row(0)<<"\n\n";
    for (int i=0; i<pointIndexingOfEachTriangleAroundPoint.size(); i++) {
        Mat out(1,2, t.row(i).type());
        out = Scalar::all(0.0);
        
        vector<double> xNewVector, yNewvector;
        
        for (int j=0; j<pointIndexingOfEachTriangleAroundPoint[i].size(); j++) {
            Point2f meanShapeTriangle[3], currentShapeTriangle[3];
            
            /*for (int o=0; o<meanShape.size(); o++) {
                cout<<"\n"<<_fittingShape[o];
            }*/
            
            meanShapeTriangle[0] = meanShape[pointIndexingOfEachTriangleAroundPoint[i][j][0]];
            meanShapeTriangle[1] = meanShape[pointIndexingOfEachTriangleAroundPoint[i][j][1]];
            meanShapeTriangle[2] = meanShape[pointIndexingOfEachTriangleAroundPoint[i][j][2]];
            
            currentShapeTriangle[0] = _fittingShape[pointIndexingOfEachTriangleAroundPoint[i][j][0]];
            currentShapeTriangle[1] = _fittingShape[pointIndexingOfEachTriangleAroundPoint[i][j][1]];
            currentShapeTriangle[2] = _fittingShape[pointIndexingOfEachTriangleAroundPoint[i][j][2]];
            
            double denominator = (meanShapeTriangle[1].x - meanShapeTriangle[0].x) * (meanShapeTriangle[2].y - meanShapeTriangle[0].y) - (meanShapeTriangle[1].y - meanShapeTriangle[0].y) * (meanShapeTriangle[2].x - meanShapeTriangle[0].x);
            
            double a0 = currentShapeTriangle[0].x + ((meanShapeTriangle[0].y * (meanShapeTriangle[2].x - meanShapeTriangle[1].x) - meanShapeTriangle[1].x * (meanShapeTriangle[2].y - meanShapeTriangle[0].y)) * (currentShapeTriangle[1].x - currentShapeTriangle[0].x) + (meanShapeTriangle[0].x * (meanShapeTriangle[1].y - meanShapeTriangle[0].y) - meanShapeTriangle[0].y * (meanShapeTriangle[1].x - meanShapeTriangle[0].x)) * (currentShapeTriangle[2].x - currentShapeTriangle[0].x)) / denominator;
            
            double a1 = ((meanShapeTriangle[2].y - meanShapeTriangle[0].y) * (currentShapeTriangle[1].x - currentShapeTriangle[0].x) - (meanShapeTriangle[1].y - meanShapeTriangle[0].y) * (currentShapeTriangle[2].x - currentShapeTriangle[0].x)) / denominator;
            
            double a2 = ((meanShapeTriangle[1].x - meanShapeTriangle[0].x) * (currentShapeTriangle[2].x - currentShapeTriangle[0].x) - (meanShapeTriangle[2].x - meanShapeTriangle[0].x) * (currentShapeTriangle[1].x - currentShapeTriangle[0].x)) / denominator;
            
            double a3 = currentShapeTriangle[0].y + ((meanShapeTriangle[0].y * (meanShapeTriangle[2].x - meanShapeTriangle[1].x) - meanShapeTriangle[1].x * (meanShapeTriangle[2].y - meanShapeTriangle[0].y)) * (currentShapeTriangle[1].y - currentShapeTriangle[0].y) + (meanShapeTriangle[0].x * (meanShapeTriangle[1].y - meanShapeTriangle[0].y) - meanShapeTriangle[0].y * (meanShapeTriangle[1].x - meanShapeTriangle[0].x)) * (currentShapeTriangle[2].y - currentShapeTriangle[0].y)) / denominator;
            
            double a4 = ((meanShapeTriangle[2].y - meanShapeTriangle[0].y) * (currentShapeTriangle[1].y - currentShapeTriangle[0].y) - (meanShapeTriangle[1].y - meanShapeTriangle[0].y) * (currentShapeTriangle[2].y - currentShapeTriangle[0].y)) / denominator;
            
            double a5 = ((meanShapeTriangle[1].x - meanShapeTriangle[0].x) * (currentShapeTriangle[2].y - currentShapeTriangle[0].y) - (meanShapeTriangle[2].x - meanShapeTriangle[0].x) * (currentShapeTriangle[1].y - currentShapeTriangle[0].y)) / denominator;
            
            Mat temp = t.row(i);
            double xnew,ynew;
            xnew = a0 + a1 * temp.col(0).at<double>(0,0) + a2 * temp.col(1).at<double>(0,0);
            ynew = a3 + a4 * temp.col(0).at<double>(0,0) + a5 * temp.col(1).at<double>(0,0);
            
            //if(xnew >= 0 || ynew >= 0) {
                xNewVector.push_back(xnew);
                yNewvector.push_back(ynew);
            //}
            
            //cout<<"\n\nMeanShape: "<<currentShapeTriangle[0]<<" "<<currentShapeTriangle[1]<<" "<<currentShapeTriangle[2];
            
            //Mat transform = getAffineTransform(currentShapeTriangle, meanShapeTriangle);
            
            //cout<<"\n\nTrans:\n"<<transform;
            
            
            /*Mat aa(3,1,CV_64FC1, Scalar::all(1.0));
            aa.row(0).at<double>(0,0) = temp.col(0).at<double>(0,0);
            aa.row(1).at<double>(0,0) = temp.col(1).at<double>(0,0);*/
            /*out.at<double>(0,0) = xnew;
            out.at<double>(0,1) = ynew;
            cout<<"\n\nOut: \n("<<i<<"): "<<out;*/
            //warpAffine(t.row(i), out, transform, t.row(i).size());
            
        }
        std::sort(xNewVector.begin(), xNewVector.begin()+xNewVector.size());
        std::sort(yNewvector.begin(), yNewvector.begin()+yNewvector.size());
        cout<<"\nNew: "<<xNewVector.size()<<" "<<yNewvector.size();
        //out = out/pointIndexingOfEachTriangleAroundPoint[i].size();
        s_new.row(i).at<double>(0,0) = xNewVector[xNewVector.size()/2];
        s_new.row(i).at<double>(0,1) = yNewvector[yNewvector.size()/2];
    }
    
    cout<<"\n\n"<<s_new;
    
    r = Q.t() * (s_new.reshape(1,meanMat.rows*2) - meanMat.reshape(1, meanMat.rows*2));
    p = S.t() * (s_new.reshape(1,meanMat.rows*2) - meanMat.reshape(1, meanMat.rows*2));
    s = meanMat.reshape(1, meanMat.rows*2) + S * p + Q * r;
    s = s.reshape(1, meanMat.rows);
    
    cout<<"\n\nMy final:\n"<<_fittingShape;
    for (int i=0; i<s.rows; i++) {
        _fittingShape[i].x = s.row(i).col(0).at<double>(0,0);// + (double)_faceRegion.x;
        _fittingShape[i].y = s.row(i).col(1).at<double>(0,0);// + (double)_faceRegion.y;
    }
    cout<<"\n\nMy final:\n"<<_fittingShape;
    
    //displayThesePoints("FinalPts", _inputImage, _fittingShape);
    
    Mat image;
    _inputImage.copyTo(image);
    for(Point2f point : _fittingShape) {
        Scalar color(255,0,255);
        circle(image, point, 1, color, 2, -1);
    }
    //imshow("FinalPts", image);
    //waitKey(2000);
}

Mat AAMFitting::project(Mat mean, Mat evec, Mat data) {
    Mat tempDiff;
    data.convertTo(data, mean.type());
    subtract(data, mean, tempDiff);
    imshow("Err", _aam.norm_0_255(tempDiff.reshape(1, _aam._coordinate._coordinateSize.height)));
    Mat c;
    //gemm(tempDiff, evec, 1, Mat(), 0, c, GEMM_2_T);
    _aam._pca.mean = mean;
    _aam._pca.eigenvectors = evec;
    c = _aam._pca.project(data);
    return c;
}

Mat AAMFitting::backProject(Mat mean, Mat evec, Mat c) {
    Mat result, temp_mean;
    //c.convertTo(c, mean.type());
    temp_mean = repeat(mean, c.rows, 1);
    cout<<"\n\nC:"<<c.size();
    cout<<_aam._pca.eigenvectors.size();
    Mat bk = _aam._pca.backProject(c); //c*evec + mean;
    //gemm(c, evec, 1, mean, 1, result, 0);
    return bk;
}

void AAMFitting::loadModelData(String path) {
    _asm.loadShapeModel(path);
    _aam.loadActiveModel(path);
}

void AAMFitting::loadTemplateImage(Mat templateImage) {
    _templateImage = templateImage;
}

void AAMFitting::findTemplateGradient(Mat data) {
    _gradientX = Mat::ones(_aam._coordinate._coordinateSize.height, _aam._coordinate._coordinateSize.width, CV_64F);
    _gradientY = Mat::ones(_aam._coordinate._coordinateSize.height, _aam._coordinate._coordinateSize.width, CV_64F);
    
    Scharr(data, _gradientX, CV_8U, 1, 0);
    Scharr(data, _gradientY, CV_8U, 0, 1);
    
    Mat maskWOB = _aam.maskForFaceWOBoundry();
    maskWOB.convertTo(maskWOB, CV_8U);
    multiply(maskWOB, _gradientX, _gradientX);
    multiply(maskWOB, _gradientY, _gradientY);
    
    //convertScaleAbs(_gradientX, _gradientX);
    //convertScaleAbs(_gradientY, _gradientY);
    
    //_gradientX.convertTo(_gradientX, CV_64FC1);
    //_gradientY.convertTo(_gradientY, CV_64FC1);
    
    namedWindow("Gradient");
    Mat temp;
    imwrite("model/dTx.jpg", _gradientX);
    imwrite("model/dTy.jpg", _gradientY);
    hconcat(_gradientX, _gradientY, temp);
    imshow("Gradient", temp);
}

void AAMFitting::imageJacobian() {
    Mat _gradientXChannels, _gradientYChannels, tp1, tp2, imgJacob;
    repeat(_gradientX, 1, _Wx_dp.channels(), _gradientXChannels);
    _gradientXChannels = _gradientXChannels.reshape(_Wx_dp.channels());
    repeat(_gradientY, 1, _Wx_dp.channels(), _gradientYChannels);
    _gradientYChannels = _gradientYChannels.reshape(_Wx_dp.channels());
    
    _gradientXChannels.convertTo(_gradientXChannels, _Wx_dp.type());
    _gradientYChannels.convertTo(_gradientYChannels, _Wy_dp.type());
    multiply(_gradientXChannels, _Wx_dp, tp1);
    multiply(_gradientYChannels, _Wy_dp, tp2);
    
    imgJacob = tp1 + tp2;
    
    _imageJacobian = imgJacob.reshape(1, _aam._coordinate._coordinateSize.height * _aam._coordinate._coordinateSize.width);
    
    //namedWindow("Image Jacob");
    vector<Mat> bb;
    split(imgJacob, bb);
    //imshow("Image Jacob", _aam.norm_0_255(bb[0]));
}

void AAMFitting::createWarpJacobian() {
    namedWindow("Mask");
    namedWindow("Jacobian");
    namedWindow("Trii");
    //cout<<"Inside create warp jacobian\n"<<_aam._folder;
    //namedWindow("Cre");
    //imshow("Cre", _aam.getMeanAppearance());
    vector<Point2f> meanShape = _aam.getMeanShape();
    vector<vector<Point2f> > meanShapeTriangles = _aam.getMeanShapeTriangles();
    /*for (int i=0; i<meanShapeTriangles.size(); i++) {
        for (int j=0; j<meanShapeTriangles[i].size(); j++) {
            cout<<"["<<i<<"]["<<j<<"]: "<<meanShapeTriangles[i][j]<<"\t";
        }
        cout<<"\n";
    }*/
    triangleIndexOfAllPoints.clear();
    vector<Point2f> triangleIndexOfThisPoint;
    vector<Mat> dw_dxy;
    Mat triangleWithIndex = _aam.createShapeWithTriangleIndex();
    imshow("Mask", triangleWithIndex>0);
    for (int i=0; i<meanShape.size(); i++) {
        Mat dw_dxy_eachPoint(_aam._coordinate._coordinateSize, CV_64FC1, Scalar::all(0.0000));
        //dw_dxy_eachPoint = 0;
        //dw_dxy_eachPoint.convertTo(dw_dxy_eachPoint, CV_64FC1);
        triangleIndexOfThisPoint.clear();
        triangleIndexOfThisPoint = getAllTrianglesAtPoint(meanShape[i], meanShapeTriangles);
        //cout<<triangleIndexOfThisPoint[0]<<" "<<triangleIndexOfThisPoint[1]<<" "<<triangleIndexOfThisPoint[2];
        triangleIndexOfAllPoints.push_back(triangleIndexOfThisPoint);
        
        for (int j=0; j<triangleIndexOfAllPoints[i].size(); j++) {
            int triangleRowIndex = triangleIndexOfAllPoints[i][j].x;
            int triangleColIndex = triangleIndexOfAllPoints[i][j].y;
            Point2f thisTrianglePoint[3];
            
            thisTrianglePoint[0].x = meanShapeTriangles[triangleRowIndex][triangleColIndex].x;
            thisTrianglePoint[0].y = meanShapeTriangles[triangleRowIndex][triangleColIndex].y;
            
            int countCol = 1;
            for (int colNo=0; colNo<3; colNo++) {
                if (colNo != triangleColIndex) {
                    thisTrianglePoint[countCol].x = meanShapeTriangles[triangleRowIndex][colNo].x;
                    thisTrianglePoint[countCol].y = meanShapeTriangles[triangleRowIndex][colNo].y;
                    countCol++;
                }
            }
            
            //cout<<"\nTriangle["<<j<<"]: "<<thisTrianglePoint[0]<<" "<<thisTrianglePoint[1]<<" "<<thisTrianglePoint[2];

            Mat trianglePositionMatrix;
            Mat mask;
            mask = (triangleWithIndex == (triangleIndexOfAllPoints[i][j].x+1));
            erode(mask, mask, Mat(3,3,CV_8UC1));
            imshow("Trii", mask>0);
            //cout<<"\n\n"<<static_cast<int>(_shapeWithTriangleIndex.at<uchar>(63, 287));
            //imshow("We", temp1);
            findNonZero(mask, trianglePositionMatrix);
            Mat trianglePositionMatrixChannels[2], trianglePositionMatrixChannelsOri[2];
            split(trianglePositionMatrix, trianglePositionMatrixChannelsOri);
            
            trianglePositionMatrixChannelsOri[0].copyTo(trianglePositionMatrixChannels[0]);
            trianglePositionMatrixChannelsOri[1].copyTo(trianglePositionMatrixChannels[1]);
            
            //cout<<"After Split: "<<(float)trianglePositionMatrixChannels[0].at<int>(605,0);
            //cout<<"\nPuri: "<<trianglePositionMatrix;
            //cout<<"\n Channels: "<<trianglePositionMatrixChannels[0];
            //vector<Point2f> temp = mat2Point2f(trianglePositionMatrix);
            
            trianglePositionMatrixChannels[0].convertTo(trianglePositionMatrixChannels[0], CV_64FC1);
            trianglePositionMatrixChannels[1].convertTo(trianglePositionMatrixChannels[1], CV_64FC1);
            
            float denominator = (thisTrianglePoint[1].x - thisTrianglePoint[0].x) * (thisTrianglePoint[2].y - thisTrianglePoint[0].y) - (thisTrianglePoint[1].y - thisTrianglePoint[0].y) * (thisTrianglePoint[2].x - thisTrianglePoint[0].x);
            //cout<<"\n\nDeno"<<denominator;
            Mat alpha(trianglePositionMatrixChannels[0].size(), CV_64FC1);
            alpha.convertTo(alpha, CV_64FC1);
            alpha = (trianglePositionMatrixChannels[0] - thisTrianglePoint[0].x) * (thisTrianglePoint[2].y - thisTrianglePoint[0].y) - (trianglePositionMatrixChannels[1] - thisTrianglePoint[0].y) * (thisTrianglePoint[2].x - thisTrianglePoint[0].x);
            //cout<<"\n\nAlpha"<<alpha;
            Mat beta(trianglePositionMatrixChannels[1].size(), CV_64FC1);
            beta = (trianglePositionMatrixChannels[1] - thisTrianglePoint[0].y) * (thisTrianglePoint[1].x - thisTrianglePoint[0].x) - (trianglePositionMatrixChannels[0] - thisTrianglePoint[0].x) * (thisTrianglePoint[1].y - thisTrianglePoint[0].y);
            //cout<<"\n\nBeta"<<beta;
            Mat dw_dxy_eachTriangle = (1 - alpha/denominator - beta/denominator);
            //cout<<"\n\ndw_dxy_each: \n"<<dw_dxy_eachTriangle;
            double loc[2];
            minMaxIdx(trianglePositionMatrixChannelsOri[1], loc);
            //cout<<"\n"<<loc[0]<<" "<<loc[1];
            
            for (int indexOfMat=0; indexOfMat<trianglePositionMatrixChannels[0].rows; indexOfMat++) {
                dw_dxy_eachPoint.at<double>((int)trianglePositionMatrixChannelsOri[1].at<int>(indexOfMat, 0), (int)trianglePositionMatrixChannelsOri[0].at<int>(indexOfMat, 0)) = (double)dw_dxy_eachTriangle.at<double>(indexOfMat,0);
                //dw_dxy_eachPoint.at<double>((int)trianglePositionMatrixChannelsOri[0].at<int>(indexOfMat, 0), (int)trianglePositionMatrixChannelsOri[0].at<int>(indexOfMat, 0)) = 2.999;
                //cout<<"PP: "<<(double)dw_dxy_eachPoint.at<double>(0,0);
                //cout<<"\n Matt: "<<indexOfMat<<" "<<dw_dxy_eachPoint.at<double>((int)trianglePositionMatrixChannelsOri[0].at<int>(indexOfMat, 0), (int)trianglePositionMatrixChannelsOri[0].at<int>(indexOfMat, 0));
            }
            //break;
        }
        dw_dxy.push_back(dw_dxy_eachPoint);
        //break;
    }
    Mat dw_dxy_multichannel;
    merge(dw_dxy, dw_dxy_multichannel);
    
    //dw_dxy[i] = abs(dw_dxy[i]);
    //normalize(dw_dxy[i], dw_dxy[i], 0, 255, NORM_MINMAX);
    //imshow("Jacobian", dw_dxy[0]);
    //cout<<"\n\n\n\nJacobian:\n"<<dw_dxy[i];
    
    //cout<<"\n\n"<<_asm.getEigenvector();
    Mat dx_dp = createSimilarityEigens();
    
    
    int meanShapeRows = (int)_asm.getMeanShapeVector().size();
    Mat Wx_dp = dw_dxy_multichannel.reshape(1, (int)dw_dxy[0].total()) * dx_dp.rowRange(0, meanShapeRows);
    Mat Wy_dp = dw_dxy_multichannel.reshape(1, (int)dw_dxy[0].total()) * dx_dp.rowRange(meanShapeRows, 2*meanShapeRows);
    Wx_dp = Wx_dp.reshape(Wx_dp.cols, dw_dxy[0].rows);
    Wy_dp = Wy_dp.reshape(Wy_dp.cols, dw_dxy[0].rows);
    
    _Wx_dp = Wx_dp;
    _Wy_dp = Wy_dp;
    
    namedWindow("Jacob");
    vector<Mat> cha;
    split(Wx_dp, cha);
    for (int i=0; i<Wx_dp.channels(); i++) {
        cha[i] *= 7;
        //cout<<"\n"<<cha[i].size()<<" " <<_aam.getMaskForFace().size()<<" "<<_aam._coordinate._coordinateSize;
        cha[i].convertTo(cha[i], CV_64FC1);
        Mat cc = _aam.getMaskForFace();
        cc.convertTo(cc, CV_64FC1);
        multiply(cha[i], cc, cha[i]);
        imshow("Jacob", cha[i]);
        waitKey(700);
    }
    setIndexOfPointOfEachTriangleAroundPoint();
}
//CODE CHECKED

Mat AAMFitting::createSimilarityEigens() {
    vector<Point2f> meanShapeV = _asm.getMeanShapeVector();
    Mat meanShape(meanShapeV);
    int meanShapeRows = (int)meanShape.rows;
    meanShape = meanShape.reshape(1);
    Mat eigenVector = _asm.getEigenvector();
    eigenVector.convertTo(eigenVector, CV_64FC1);
    Q = Mat::zeros(meanShapeRows * 2, 4, CV_64FC1);
    
    Mat temp;
    vconcat(meanShape.col(0), meanShape.col(1), temp);
    temp.copyTo(Q.col(0));
    vconcat(-meanShape.col(1), meanShape.col(0), temp);
    temp.copyTo(Q.col(1));
    vconcat(Mat::ones(meanShapeRows, 1, CV_64FC1), Mat::zeros(meanShapeRows, 1, CV_64FC1), temp);
    temp.copyTo(Q.col(2));
    vconcat(Mat::zeros(meanShapeRows, 1, CV_64FC1), Mat::ones(meanShapeRows, 1, CV_64FC1), temp);
    temp.copyTo(Q.col(3));
    Q = orthonoram(Q);
    
    hconcat(Q, eigenVector, _similarityEigen);
    _similarityEigen = orthonoram(_similarityEigen);
    
    return _similarityEigen;
}

void AAMFitting::setIndexOfPointOfEachTriangleAroundPoint() {
    vector<Point2f> meanShape = _aam.getMeanShape();
    vector<vector<Point2f> > meanShapeTriangles = _aam.getMeanShapeTriangles();
    pointIndexingOfEachTriangleAroundPoint.clear();
    for (int i=0; i<triangleIndexOfAllPoints.size(); i++) {
        vector<vector<int> > locTriangle;
        for (int j=0; j<triangleIndexOfAllPoints[i].size(); j++) {
            int row = triangleIndexOfAllPoints[i][j].x;
            vector<int> tempLoc;
            for (int k=0; k<3; k++) {
                Point2f point = meanShapeTriangles[row][k];
                tempLoc.push_back(indexInMean(point));
            }
            locTriangle.push_back(tempLoc);
        }
        pointIndexingOfEachTriangleAroundPoint.push_back(locTriangle);
    }
}

Mat AAMFitting::orthonoram(Mat data) {
    Mat U,S,Vt;
    SVD::compute(data, S, U, Vt);
    return U;
}

vector<Point2f> AAMFitting::getAllTrianglesAtPoint(Point2f point, vector<vector<Point2f> > meanShapeTriangles) {
    
    vector<Point2f> triangleIndex;
    bool flag;
    for (int i=0; i<meanShapeTriangles.size(); i++) {
        Point2f eachPoint;
        flag = false;
        for (int j=0; j<3; j++) {
            if (meanShapeTriangles[i][j] == point) {
                eachPoint.x = i;
                eachPoint.y = j;
                flag = true;
            }
        }
        if (flag == true) {
            triangleIndex.push_back(eachPoint);
            //cout<<"\n\nPush: "<<eachPoint;
        }
    }
    return triangleIndex;
}

void AAMFitting::detectFaceRegionAndAlignFittingShape() {
    CascadeClassifier faceCascade;
    vector<Rect> faces;
    Mat image, ori;
    image = imread("images/IMM/image_1.jpg");
    image.copyTo(ori);
    faceCascade.load("haarcascade_frontalface_alt.xml");
    namedWindow("face");
    resize(image, image, Size(image.cols/3, image.rows/3));
    cout<<image.size();
    cvtColor(image, image, COLOR_BGR2GRAY);
    faceCascade.detectMultiScale(image, faces);
    faces[0].y += floor((20.0/100)*faces[0].y);
    faces[0].x *= 3;
    faces[0].y *= 3;
    faces[0].height *= 3;
    faces[0].width *= 3;
    _faceRegion = faces[0];
    
    Mat fittingShapeMat(_fittingShape);
    fittingShapeMat = fittingShapeMat.reshape(1, fittingShapeMat.rows);
    double scaleFactor = (double)_faceRegion.width/_aam._coordinate._coordinateSize.width;
    fittingShapeMat *= scaleFactor;
    cout<<(_aam._coordinate._coordinateSize.width - _faceRegion.width)/2;
    fittingShapeMat.col(0) += _faceRegion.x;// + 80;
    fittingShapeMat.col(1) += _faceRegion.y;// + (_aam._coordinate._coordinateSize.height - _faceRegion.height)/2;
    fittingShapeMat = fittingShapeMat.reshape(2);
    fittingShapeMat.copyTo(_fittingShape);
    resize(image, image, Size(image.cols*3, image.rows*3));
    image.copyTo(_inputImage);
    
    displayThesePoints("Initial", ori, _fittingShape);
    
    Mat fc = ori(faces[0]);
    imshow("face", fc);
}

void AAMFitting::displayThesePoints(String name, Mat image, vector<Point2f> points) {
    for(Point2f point : points) {
        Scalar color(255,0,255);
        circle(image, point, 1, color, 2, -1);
    }
    namedWindow(name);
    imshow(name, image);
}

Mat AAMFitting::wrapAffineToMean() {
    vector<vector<Point2f> > meanShapeTriangleList = _aam.getMeanShapeTriangles();
    vector<vector<Point2f> > trainingSetImageTriangleList;

    namedWindow("Warp Data");
    
    trainingSetImageTriangleList.clear();
    for (int i=0; i<meanShapeTriangleList.size(); i++) {
        vector<Point2f> tempTriPoints;
        for (int j=0; j<3; j++) {
            tempTriPoints.push_back(_fittingShape[indexInMean(meanShapeTriangleList[i][j])]);
        }
        trainingSetImageTriangleList.push_back(tempTriPoints);
    }
    
    
    Mat warp_final(_aam._coordinate._coordinateSize, CV_8UC1);
    warp_final = Scalar::all(0);
    
    for(int i=0; i<trainingSetImageTriangleList.size(); i++) {
        Mat warp_mat( 2, 3, CV_32FC1 );
        warp_mat = getAffineTransform(trainingSetImageTriangleList[i], meanShapeTriangleList[i]);
        
        Mat dst(_aam._coordinate._coordinateSize, CV_8UC1);
        Mat mask(_aam._coordinate._coordinateSize, CV_8UC1);
        mask = Scalar::all(0);
        
        warpAffine(_inputImage, dst, warp_mat, dst.size());
        
        Point meanTri[3];
        meanTri[0] = meanShapeTriangleList[i][0];
        meanTri[1] = meanShapeTriangleList[i][1];
        meanTri[2] = meanShapeTriangleList[i][2];
        
        fillConvexPoly(mask, meanTri, 3, Scalar(255,255,255));
        
        dst.copyTo(warp_final, mask);
        imshow("Warp Data", warp_final);
    }
    //multiply(warp_final, _aam.getMaskForFace(), warp_final);
    //imshow("Warp Data", warp_final);
    /*for (int i=0; i<6; i++) {
     imshow("Warp Data", _shapeFreeImages[i]);
     //waitKey(1500);
     }*/
    return warp_final;
}

int AAMFitting::indexInMean(Point2f point) {
    for (int i=0; i< _aam._meanShapeVector.size(); i++) {
        if(point == _aam._meanShapeVector[i]) {
            return i;
        }
    }
    return -1;
}



























































































/*void AAMFitting::computeSteepestDescentImages() {
    const int ws = _templateImage.cols;
    const int hs = _templateImage.rows;
    
    Mat Xcoord = Mat(1, ws, CV_32F);
    Mat Ycoord = Mat(hs, 1, CV_32F);
    Mat Xgrid = Mat(hs, ws, CV_32F);
    Mat Ygrid = Mat(hs, ws, CV_32F);
    float* XcoPtr = Xcoord.ptr<float>(0);
    float* YcoPtr = Ycoord.ptr<float>(0);
    int j;
    for (j=0; j<ws; j++)
        XcoPtr[j] = (float) j;
    for (j=0; j<hs; j++)
        YcoPtr[j] = (float) j;
    
    repeat(Xcoord, ws, 1, Xgrid);
    repeat(Ycoord, 1, hs, Ygrid);
    
    Xcoord.release();
    Ycoord.release();
    
    Mat jacobian = Mat(hs, ws*6, CV_32F);
    
    const int w = _gradientXTemplate.cols;
    
    _gradientXTemplate.convertTo(_gradientXTemplate, CV_32F);
    _gradientYTemplate.convertTo(_gradientYTemplate, CV_32F);
    
    //compute Jacobian blocks (6 blocks)
    jacobian.colRange(0,w) = _gradientXTemplate.mul(Xgrid);//1
    cout<<w;
    jacobian.colRange(w,2*w) = _gradientYTemplate.mul(Xgrid);//2
    jacobian.colRange(2*w,3*w) = _gradientXTemplate.mul(Ygrid);//3
    jacobian.colRange(3*w,4*w) = _gradientYTemplate.mul(Ygrid);//4
    normalize(jacobian, jacobian, 0, 255, NORM_MINMAX, CV_32F);
    convertScaleAbs(jacobian, jacobian);
    _gradientXTemplate.copyTo(jacobian.colRange(4*w,5*w));//5
    _gradientYTemplate.copyTo(jacobian.colRange(5*w,6*w));
    
    
    namedWindow("Jacobian");
    
    Mat hessian = Mat(6, 6, CV_32F);
    project_onto_jacobian_ECC(jacobian, jacobian, hessian);
    Mat h(ws*6,ws*6,CV_32F), hi;
    //jacobian.convertTo(jacobian, CV_32F);
    //h = jacobian.t()*jacobian;
    //normalize(hessian, hessian, 0, 255, NORM_MINMAX, CV_32F);
    //cout<<"\nDet: "<<determinant(h);
    hi = hessian.inv();
    //normalize(hi, hi, 0, 255, NORM_MINMAX, CV_32F);
    imshow("\nJacobian", hessian);
    cout<<"\nHessian: "<<hessian;
    //cout<<"\nHi "<<hi.colRange(0, 20);
}

void AAMFitting::project_onto_jacobian_ECC(const Mat& src1, const Mat& src2, Mat& dst)
{

    CV_Assert(src1.rows == src2.rows);
    CV_Assert((src1.cols % src2.cols) == 0);
    int w;
    
    float* dstPtr = dst.ptr<float>(0);
    
    if (src1.cols !=src2.cols){//dst.cols==1
        w  = src2.cols;
        for (int i=0; i<dst.rows; i++){
            dstPtr[i] = (float) src2.dot(src1.colRange(i*w,(i+1)*w));
        }
    }
    
    else {
        CV_Assert(dst.cols == dst.rows); //dst is square (and symmetric)
        w = src2.cols/dst.cols;
        Mat mat;
        for (int i=0; i<dst.rows; i++){
            
            mat = Mat(src1.colRange(i*w, (i+1)*w));
            dstPtr[i*(dst.rows+1)] = (float) pow(norm(mat),2); //diagonal elements
            
            for (int j=i+1; j<dst.cols; j++){ //j starts from i+1
                dstPtr[i*dst.cols+j] = (float) mat.dot(src2.colRange(j*w, (j+1)*w));
                dstPtr[j*dst.cols+i] = dstPtr[i*dst.cols+j]; //due to symmetry
            }
        }
    }
}*/