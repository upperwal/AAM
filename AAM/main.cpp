//
//  main.cpp
//  AAM
//
//  Created by Abhishek on 01/02/15.
//  Copyright (c) 2015 Abhishek. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "annotate.h"
#include "ASM.h"
#include "AAM.h"
#include "AAMFitting.h"
#include "coordinateSystem.h"

using namespace cv;

int main(int argc, const char * argv[]) {
    const int images = 30;
    const String folder = "IMM";
    const String prefix = "image_";
    const String postfix = "jpg";
    
    vector<vector <Point2f> > annotatedPoints;
    vector<Point2f> meanShape;
    vector<Mat> result;
    
    //Annotation
    
    
    Annotation annotate(images, folder, prefix, postfix);
    
    //annotatedPoints = annotate.annotateImage();
    //annotate.savePoints();
    
    annotatedPoints = annotate.loadPoints();
    //annotate.displayPoints();
    
    waitKey(1000);
    //Active Shape Model
    ASM activeSM(annotatedPoints);
    activeSM.generateASM();
    activeSM.displayMean();
    waitKey(10000); //#pause 1
    //activeSM.displayProcrustesResult(900, Scalar(300,200));
    vector<vector<Point2f> > meanShapeTrianglesList = activeSM.findMeanDelaunayTriangles();
    
    for (int i=0; i<meanShapeTrianglesList.size(); i++) {
        for (int j=0; j<meanShapeTrianglesList[i].size(); j++) {
            cout<<meanShapeTrianglesList[i][j]<<" ";
        }
        cout<<"\n";
    }
    
    activeSM.saveShapeModel("model");
   
    
    //Active Appearance Model
    AAM activeAM(activeSM.getCoordinate(), activeSM.getMeanShapeVector(), activeSM._annotatedProcustes,annotatedPoints, images, folder, prefix, postfix);
    activeAM.loadImages();
    activeAM.wrapAffineToMean(meanShapeTrianglesList);
    activeAM.computeTexture();
    
    activeAM.saveActiveModel("model");
    activeAM.createShapeWithTriangleIndex();
    waitKey(60000);
     
    
    
    AAMFitting fitting;
    fitting.loadModelData("model");
    
    fitting.startFitting(1);
    
     
    waitKey();
}