//
//  coordinateSystem.cpp
//  AAM
//
//  Created by Abhishek on 23/02/15.
//  Copyright (c) 2015 Abhishek. All rights reserved.
//

#include "coordinateSystem.h"

void Coordinate::setCoordinateSize(vector<Point2f> meanShape) {
    Mat meanShapeMat(meanShape);
    meanShapeMat = meanShapeMat.reshape(1,(int)meanShape.size());
    minMaxLoc(meanShapeMat.col(0), &minX, &maxX);
    minMaxLoc(meanShapeMat.col(1), &minY, &maxY);
    Size coordSize((int)(maxX-minX+1), (int)(maxY-minY+1));
    _coordinateSize = coordSize;
    cout<<"Coordinate Size:"<<"Width: "<<_coordinateSize.width<<" Height:"<<_coordinateSize.height<<"\n";
}

Scalar Coordinate::getMinXY() {
    Scalar minXY(minX, minY);
    return minXY;
}