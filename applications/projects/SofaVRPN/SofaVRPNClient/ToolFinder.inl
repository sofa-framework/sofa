/*
 * ToolFinder.cpp
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#ifndef SOFAVRPNCLIENT_TOOLFINDER_INL_
#define SOFAVRPNCLIENT_TOOLFINDER_INL_

#include <ToolFinder.h>

#include <sofa/core/ObjectFactory.h>

namespace sofavrpn
{

namespace client
{

template<class Datatypes>
ToolFinder<Datatypes>::ToolFinder()
    : f_points(initData(&f_points, "points", "Incoming 3D Points"))
    , f_distance(initData(&f_distance, "distance", "Distance between the 2 ir transmetter when claws are closed"))
    , f_leftPoint(initData(&f_leftPoint, "leftPoint", "Guessed 3D position of the left part of the tool"))
    , f_rightPoint(initData(&f_rightPoint, "rightPoint", "Guessed 3D position of the right part of the tool"))
    , f_topPoint(initData(&f_topPoint, "topPoint", "Guessed 3D position and orientation of the top part of the tool"))
    , f_angle(initData(&f_angle, "angle", "Angle of the claws"))
    , f_angleArticulated(initData(&f_angleArticulated, "angleArticulated", "Angle of the claws for articulated system"))
{
    addInput(&f_points);
    addInput(&f_distance);

    addOutput(&f_leftPoint);
    addOutput(&f_rightPoint);
    addOutput(&f_topPoint);
    addOutput(&f_angle);
    addOutput(&f_angleArticulated);

    setDirtyValue();

}

template<class Datatypes>
ToolFinder<Datatypes>::~ToolFinder()
{
    // TODO Auto-generated destructor stub
}

template<class Datatypes>
void ToolFinder<Datatypes>::update()
{
    cleanDirty();

    sofa::helper::ReadAccessor< Data<VecCoord > > inPoints = f_points;
    //sofa::helper::WriteAccessor< Data<Coord > > leftPointW = f_leftPoint;
    //sofa::helper::WriteAccessor< Data<Coord > > rightPointW = f_rightPoint;
    //sofa::helper::WriteAccessor< Data<RCoord > > topPointRigid = f_topPoint;
    RCoord & topPointRigid = *f_topPoint.beginEdit();
    Coord & leftPointW = *f_leftPoint.beginEdit();
    Coord & rightPointW = *f_rightPoint.beginEdit();
    double & angleW = *f_angle.beginEdit();
    sofa::helper::vector<double> & angleArticulatedW = *f_angleArticulated.beginEdit();
    const double &distance = f_distance.getValue();

    angleArticulatedW.clear();
    angleArticulatedW.push_back(0);
    angleArticulatedW.push_back(0);
    angleArticulatedW.push_back(0);

    if (inPoints.size () != 3)
        return;

    //Get the Top Point
    std::vector<double> lengths;
    //get the (possible) top point
    double length01 = (inPoints[1] - inPoints[0]).norm();
    double length02 = (inPoints[2] - inPoints[0]).norm();
    double length12 = (inPoints[2] - inPoints[1]).norm();
    Coord topPoint, leftPoint, rightPoint;

    if ( (length01 < length02) && (length01 < length12) )
    {
        topPoint = inPoints[2];
        leftPoint = inPoints[0];
        rightPoint = inPoints[1];
        if (inPoints[1][0] < inPoints[0][0])
        {
            leftPoint = inPoints[1];
            rightPoint = inPoints[0];
        }
    }
    else if ( (length02 < length01) && (length02 < length12) )
    {
        topPoint = inPoints[1];
        leftPoint = inPoints[0];
        rightPoint = inPoints[1];
        if (inPoints[2][0] < inPoints[0][0])
        {
            leftPoint = inPoints[2];
            rightPoint = inPoints[0];
        }
    }
    else
    {
        topPoint = inPoints[0];
        leftPoint = inPoints[1];
        rightPoint = inPoints[2];
        if (inPoints[2][0] < inPoints[1][0])
        {
            leftPoint = inPoints[2];
            rightPoint = inPoints[1];
        }
    }

    double topLeftLength = (leftPoint - topPoint).norm();
    double leftRightLength = (leftPoint - rightPoint).norm();
    //
    double angle = atan( (leftRightLength - distance)/topLeftLength );

    //
    Coord xAxis = (leftPoint+rightPoint)*0.5 - topPoint;
    xAxis.normalize();
    Coord yAxis = (leftPoint - topPoint).cross(rightPoint - topPoint);
    yAxis.normalize();
    Coord zAxis = xAxis.cross(yAxis);
    zAxis.normalize();

    sofa::defaulttype::Quat q;
    q = q.createQuaterFromFrame(xAxis, yAxis, zAxis);
    topPointRigid.getCenter() = topPoint;
    topPointRigid.getOrientation() = q;

    leftPointW = leftPoint;
    rightPointW = rightPoint;
    angleW = angle;
    angleArticulatedW.clear();
    angleArticulatedW.push_back(0);
    angleArticulatedW.push_back(angle);
    angleArticulatedW.push_back(angle);

    f_topPoint.endEdit();
    f_leftPoint.endEdit();
    f_rightPoint.endEdit();
    f_angle.endEdit();
    f_angleArticulated.endEdit();
}

}

}

#endif //SOFAVRPNCLIENT_TOOLFINDER_INL_

