/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*
 * ToolTracker.cpp
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#ifndef SOFAVRPNCLIENT_TOOLTRACKER_INL_
#define SOFAVRPNCLIENT_TOOLTRACKER_INL_

#include <ToolTracker.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/gl/template.h>

namespace sofavrpn
{

namespace client
{

using namespace sofa;
using namespace sofa::defaulttype;
using namespace sofa::core::topology;

template<class Datatypes>
ToolTracker<Datatypes>::ToolTracker()
    : f_points(initData(&f_points, "points", "Incoming 3D Points"))
    , f_distances(initData(&f_distances, "distances", "Distances between each point"))
    , f_center(initData(&f_center, "center", "Tool's center"))
    , f_orientation(initData(&f_orientation, "orientation", "Tool's orientation"))
    , f_rigidCenter(initData(&f_rigidCenter, "rigidCenter", "Rigid center of the tool"))
    , f_drawTool(initData(&f_drawTool, false, "drawTool", "Draw tool's contour"))
{
    addInput(&f_points);
    addInput(&f_distances);

    addOutput(&f_center);
    addOutput(&f_orientation);
    addOutput(&f_rigidCenter);

    setDirtyValue();

}

template<class Datatypes>
ToolTracker<Datatypes>::~ToolTracker()
{
    // TODO Auto-generated destructor stub
}

template<class Datatypes>
void ToolTracker<Datatypes>::update()
{
    cleanDirty();

    typedef BaseMeshTopology::Edge Edge;
    typedef BaseMeshTopology::PointID PointID;

    sofa::helper::ReadAccessor< Data<VecCoord > > inPoints = f_points;
    sofa::helper::ReadAccessor< Data<helper::vector<double> > > realDistances = f_distances;
    //sofa::helper::WriteAccessor< Data<Coord > > centerPoint = f_center;
    Coord centerPoint = *f_center.beginEdit();
    //sofa::helper::WriteAccessor< Data<Quat > > orientation = f_orientation;
    Quat& orientation = *f_orientation.beginEdit();

    //sofa::helper::WriteAccessor< Data<RCoord > > rigidPoint = f_rigidCenter;
    RCoord& rigidPoint = *f_rigidCenter.beginEdit();

    //double & result_angle = *f_angle.beginEdit();
    //const double &distance = f_distance.getValue();

    if ( (inPoints.size() != realDistances.size()) || (inPoints.size () != 3))
    {
        serr << "Tool Finder : not enough given point or distance ..." << sendl;
        rigidPoint = RCoord();
        centerPoint = Coord();
        orientation = Quat();

        f_rigidCenter.endEdit();
        f_orientation.endEdit();
        return;
    }

    //Compute distances from incoming points
    helper::vector<double> computedDistances;
    helper::vector<Edge> computedEdges;
    for (unsigned int i=0 ; i<inPoints.size() ; i++)
    {
        Vec3f v = inPoints[((i+1)%realDistances.size())] - inPoints[i];
        computedDistances.push_back(v.norm());
        computedEdges.push_back(Edge(i, ((i+1)%realDistances.size())));
    }

    //Guess current mapping between incoming edges and real edges
    std::map<Edge, Edge> mapEdges;

    //
    unsigned int table[18] =
    {
        0, 1, 2,
        0, 2, 1,
        1, 0, 2,
        1, 2, 0,
        2, 0, 1,
        2, 1, 0,
    };

    double min = 99999.0;
    unsigned int bestCombi=0;

    for (unsigned int i=0 ; i<18 ; i+=3)
    {
        double temp;
        temp = fabs (realDistances[0] - computedDistances[table[i]])
                + fabs (realDistances[1] - computedDistances[table[i+1]])
                + fabs (realDistances[2] - computedDistances[table[i+2]]);

        if(temp < min)
        {
            min = temp;
            bestCombi = i;
        }
    }
    for (unsigned int i=0 ; i<3 ; i++)
    {
        Edge currentRealEdge;
        currentRealEdge = Edge(i, ((i+1)%realDistances.size()) );

        mapEdges[currentRealEdge] = computedEdges[table[bestCombi+i]];
        //std::cout << currentRealEdge << " <-->" << computedEdges[table[bestCombi+i]] << std::endl;
    }

    /*
     * 	double EPSILON = 0.0000001;
    for (unsigned int i=0 ; i<realDistances.size() ; i++)
    {
    	double dist = realDistances[i];
    	Edge currentRealEdge;
    	currentRealEdge = Edge(i, ((i+1)%realDistances.size()) );

    	unsigned int minIndex=0;
    	double min = fabs(dist - computedDistances[minIndex]);
    	std::cout << "Diff? " << 0 << " " << min << std::endl;
    	//search the nearest distances we have
    	for (unsigned int j=1 ; j<computedDistances.size() ; j++)
    	{
    		std::cout << "Diff? " << j << " " << fabs(dist - computedDistances[j]) << std::endl;
    		if(fabs(dist - computedDistances[j]) < min )
    		{
    			min = fabs(dist - computedDistances[j]);
    			minIndex = j;
    		}
    	}
    	mapEdges[currentRealEdge] = computedEdges[minIndex];
    	std::cout << currentRealEdge << " <-->" << computedEdges[minIndex]<< " -> error " << min << std::endl;
    }
    */

    //Finally, get the real points from the mapping
    std::map<PointID, Coord> mapRealPoints;
    std::map<Edge, Edge>::const_iterator edgeIt = mapEdges.begin();

    Edge oldRealEdge = (*edgeIt).first;
    Edge oldIREdge = (*edgeIt).second;
    //edgeIt++;

    Edge currentRealEdge, currentIREdge;
    PointID commonRealPoint, commonIRPoint;

    for (edgeIt++ ; edgeIt != mapEdges.end(); edgeIt++)
        //for (edgeIt++ ; mapRealPoints.size() < inPoints.size(); edgeIt++)
    {
        currentRealEdge = (*edgeIt).first;
        currentIREdge = (*edgeIt).second;

        //We know that edges are consecutive
        if (currentRealEdge[0] == oldRealEdge[0] || currentRealEdge[0] == oldRealEdge[1])
            commonRealPoint = currentRealEdge[0];
        else
            commonRealPoint = currentRealEdge[1];

        if (currentIREdge[0] == oldIREdge[0] || currentIREdge[0] == oldIREdge[1])
            commonIRPoint = currentIREdge[0];
        else
            commonIRPoint = currentIREdge[1];

        oldRealEdge = currentRealEdge;
        oldIREdge = currentIREdge;

        mapRealPoints[commonRealPoint] = inPoints[commonIRPoint];
        //std::cout << commonRealPoint << " -> " << inPoints[commonIRPoint] << std::endl;
    }
    //get the last point
    edgeIt = mapEdges.begin();
    currentRealEdge = (*edgeIt).first;
    currentIREdge = (*edgeIt).second;

    //We know that edges are consecutive
    if (currentRealEdge[0] == oldRealEdge[0] || currentRealEdge[0] == oldRealEdge[1])
        commonRealPoint = currentRealEdge[0];
    else
        commonRealPoint = currentRealEdge[1];

    if (currentIREdge[0] == oldIREdge[0] || currentIREdge[0] == oldIREdge[1])
        commonIRPoint = currentIREdge[0];
    else
        commonIRPoint = currentIREdge[1];

    oldRealEdge = currentRealEdge;
    oldIREdge = currentIREdge;

    mapRealPoints[commonRealPoint] = inPoints[commonIRPoint];
    //std::cout << commonRealPoint << " -> " << inPoints[commonIRPoint] << std::endl;

    //not generic code ...
    //assume that point :
    //0 is left point from above view of the tool
    //1 is right ...
    //2 is top
    // 0-----1
    //    |
    //    |
    //    2

    //Compute orientation
    Coord leftPoint = mapRealPoints[0];
    Coord rightPoint = mapRealPoints[1];
    Coord topPoint = mapRealPoints[2];

    Coord xAxis = topPoint - leftPoint;
    xAxis.normalize();
    Coord yAxis = (leftPoint - topPoint).cross(rightPoint - topPoint);
    yAxis.normalize();
    Coord zAxis = xAxis.cross(yAxis);
    zAxis.normalize();

    sofa::defaulttype::Quat q;
    q = q.createQuaterFromFrame(xAxis, yAxis, zAxis);
    q.normalize();
    //Compute center
    centerPoint = topPoint;
    orientation = q;
    rigidPoint.getCenter() = topPoint;
    rigidPoint.getOrientation() = q;
    /*

    	Real truc = dot((topPoint - leftPoint), (rightPoint-leftPoint));
    	Coord projection = leftPoint+((rightPoint-leftPoint)*truc);
    	Coord centerTool = (topPoint+projection)/2.0;


    	Coord xAxis = (topPoint - projection);
    	xAxis.normalize();
    	Coord yAxis = rightPoint -leftPoint;
    	yAxis.normalize();
    	Coord zAxis = xAxis.cross(yAxis);
    	zAxis.normalize();

    	//std::cout << "X: " << xAxis << std::endl;
    	//std::cout << "Y: " << yAxis << std::endl;
    	//std::cout << "Z: " << zAxis << std::endl;
    	centerPoint = centerTool;
    	rigidPoint.getCenter() = centerTool;

    	sofa::defaulttype::Quat q;
    	q = q.createQuaterFromFrame(xAxis, yAxis, zAxis);
    	q.normalize();

    	orientation = q;
    	rigidPoint.getOrientation() = q;

    	f_rigidCenter.endEdit();
    	f_orientation.endEdit();

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
    	else
    		if ( (length02 < length01) && (length02 < length12) )
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
    	*/
}

template<class Datatypes>
void ToolTracker<Datatypes>::draw()
{
    sofa::helper::ReadAccessor< Data<VecCoord > > inPoints = f_points;

    if (!f_drawTool.getValue() || inPoints.empty()) return ;

    glDisable(GL_LIGHTING);

    glLineWidth(2);
    glBegin(GL_LINES);
    for (unsigned int i=0 ; i<inPoints.size() ; i++)
    {
        glColor3f(1.0,1.0,1.0);
        helper::gl::glVertexT(inPoints[i]);
        helper::gl::glVertexT(inPoints[((i+1)%inPoints.size())]);
    }

    glEnd();
    glEnable(GL_LIGHTING);

}

}

}

#endif //SOFAVRPNCLIENT_TOOLTRACKER_INL_

