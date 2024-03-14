/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once
#include <sofa/component/engine/generate/RandomPointDistributionInSurface.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>
#include <cstdlib>
#include <ctime>
#include <climits>

namespace sofa::component::engine::generate
{

template <class DataTypes>
RandomPointDistributionInSurface<DataTypes>::RandomPointDistributionInSurface()
    : initialized(false)
    , randomSeed( initData (&randomSeed, (unsigned int) 0, "randomSeed", "Set a specified seed for random generation (0 for \"true pseudo-randomness\" ") )
    , isVisible( initData (&isVisible, bool (true), "isVisible", "is Visible ?") )
    , drawOutputPoints( initData (&drawOutputPoints, bool (false), "drawOutputPoints", "Output points visible ?") )
    , minDistanceBetweenPoints( initData (&minDistanceBetweenPoints, Real (0.1), "minDistanceBetweenPoints", "Min Distance between 2 points (-1 for true randomness)") )
    , numberOfInPoints( initData (&numberOfInPoints, (unsigned int) 10, "numberOfInPoints", "Number of points inside") )
    , numberOfTests( initData (&numberOfTests, (unsigned int) 5, "numberOfTests", "Number of tests to find if the point is inside or not (odd number)") )
    , f_vertices( initData (&f_vertices, "vertices", "Vertices") )
    , f_triangles( initData (&f_triangles, "triangles", "Triangles indices") )
    , f_inPoints( initData (&f_inPoints, "inPoints", "Points inside the surface") )
    , f_outPoints( initData (&f_outPoints, "outPoints", "Points outside the surface") )
    , safeCounter(0), safeLimit(UINT_MAX)
{
    addInput(&f_triangles);
    addInput(&f_vertices);

    addOutput(&f_inPoints);
}

template <class DataTypes>
void RandomPointDistributionInSurface<DataTypes>::init()
{
    const unsigned int nb = numberOfTests.getValue();
    if (nb%2 == 0)
    {
        msg_warning() << "even number of tests, adding an other ...";
        numberOfTests.setValue(nb+1);
    }

    // initialize random seed
    if (randomSeed.getValue() == 0)
    {
        randomSeed.setValue((unsigned int)time(nullptr));
    }

    //srand(randomSeed.getValue());
    rg.initSeed(randomSeed.getValue());

    generateRandomDirections();

    safeLimit = numberOfInPoints.getValue()*numberOfInPoints.getValue()*numberOfInPoints.getValue()*numberOfInPoints.getValue();

    setDirtyValue();
}

template <class DataTypes>
void RandomPointDistributionInSurface<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void RandomPointDistributionInSurface<DataTypes>::getBBox(Vec3 &minBBox, Vec3 &maxBBox)
{
    const VecCoord& vertices = f_vertices.getValue();

    if (vertices.size() > 0)
    {
        minBBox = vertices[0];
        maxBBox = vertices[0];

        for (unsigned int i=1; i<vertices.size() ; i++)
        {
            for (unsigned int j=0; j<3 ; j++)
            {
                if (vertices[i][j] < minBBox[j])
                    minBBox[j] = vertices[i][j];
                if (vertices[i][j] > maxBBox[j])
                    maxBBox[j] = vertices[i][j];
            }
        }
    }
}

template <class DataTypes>
void RandomPointDistributionInSurface<DataTypes>::generateRandomDirections()
{
    /*Real d[3];
    for (unsigned int i=0 ;i<3 ;i++)
        d[i] = (2.0*((Real) rand())/RAND_MAX) - 1.0; //[-1; 1]

    for (unsigned int i=0 ;i<numberOfTests.getValue() ;i++)
    {
        Vec3 v(d[i%3], d[(i+1)%3], d[(i+2)%3]);
        directions.push_back(v);
    }
    */

    Vec3 d;
    for (unsigned int i=0 ; i<numberOfTests.getValue() ; i++)
    {
        for (unsigned int i=0 ; i<3 ; i++)
            //d[i] = (2.0*((Real) rand())/RAND_MAX) - 1.0; //[-1; 1]
            d[i] = (Real)rg.random<double>(-1.0,1.0); //[-1; 1]

        directions.push_back(d);
    }


}

template <class DataTypes>
type::Vec<3,typename DataTypes::Real> RandomPointDistributionInSurface<DataTypes>::generateRandomPoint(const Vec3 &minBBox, const Vec3 &maxBBox)
{
    Vec3 r;
    for (unsigned int i= 0 ; i<3 ; i++)
        //r[i] = (minBBox[i] + ((maxBBox[i] - minBBox[i])*rand())/RAND_MAX);
        r[i] = (Real)rg.random<double>(minBBox[i], maxBBox[i]);

    return r;
}

template <class DataTypes>
bool RandomPointDistributionInSurface<DataTypes>::isInside(Coord p)
{
    using sofa::core::topology::BaseMeshTopology;

    const VecCoord& vertices = f_vertices.getValue();
    const type::vector<BaseMeshTopology::Triangle>& triangles = f_triangles.getValue();

    unsigned int numberOfInsideTest=0;
    helper::TriangleOctree::traceResult result;

    for (unsigned int i=0 ; i<numberOfTests.getValue() ; i++)
    {
        trianglesOctree.octreeRoot->trace(p, directions[i], result);
        if(result.tid > -1.0)
        {
            BaseMeshTopology::Triangle triangle = triangles[result.tid];
            //test if the point is inside or outside (using triangle's normal)
            Coord n = cross(vertices[triangle[1]]-vertices[triangle[0]], vertices[triangle[2]]-vertices[triangle[0]]);
            n.normalize();
            if (dot(directions[i],n) > 0.0)
            {
                numberOfInsideTest++;
            }
        }
    }

    //return numberOfInsideTest > (numberOfTests.getValue()/2 + 1);
    return numberOfInsideTest == numberOfTests.getValue();
}

template <class DataTypes>
bool RandomPointDistributionInSurface<DataTypes>::testDistance(Coord p)
{
    const VecCoord& in = f_inPoints.getValue();

    for (unsigned int i=0 ; i<in.size() ; i++)
    {
        if ((p-in[i]).norm2() < minDistanceBetweenPoints.getValue()*minDistanceBetweenPoints.getValue())
            return false;
    }

    return true;
}

template <class DataTypes>
void RandomPointDistributionInSurface<DataTypes>::doUpdate()
{
    const VecCoord& vertices = f_vertices.getValue();
    const type::vector<sofa::core::topology::BaseMeshTopology::Triangle>& triangles = f_triangles.getValue();

    if (triangles.size() <= 1 ||  vertices.size() <= 1)
    {
        msg_error() << "In input data (number of vertices of triangles is less than 1).";
        return;
    }

    VecCoord* inPoints = f_inPoints.beginWriteOnly();
    inPoints->clear();
    VecCoord* outPoints = f_outPoints.beginWriteOnly();
    outPoints->clear();


    type::vector<type::Vec3> verticesD;
    for (unsigned int i=0 ; i<vertices.size() ; i++)
        verticesD.push_back(vertices[i]);

    trianglesOctree.buildOctree(&triangles, &verticesD);

    unsigned int indexInPoints = 0;

    Vec3 minBBox, maxBBox;
    getBBox(minBBox, maxBBox);
    safeCounter=0;
    while(indexInPoints < numberOfInPoints.getValue() && safeCounter < safeLimit)
    {
        Coord p = generateRandomPoint(minBBox, maxBBox);

        if (!isInside(p))
            outPoints->push_back(p);
        else
        {
            if (!testDistance(p))
                outPoints->push_back(p);
            else
            {
                inPoints->push_back(p);
                indexInPoints++;
            }
        }
        safeCounter++;
    }

    if (safeCounter == safeLimit)
        msg_error() << "While generating point ; cancelling to break infinite loop";

    f_inPoints.endEdit();
    f_outPoints.endEdit();

}

template <class DataTypes>
void RandomPointDistributionInSurface<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels() || !isVisible.getValue())
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    
    const VecCoord& in = f_inPoints.getValue();
    const VecCoord& out = f_outPoints.getValue();
    vparams->drawTool()->disableLighting();

    std::vector<sofa::type::Vec3> vertices;

    for (unsigned int i=0 ; i<in.size() ; i++)
        vertices.push_back(in[i]);

    vparams->drawTool()->drawPoints(vertices, 5.0, sofa::type::RGBAColor::red());
    vertices.clear();

    if (drawOutputPoints.getValue())
    {
        for (unsigned int i=0 ; i<out.size() ; i++)
            vertices.push_back(out[i]);

        vparams->drawTool()->drawPoints(vertices, 5.0, sofa::type::RGBAColor::blue());
    }


}

} //namespace sofa::component::engine::generate
