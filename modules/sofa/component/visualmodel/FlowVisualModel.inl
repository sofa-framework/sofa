/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*
 * FlowVisualModel.h
 *
 *  Created on: 18 févr. 2009
 *      Author: froy
 */

#ifndef FLOWVISUALMODEL_INL_
#define FLOWVISUALMODEL_INL_

#include "FlowVisualModel.h"
#include <sofa/simulation/common/Simulation.h>
#include <sofa/helper/system/glut.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace sofa::defaulttype;
using namespace sofa::core::topology;

template <class DataTypes>
const double FlowVisualModel<DataTypes>::STREAMLINE_NUMBER_OF_POINTS_BY_TRIANGLE = 5.0;

template <class DataTypes>
FlowVisualModel<DataTypes>::FlowVisualModel()
    :m_tetraTopo(NULL), m_tetraGeo(NULL), meanEdgeLength(0.0)
    ,m_tag2D(initData(&m_tag2D, (std::string) "", "tag2D", "Set tag which defines 2D mesh"))
    ,m_tag3D(initData(&m_tag3D, (std::string) "", "tag3D", "Set tag which defines 3D mesh"))
    ,showVelocityLines(initData(&showVelocityLines, bool(true), "showVelocityLines", "Show velocities lines"))
    ,viewVelocityFactor(initData(&viewVelocityFactor, double(0.001), "viewVelocityFactor", "Set factor for velocity arrows"))
    ,velocityMin(initData(&velocityMin, double(-1.0), "velocityMin", "Set the minimum value of velocity for drawing"))
    ,velocityMax(initData(&velocityMax, double(1.0), "velocityMax", "Set the maximum value of velocity for drawing"))
    ,showStreamLines(initData(&showStreamLines, bool(true), "showStreamLines", "Set stream lines"))
    ,streamlineSeeds(initData(&streamlineSeeds, helper::vector<Coord>(), "streamlineSeeds", "Set streamlineSeeds for Stream Lines"))
    ,streamlineMaxNumberOfPoints(initData(&streamlineMaxNumberOfPoints, (unsigned int) 50 , "streamlineMaxNumberOfPoints", "Set the maximum number of points for each stream line"))
    ,streamlineDtNumberOfPointsPerTriangle(initData(&streamlineDtNumberOfPointsPerTriangle, (double) 5.0 , "streamlineDtNumberOfPointsPerTriangle", "Set the number of points for each step (equals ~ a triangle)"))
    ,showColorScale(initData(&showColorScale, bool(true), "showColorScale", "Set color scale"))
    ,showTetrahedra(initData(&showTetrahedra, bool(false), "showTetrahedra", "Show Tetrahedra"))
    ,minAlpha(initData(&minAlpha, float(0.2), "minAlpha", "Minimum alpha value for triangles"))
    ,maxAlpha(initData(&maxAlpha, float(0.8), "maxAlpha", "Maximum alpha value for triangles"))
{
    addAlias(&showTetrahedra, "showTetras");
}

template <class DataTypes>
FlowVisualModel<DataTypes>::~FlowVisualModel()
{

}

template <class DataTypes>
void FlowVisualModel<DataTypes>::init()
{
    //sofa::core::objectmodel::BaseContext* context = this->getContext();

    //locate necessary objects with tags
    core::objectmodel::TagSet::const_iterator tagIt = this->getTags().begin();

    core::objectmodel::Tag triangles(m_tag2D.getValue());
    core::objectmodel::Tag tetrahedra(m_tag3D.getValue());
    core::objectmodel::Tag geometry("geometry");
    core::objectmodel::Tag state("state");
    core::objectmodel::Tag surface("surface");

    core::objectmodel::TagSet trianglesTS(triangles);
    core::objectmodel::TagSet tetraTS(tetrahedra);
    core::objectmodel::TagSet tetraStateTS(tetrahedra);
    core::objectmodel::TagSet tetraSurfaceTS(surface);
    core::objectmodel::TagSet tetraGeometryTS(tetrahedra);
    tetraStateTS.insert(state);
    tetraGeometryTS.insert(geometry);

    tetraCenters = NULL;
    triangleGeometry = NULL;
    tetraGeometry = NULL;

    this->getContext()->get(tetraGeometry, tetraGeometryTS, core::objectmodel::BaseContext::SearchRoot);
    //TetraGeometry
    if (!tetraGeometry)
    {
        serr << "WARNING: FlowVisualModel has no binding TetraGeometry, will considerer 2D model" <<sendl;
    }
    else
    {
        this->getContext()->get(surfaceVolume, tetraSurfaceTS, core::objectmodel::BaseContext::SearchRoot);
        if (!surfaceVolume)
        {
            serr << "WARNING: FlowVisualModel has no surface" <<sendl;
        }

        this->getContext()->get(tetraCenters, tetraStateTS, core::objectmodel::BaseContext::SearchRoot);
        if (!tetraCenters)
        {
            serr << "WARNING: FlowVisualModel has no binding FluidState" <<sendl;
            return;
        }
        else
        {
            this->getContext()->get (m_tetraTopo, tetraTS,core::objectmodel::BaseContext::SearchRoot);
            if (m_tetraTopo == NULL)
            {
                serr << "WARNING: FlowVisualModel has no binding TetrahedraSetTopology" <<sendl;
                return;
            }
            else
            {
                this->getContext()->get (m_tetraGeo, tetraTS, core::objectmodel::BaseContext::SearchRoot);
                if (m_tetraGeo == NULL)
                {
                    serr << "WARNING: FlowVisualModel has no binding TetrahedraSetGeometry" <<sendl;
                    return;
                }
            }
        }
        this->getContext()->get(shader, tetraTS,core::objectmodel::BaseContext::SearchRoot);
        if(!shader)
        {
            serr << "WARNING: FlowVisualModel has no binding Shader ; no volumic rendering" <<sendl;
        }
    }

    this->getContext()->get(triangleGeometry, trianglesTS, core::objectmodel::BaseContext::SearchRoot);

    //TriangleGeometry
    if (!triangleGeometry)
    {
        serr << "WARNING: FlowVisualModel has no binding TriangleGeometry" <<sendl;
        return;
    }
    else
    {

        this->getContext()->get (m_triTopo, trianglesTS,core::objectmodel::BaseContext::SearchRoot);
        if (m_triTopo == NULL)
        {
            serr << "WARNING: FlowVisualModel has no binding TriangleSetTopology" <<sendl;
            return;
        }

        this->getContext()->get (m_triGeo, trianglesTS,core::objectmodel::BaseContext::SearchRoot);
        if (m_triGeo == NULL)
        {
            serr << "WARNING: FlowVisualModel has no binding TriangleSetGeometry" <<sendl;
            return;
        }
    }
    reinit();

}

template <class DataTypes>
void FlowVisualModel<DataTypes>::initVisual()
{
    const BaseMeshTopology::SeqEdges& edges = m_triTopo->getEdges();

    for (unsigned int i=0 ; i<edges.size() ; i++)
        meanEdgeLength += m_triGeo->computeEdgeLength(i);
    meanEdgeLength /= edges.size();

    if (tetraCenters)
    {
        unsigned int nbPoints = m_triTopo->getNbPoints();
        unsigned int nbTetrahedra = (*this->tetraCenters->getX()).size();
        (this->tetraCenters->write(core::VecDerivId::velocity()))->beginEdit()->resize(nbTetrahedra);
        (this->tetraCenters->write(core::VecDerivId::velocity()))->endEdit();
        tetraShellPerTriangleVertex.resize(nbPoints);
        isPointInTetra.resize(nbPoints);
        tetraSize.resize(nbTetrahedra);
        std::fill(tetraSize.begin(), tetraSize.end(), (float)0.0);
        std::fill(isPointInTetra.begin(), isPointInTetra.end(), true);

        //Store some data
        //** Tetra Shell of Each Triangle Vertex **//
        //Loop for each vertex of the triangle mesh
        for (unsigned int i=0 ; i<nbPoints ; i++)
        {
            std::cout << "Precomputing neighborhood information : " << (float)i/(float)nbPoints*100.0f << "%." << '\xd';
            //helper::set<BaseMeshTopology::TetraID> tetrasShell;
            Coord pTriangle = m_triGeo->getPointPosition(i);
            //Search the closest vertex of the volumetric mesh
            unsigned int indexClosestPoint = getIndexClosestPoint(*this->tetraGeometry->getX(), pTriangle);

            //get the TetraShell of the closest Point
            helper::vector<BaseMeshTopology::TetraID> closestTetraShell = m_tetraTopo->getTetrahedraAroundVertex(indexClosestPoint);

            //helper::vector<BaseMeshTopology::Tetra> tetrahedra = m_tetraTopo->getTetrahedra();
            unsigned int t, closestTetra = 0;
            bool found = false;
            for(t=0 ; t<closestTetraShell.size() && !found; t++)
            {
                found = m_tetraGeo->isPointInTetrahedron(closestTetraShell[t], pTriangle);
                if (found)
                {
                    closestTetra = closestTetraShell[t];
                }
            }


            //the point is outside all the tetra, so we look for the closest one
            if (!found)
            {
                isPointInTetra[i] = false;
                float minDist = 1e10;
                float dist = 0.0;
                for(t=0 ; t<closestTetraShell.size() ; t++)
                {
                    Coord c = m_tetraGeo->computeTetrahedronCenter(closestTetraShell[t]);
                    dist =(pTriangle - c).norm();

                    if  (dist < minDist )
                    {
                        minDist = dist;
                        closestTetra = closestTetraShell[t];
                    }
                }
            }


            Coord points[4];
            m_tetraGeo->getTetrahedronVertexCoordinates(closestTetra, points);

            if (! (tetraSize[closestTetra] > 0.0) )
            {
                Coord c = m_tetraGeo->computeTetrahedronCenter(closestTetra);
                for (unsigned int p=0; p<4 ; p++)
                {
                    float dist = (points[p] - c).norm();
                    if( dist > tetraSize[closestTetra])
                        tetraSize[closestTetra] = dist;
                }
            }

            helper::vector<unsigned int> tetrasShell;
            m_tetraGeo->getTetraInBall(closestTetra , tetraSize[closestTetra] * 5.0f, tetrasShell);

            tetraShellPerTriangleVertex[i] = tetrasShell;

        }
        std::cout << std::endl;

        // Store neighborhood for Each Tetra Vertex //
        const VecCoord& tetraGeometryPoints = *this->tetraGeometry->getX();
        unsigned int nbTetraPoints = tetraGeometryPoints.size();
        tetraShellPerTetraVertex.resize(nbTetraPoints);
        for (unsigned int i=0 ; i<nbTetraPoints ; i++)
        {
            std::cout << "Precomputing tetra neighborhood : " << (float)i/(float)nbTetraPoints*100.0f << "%." << '\xd';
            Coord pTetra = m_tetraGeo->getPointPosition(i);
            //get the TetraShell of Point
            helper::vector<BaseMeshTopology::TetraID> tetraShell = m_tetraTopo->getTetrahedraAroundVertex(i);

            tetraShellPerTetraVertex[i] = tetraShell;
        }
        std::cout << std::endl;
    }
}

template <class DataTypes>
void FlowVisualModel<DataTypes>::reinit()
{

}

template <class DataTypes>
unsigned int FlowVisualModel<DataTypes>::getIndexClosestPoint(const VecCoord &x, typename DataTypes::Coord p)
{
    unsigned int indexClosestPoint = 0;
    double dist = (p - x[0]).norm();
    for (unsigned int i=1 ; i<x.size() ; i++)
    {
        double d = (p - x[i]).norm();
        if (d < dist)
        {
            indexClosestPoint = i;
            dist = d;
        }
    }
    return indexClosestPoint;
}

template <class DataTypes>
bool FlowVisualModel<DataTypes>::isInDomainT(unsigned int index, typename DataTypes::Coord p)
{
    //test if p is in the mesh
    //1-find closest point from seed to mesh
    bool found = false;
    unsigned int tetraID = BaseMeshTopology::InvalidID;
    helper::vector<BaseMeshTopology::TetraID> tetrahedra;
    const VecCoord& centers = *this->tetraGeometry->getX();

    if (centers.size() > 0)
    {
        //if a triangle was not already found, test all the points
        if (streamLines[index].currentPrimitiveID == BaseMeshTopology::InvalidID)
        {
            unsigned int indexClosestPoint = getIndexClosestPoint(centers, p);
            //2-get its TriangleShell
            tetrahedra = m_tetraTopo->getTetrahedraAroundVertex(indexClosestPoint);
            //3-check if the seed is in one of these triangles
            streamLines[index].primitivesAroundLastPoint.clear();
            for (unsigned int i=0 ; i<tetrahedra.size() ; i++)
            {
                if (!found)
                {
                    if ( (found = m_tetraGeo->isPointInTetrahedron(tetrahedra[i], p)) )
                    {
                        tetraID = tetrahedra[i];
                    }
                }

                //fill the set of triangles
                streamLines[index].primitivesAroundLastPoint.insert(tetrahedra[i]);
            }
        }
        else
        {
            //test if the new point is still in the current triangle
            if((found = m_tetraGeo->isPointInTetrahedron(streamLines[index].currentPrimitiveID, p)) )
            {
                tetraID = streamLines[index].currentPrimitiveID;
            }
            //find the new triangle (if any) and compute the new triangles set
            else
            {
                streamLines[index].primitivesAroundLastPoint.clear();
                const BaseMeshTopology::Tetra& currentTetra = m_tetraTopo->getTetrahedron(streamLines[index].currentPrimitiveID);

                for (unsigned int i=0 ; i<3; i++)
                {
                    tetrahedra = m_tetraTopo->getTetrahedraAroundVertex(currentTetra[i]);
                    for (unsigned int i=0 ; i<tetrahedra.size() ; i++)
                    {
                        if (!found)
                        {
                            if ( (found = m_tetraGeo->isPointInTetrahedron(tetrahedra[i], p)) )
                            {
                                tetraID = tetrahedra[i];
                            }
                        }
                        streamLines[index].primitivesAroundLastPoint.insert(tetrahedra[i]);
                    }
                }
            }
        }
    }

    if (found)
        streamLines[index].currentPrimitiveID = tetraID;

    return found;
}

template <class DataTypes>
bool FlowVisualModel<DataTypes>::isInDomain(unsigned int index, typename DataTypes::Coord p)
{
    //test if p is in the mesh
    //1-find closest point from seed to mesh
    bool found = false;
    unsigned int triangleID = BaseMeshTopology::InvalidID;
    helper::vector<BaseMeshTopology::TriangleID> triangles;
    //const VecCoord& x = *this->triangleGeometry->getV();


    unsigned int indTest;

    if (triangleCenters.size() > 0)
    {
        //if a triangle was not already found, test all the points
        if (streamLines[index].currentPrimitiveID == BaseMeshTopology::InvalidID)
        {
            unsigned int indexClosestPoint = getIndexClosestPoint(*this->triangleGeometry->getX(), p);

            //2-get its TriangleShell
            triangles = m_triTopo->getTrianglesAroundVertex(indexClosestPoint);
            //3-check if the seed is in one of these triangles
            streamLines[index].primitivesAroundLastPoint.clear();
            for (unsigned int i=0 ; i<triangles.size() ; i++)
            {
                if (!found)
                {
                    if ( (found = m_triGeo->isPointInsideTriangle(triangles[i], false, p, indTest)) )
                    {
                        triangleID = triangles[i];
                    }
                }

                //fill the set of triangles
                streamLines[index].primitivesAroundLastPoint.insert(triangles[i]);
            }
        }
        else
        {
            //test if the new point is still in the current triangle
            if((found = m_triGeo->isPointInTriangle(streamLines[index].currentPrimitiveID, false, p, indTest)) )
            {
                triangleID = streamLines[index].currentPrimitiveID;
            }
            //find the new triangle (if any) and compute the new triangles set
            else
            {
                streamLines[index].primitivesAroundLastPoint.clear();
                const BaseMeshTopology::Triangle& currentTriangle = m_triTopo->getTriangle(streamLines[index].currentPrimitiveID);

                for (unsigned int i=0 ; i<3; i++)
                {
                    triangles = m_triTopo->getTrianglesAroundVertex(currentTriangle[i]);
                    for (unsigned int i=0 ; i<triangles.size() ; i++)
                    {
                        if (!found)
                        {
                            if ( (found = m_triGeo->isPointInsideTriangle(triangles[i], false, p, indTest)) )
                            {
                                triangleID = triangles[i];
                            }
                        }
                        streamLines[index].primitivesAroundLastPoint.insert(triangles[i]);
                    }
                }
            }
        }
    }

    if (found)
        streamLines[index].currentPrimitiveID = triangleID;

    return found;
}

template <class DataTypes>
typename DataTypes::Coord FlowVisualModel<DataTypes>::interpolateVelocity(unsigned int index, Coord p, bool &atEnd)
{
    const VecDeriv* velocities;
    const VecCoord* geometry;

    if (!m_tetraTopo)
    {
        velocities = &(*this->triangleGeometry->getV());
        geometry = &(triangleCenters);

        if (!isInDomain(index,p))
        {
            atEnd = true;
            return Coord();
        }
    }
    else
    {
        velocities = &(*this->tetraCenters->getV());
        geometry = &(*this->tetraCenters->getX());

        if (!isInDomainT(index,p))
        {
            atEnd = true;
            return Coord();
        }
    }

    double distCoeff=0.0;
    double sumDistCoeff=0.0;
    Coord velocitySeed;

    for(helper::set<unsigned int>::iterator it = streamLines[index].primitivesAroundLastPoint.begin() ; it != streamLines[index].primitivesAroundLastPoint.end() ; it++)
    {
        unsigned int ind = (*it);
        distCoeff = 1/(p-(*geometry)[ind]).norm2();
        velocitySeed += (*velocities)[ind]*distCoeff;
        sumDistCoeff += distCoeff;
    }
    velocitySeed /= sumDistCoeff;

//	std::cout << p << " : " << velocitySeed << std::endl;

    return velocitySeed;
}

template <class DataTypes>
void FlowVisualModel<DataTypes>::computeStreamLine(unsigned int index, unsigned int maxNbPoints, double dt)
{
    StreamLine &streamLine = streamLines[index];
    streamLines[index].positions.clear();

    Coord currentPos = streamlineSeeds.getValue()[index];
    bool finished = false;

    while(streamLine.positions.size() < maxNbPoints && !finished)
    {
        streamLine.positions.push_back(currentPos);
        //p'k  	=  	pk  	+  	(1)/(2)hv(pk)
        //pk+1 	= 	pk 	+ 	hv(p'k)

        Coord v1 =	interpolateVelocity(index, currentPos, finished);
        Coord nextPositionPrime = currentPos + v1*dt*0.5;
        Coord v2 =	interpolateVelocity(index, nextPositionPrime, finished);
        Coord nextPosition = currentPos + v2*dt;
        currentPos = nextPosition;
    }

}

template <class DataTypes>
void FlowVisualModel<DataTypes>::interpolateVelocityAtTriangleVertices()
{
    unsigned int nbPoints =  m_triTopo->getNbPoints();
    helper::vector<double> weight;
//	const VecDeriv& v2d = *this->tetraGeometry->getV();
    velocityAtTriangleVertex.resize(nbPoints);
    normAtTriangleVertex.resize(nbPoints);
    weight.resize(nbPoints);

    std::fill( weight.begin(), weight.end(), 0 );
    std::fill( velocityAtTriangleVertex.begin(), velocityAtTriangleVertex.end(), Coord() );
    std::fill( normAtTriangleVertex.begin(), normAtTriangleVertex.end(), 0.0 );

    const core::topology::BaseMeshTopology::SeqTriangles& triangles =  m_triTopo->getTriangles();
    if (!m_tetraTopo)
    {
        const VecDeriv& v2d = *this->triangleGeometry->getV();

        for(unsigned int i=0 ; i<triangles.size() ; i++)
        {
            core::topology::BaseMeshTopology::Triangle t = (triangles[i]);
            for(unsigned int j=0 ; j<3 ; j++)
            {
                velocityAtTriangleVertex[t[j]] += v2d[i];
                normAtTriangleVertex[t[j]] += v2d[i].norm();
                weight[t[j]]++;
            }
        }
        for(unsigned int i=0 ; i<nbPoints ; i++)
        {
            velocityAtTriangleVertex[i] /= weight[i];
            normAtTriangleVertex[i] /= weight[i];
        }

    }
    else
    {
        const VecDeriv& x3d = *this->tetraCenters->getX();
        const VecDeriv& v3d = *this->tetraCenters->getV();
        const VecDeriv& p2d = *this->triangleGeometry->getX();
        if (v3d.size() > 0)
        {
            //Loop for each vertex of the triangle mesh
            for (unsigned int i=0 ; i<nbPoints ; i++)
            {
                Coord pTriangle = p2d[i]; //m_triGeo->getPointPosition(i);

                //Finally, loop over all vertices of the set and compute velocities
                for (unsigned int j = 0; j<tetraShellPerTriangleVertex[i].size() ; j++)
                {
                    //compute weight according of the distance between the triangle vertex and the closest tetra vertex
                    unsigned int index = (tetraShellPerTriangleVertex[i])[j];
                    double distCoeff = 1.0/(pTriangle-x3d[index]).norm2();
                    weight[i]+= distCoeff;
                    if (!isPointInTetra[i])
                        distCoeff = distCoeff*distCoeff;

                    velocityAtTriangleVertex[i] += v3d[index] * distCoeff;
                    normAtTriangleVertex[i] += v3d[index].norm() * distCoeff;
                }

                //		if (!isPointInTetra[i])
                //			weight[i] = exp((weight[i]*weight[i]));

                //Normalize velocity per vertex
                if (weight[i] > 0.0)
                {
                    velocityAtTriangleVertex[i] /= weight[i];
                    normAtTriangleVertex[i] /= weight[i];
                }
                else
                {
                    velocityAtTriangleVertex[i] = Coord();
                    normAtTriangleVertex[i] = 0.0;
                }

            }
        }
    }
}


template <class DataTypes>
void FlowVisualModel<DataTypes>::interpolateVelocityAtTetraVertices()
{
    unsigned int nbPoints =  m_tetraTopo->getNbPoints();
    helper::vector<double> weight;

    velocityAtTetraVertex.resize(nbPoints);
    normAtTetraVertex.resize(nbPoints);
    weight.resize(nbPoints);

    std::fill( weight.begin(), weight.end(), 0 );
    std::fill( velocityAtTetraVertex.begin(), velocityAtTetraVertex.end(), Coord() );
    std::fill( normAtTetraVertex.begin(), normAtTetraVertex.end(), 0.0 );

//	const core::topology::BaseMeshTopology::SeqTriangles& triangles =  m_triTopo->getTriangles();

    const VecDeriv& tetraCentersVelocities = *this->tetraCenters->getV();
    const VecDeriv& tetraCentersPositions = *this->tetraCenters->getX();
    const VecDeriv& tetraGeometryPositions = *this->tetraGeometry->getX();

    if (tetraCentersVelocities.size() > 0)
    {
        for (unsigned int i = 0 ; i<nbPoints ; i++)
        {
            Coord pTetra = tetraGeometryPositions[i];

            for (unsigned int j = 0; j<tetraShellPerTetraVertex[i].size() ; j++)
            {
                //compute weight according of the distance between the tetra vertex and the tetra centers
                unsigned int index = (tetraShellPerTetraVertex[i])[j];
                double distCoeff = 1.0/(pTetra-tetraCentersPositions[index]).norm2();
                weight[i]+= distCoeff;

                velocityAtTetraVertex[i] += tetraCentersVelocities[index] * distCoeff;
                normAtTetraVertex[i] += tetraCentersVelocities[index].norm() * distCoeff;
            }
            //Normalize velocity per vertex
            if (weight[i] > 0.0)
            {
                velocityAtTetraVertex[i] /= weight[i];
                normAtTetraVertex[i] /= weight[i];
            }
            else
            {
                velocityAtTetraVertex[i] = Coord();
                normAtTetraVertex[i] = 0.0;
            }
        }
    }
}

template <class DataTypes>
void FlowVisualModel<DataTypes>::drawTetra()
{
    // const VecCoord& tetrasX = *this->tetraGeometry->getX();
//	const VecDeriv& v3d = *this->tetraGeometry->getV();
    //glEnable (GL_BLEND);

    //glBlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    Vec<4, float> colorsTemp[4];
    int indColors[4];
    const VecCoord& x = *this->tetraGeometry->getX();

    if(shader)
        shader->start();
    //Tetra
    if(m_tetraTopo)
    {
        interpolateVelocityAtTetraVertices();

        //draw tetrahedra
        const core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra=  m_tetraTopo->getTetrahedra();
        /*
        		for (unsigned int i=0 ; i<tetrahedra.size() ; i++)
        		{

        			unsigned int a = tetrahedra[i][0];
        			unsigned int b = tetrahedra[i][1];
        			unsigned int c = tetrahedra[i][2];
        			unsigned int d = tetrahedra[i][3];
        			Coord center = (x[a]+x[b]+x[c]+x[d])*0.125;
        //			Coord pa = (x[a]+center)*(Real)0.666667;
        //			Coord pb = (x[b]+center)*(Real)0.666667;
        //			Coord pc = (x[c]+center)*(Real)0.666667;
        //			Coord pd = (x[d]+center)*(Real)0.666667;

        			Coord pa = (x[a]);
        			Coord pb = (x[b]);
        			Coord pc = (x[c]);
        			Coord pd = (x[d]);

        			points[0].push_back(pa);
        			points[0].push_back(pb);
        			points[0].push_back(pc);

        			points[1].push_back(pb);
        			points[1].push_back(pc);
        			points[1].push_back(pd);

        			points[2].push_back(pc);
        			points[2].push_back(pd);
        			points[2].push_back(pa);

        			points[3].push_back(pd);
        			points[3].push_back(pa);
        			points[3].push_back(pb);

        			if((maximumVelocityAtTriangleVertex-minimumVelocityAtTriangleVertex) > 0.0)
        			{
        				for (unsigned int j=0;j<4;j++)
        				{
        					indColors[j] = (int)(64* ((normAtTetraVertex[tetrahedra[i][j]]-minimumVelocityAtTriangleVertex)/(maximumVelocityAtTriangleVertex-minimumVelocityAtTriangleVertex)));
        					if (indColors[j] < 0) indColors[j] = 0;
        					if (indColors[j] >= 64) indColors[j] = 63;
        				}
        			}
        			else indColors[0] = indColors[0] = indColors[2] = indColors[3] = 0;

        			for (unsigned int j=0;j<4;j++)
        			{
        				for (unsigned int rgb=0;rgb<3;rgb++)
        				{
        					colorsTemp[j][rgb] = ColorMap[indColors[j]][rgb];
        				}
        				colorsTemp[j][3] =  minAlpha.getValue() +  (maxAlpha.getValue() - minAlpha.getValue())*(indColors[j]/63.0);
        			}

        			colors[0].push_back(colorsTemp[0]);
        			colors[0].push_back(colorsTemp[1]);
        			colors[0].push_back(colorsTemp[2]);

        			colors[1].push_back(colorsTemp[1]);
        			colors[1].push_back(colorsTemp[2]);
        			colors[1].push_back(colorsTemp[3]);

        			colors[2].push_back(colorsTemp[2]);
        			colors[2].push_back(colorsTemp[3]);
        			colors[2].push_back(colorsTemp[0]);

        			colors[3].push_back(colorsTemp[3]);
        			colors[3].push_back(colorsTemp[0]);
        			colors[3].push_back(colorsTemp[1]);
        		}
        //		simulation::getSimulation()->DrawUtility().drawTriangles(points[0], colors[0]);
        //		simulation::getSimulation()->DrawUtility().drawTriangles(points[1], colors[1]);
        //		simulation::getSimulation()->DrawUtility().drawTriangles(points[2], colors[2]);
        //		simulation::getSimulation()->DrawUtility().drawTriangles(points[3], colors[3]);
        */
        core::topology::BaseMeshTopology::SeqTetrahedra::const_iterator it;
        Coord v;
        unsigned int i;
        for(it = tetrahedra.begin(), i=0; it != tetrahedra.end() ; it++, i++)
        {

#ifdef GL_LINES_ADJACENCY_EXT
            glBegin(GL_LINES_ADJACENCY_EXT);
#else
            glBegin(GL_POINTS);
#endif

            for (unsigned int j=0 ; j< 4 ; j++)
            {
                if((maximumVelocityAtTriangleVertex-minimumVelocityAtTriangleVertex) > 0.0)
                {
                    indColors[j] = (int)(64* ((normAtTetraVertex[tetrahedra[i][j]]-minimumVelocityAtTriangleVertex)/(maximumVelocityAtTriangleVertex-minimumVelocityAtTriangleVertex)));
                    if (indColors[j] < 0) indColors[j] = 0;
                    if (indColors[j] >= 64) indColors[j] = 63;
                }
                else indColors[j] = 0;

                for (unsigned int rgb=0; rgb<3; rgb++)
                {
                    colorsTemp[j][rgb] = ColorMap[indColors[j]][rgb];
                }
                colorsTemp[j][3] =  minAlpha.getValue() +  (maxAlpha.getValue() - minAlpha.getValue())*(indColors[j]/63.0);

                v = x[(*it)[j]];
                glColor4fv(colorsTemp[j].ptr());
                glVertex3f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2]);

            }

            glEnd();

        }
    }
    if(shader)
        shader->stop();

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);

}

template <class DataTypes>
void FlowVisualModel<DataTypes>::draw()
{
    if (!getContext()->getShowVisualModels()) return;

    glDepthMask(GL_TRUE);
    glDisable(GL_LIGHTING);
    const core::topology::BaseMeshTopology::SeqTriangles triangles =  m_triTopo->getTriangles();
    minimumVelocityAtTriangleVertex =  velocityMin.getValue();
    maximumVelocityAtTriangleVertex =  velocityMax.getValue();
    interpolateVelocityAtTriangleVertices();
    /*
        double maximumVelocityAtTriangleVertex2 = velocityAtTriangleVertex[0].norm2();
    	minimumVelocityAtTriangleVertex = velocityAtTriangleVertex[0].norm();

    	//search maximumVelocityAtTriangleVertex
    	for (unsigned int i=1 ; i<velocityAtTriangleVertex.size() ; i++)
    	{
    		if (velocityAtTriangleVertex[i].norm2() > maximumVelocityAtTriangleVertex2)
    			maximumVelocityAtTriangleVertex2=velocityAtTriangleVertex[i].norm2();
    	}

    	maximumVelocityAtTriangleVertex = (velocityMax.getValue());// < sqrt(maximumVelocityAtTriangleVertex2)) ? sqrt(maximumVelocityAtTriangleVertex2) : velocityMax.getValue();

    	if (velocityMin.getValue() < 0.0)
    	{
    		//search minimumVelocityAtTriangleVertex
    		for (unsigned int i=1 ; i<velocityAtTriangleVertex.size() ; i++)
    		{
    			if (velocityAtTriangleVertex[i].norm() < minimumVelocityAtTriangleVertex)
    				minimumVelocityAtTriangleVertex=velocityAtTriangleVertex[i].norm();
    		}
    		if (minimumVelocityAtTriangleVertex < 0.0) minimumVelocityAtTriangleVertex = 0.0;
    	}
    	else minimumVelocityAtTriangleVertex = velocityMin.getValue();
    */
    triangleCenters.clear();
    triangleCenters.resize(0);
    for(unsigned int i=0 ; i<triangles.size() ; i++)
    {
        const core::topology::BaseMeshTopology::Triangle t = (triangles[i]);

        Coord p0 = m_triGeo->getPointPosition(t[0]);
        Coord p1 = m_triGeo->getPointPosition(t[1]);
        Coord p2 = m_triGeo->getPointPosition(t[2]);

        //compute barycenter of each triangle
        Coord pb;
        pb[0] = (p0[0] + p1[0] + p2[0])/3;
        pb[1] = (p0[1] + p1[1] + p2[1])/3;
        pb[2] = (p0[2] + p1[2] + p2[2])/3;
        triangleCenters.push_back(pb);
    }


    const VecCoord& x2D = *this->triangleGeometry->getX();
    const VecCoord& v2D = *this->triangleGeometry->getV();

    //Show Velocity
    if (showVelocityLines.getValue())
    {
        glBegin(GL_LINES);
        glColor3f(0.5,0.5,0.5);
        if (!m_tetraTopo)
        {
            for(unsigned int i=0 ; i<triangleCenters.size() ; i++)
            {
                if (maximumVelocityAtTriangleVertex > 0.0)
                {
                    Coord p0 = triangleCenters[i];
                    Coord p1 = triangleCenters[i] + v2D[i]/maximumVelocityAtTriangleVertex*viewVelocityFactor.getValue();

                    glVertex3f((GLfloat)p0[0], (GLfloat)p0[1], (GLfloat)p0[2]);
                    glVertex3f((GLfloat)p1[0], (GLfloat)p1[1], (GLfloat)p1[2]);
                }
            }
        }
        else
        {
            for(unsigned int i=0 ; i<x2D.size() ; i++)
            {
                if (maximumVelocityAtTriangleVertex > 0.0)
                {
                    Coord p0 = x2D[i];
                    Coord p1 = x2D[i] + velocityAtTriangleVertex[i]/maximumVelocityAtTriangleVertex*viewVelocityFactor.getValue();

                    glVertex3f((GLfloat)p0[0], (GLfloat)p0[1], (GLfloat)p0[2]);
                    glVertex3f((GLfloat)p1[0], (GLfloat)p1[1], (GLfloat)p1[2]);
                }
            }
        }

        glEnd();
    }

    //Draw StreamLines
    if (showStreamLines.getValue())
    {
        helper::vector<Coord> seeds = streamlineSeeds.getValue();
        unsigned int seedsSize = streamlineSeeds.getValue().size();
        streamLines.clear();
        streamLines.resize(seedsSize);

        for (unsigned int i=0 ; i<seedsSize ; i++)
        {
            double dtStreamLine = (meanEdgeLength / maximumVelocityAtTriangleVertex) * (1.0/streamlineDtNumberOfPointsPerTriangle.getValue());
            streamLines[i].currentPrimitiveID = BaseMeshTopology::InvalidID;
            computeStreamLine(i,streamlineMaxNumberOfPoints.getValue(), dtStreamLine) ;
            glPointSize(10.0);
            glColor3f(1.0,1.0,1.0);
            glBegin(GL_POINTS);
            glVertex3f(seeds[i][0], seeds[i][1], seeds[i][2]);
            glEnd();
            glLineWidth(3);
            glBegin(GL_LINE_STRIP);
            for(unsigned int j=0 ; j<streamLines[i].positions.size() ; j++)
            {
                glVertex3f((GLfloat) streamLines[i].positions[j][0], (GLfloat) streamLines[i].positions[j][1], (GLfloat) streamLines[i].positions[j][2]);
            }
            glEnd();
            glLineWidth(1);
        }
    }

}

template <class DataTypes>
void FlowVisualModel<DataTypes>::drawTransparent(const core::visual::VisualParams*)
{
    if (!getContext()->getShowVisualModels()) return;
    glDepthMask(GL_FALSE);
    glDisable(GL_LIGHTING);
    const core::topology::BaseMeshTopology::SeqTriangles triangles =  m_triTopo->getTriangles();
    glEnable (GL_BLEND);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //VecCoord& x = *this->triangleGeometry->getX();


    {

        int indColor0, indColor1, indColor2;
        //VecCoord& x = *this->fstate->getX();


        if (getContext()->getShowWireFrame())
        {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        }

        //Show colored triangles

        glBegin(GL_TRIANGLES);
        for(unsigned int i=0 ; i<triangles.size() ; i++)
        {
            const core::topology::BaseMeshTopology::Triangle t = (triangles[i]);

            Coord p0 = m_triGeo->getPointPosition(t[0]);
            Coord p1 = m_triGeo->getPointPosition(t[1]);
            Coord p2 = m_triGeo->getPointPosition(t[2]);
            if((maximumVelocityAtTriangleVertex-minimumVelocityAtTriangleVertex) > 0.0)
            {
                indColor0 = (int)(64* ((normAtTriangleVertex[t[0]]-minimumVelocityAtTriangleVertex)/(maximumVelocityAtTriangleVertex-minimumVelocityAtTriangleVertex)));
                indColor1 = (int)(64* ((normAtTriangleVertex[t[1]]-minimumVelocityAtTriangleVertex)/(maximumVelocityAtTriangleVertex-minimumVelocityAtTriangleVertex)));
                indColor2 = (int)(64* ((normAtTriangleVertex[t[2]]-minimumVelocityAtTriangleVertex)/(maximumVelocityAtTriangleVertex-minimumVelocityAtTriangleVertex)));
                if (indColor0 < 0) indColor0 = 0;
                if (indColor1 < 0) indColor1 = 0;
                if (indColor2 < 0) indColor2 = 0;
                if (indColor0 >= 64) indColor0 = 63;
                if (indColor1 >= 64) indColor1 = 63;
                if (indColor2 >= 64) indColor2 = 63;

            }
            else indColor0 = indColor1 = indColor2 = 0;

            float alpha0 = minAlpha.getValue() +  (maxAlpha.getValue() - minAlpha.getValue())*(indColor0/63.0);
            float alpha1 = minAlpha.getValue() +  (maxAlpha.getValue() - minAlpha.getValue())*(indColor1/63.0);
            float alpha2 = minAlpha.getValue() +  (maxAlpha.getValue() - minAlpha.getValue())*(indColor2/63.0);

            glColor4f(ColorMap[indColor0][0], ColorMap[indColor0][1], ColorMap[indColor0][2], alpha0);
            glVertex3f((GLfloat)p0[0], (GLfloat)p0[1], (GLfloat)p0[2]);
            glColor4f(ColorMap[indColor1][0], ColorMap[indColor1][1], ColorMap[indColor1][2], alpha1);
            glVertex3f((GLfloat)p1[0], (GLfloat)p1[1], (GLfloat)p1[2]);
            glColor4f(ColorMap[indColor2][0], ColorMap[indColor2][1], ColorMap[indColor2][2], alpha2);
            glVertex3f((GLfloat)p2[0], (GLfloat)p2[1], (GLfloat)p2[2]);
        }
        glEnd();


        if (showTetrahedra.getValue())
            drawTetra();

        if (getContext()->getShowWireFrame())
        {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        //Draw color scale
        if (showColorScale.getValue())
        {
            double xMargin = 50.0;
            double yMargin = 20.0;
            GLint currentViewport[4];
            glGetIntegerv(GL_VIEWPORT, currentViewport);
            Vec2d scaleSize = Vec2d(20, 200);
            Vec2d scalePosition = Vec2d(currentViewport[2] - xMargin - scaleSize[0], currentViewport[3] - yMargin - scaleSize[1]);

            glMatrixMode(GL_PROJECTION);
            glPushMatrix();
            glLoadIdentity();
            gluOrtho2D(0,currentViewport[2],0,currentViewport[3]);
            glDisable(GL_DEPTH_TEST);
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            glLoadIdentity();

            double step = scaleSize[1]/64;
            glBegin(GL_QUADS);
            for (unsigned int i=0 ; i<63 ; i++)
            {
                glColor3fv(ColorMap[i].ptr());
                glVertex2d(scalePosition[0], scalePosition[1] + i*step);
                glVertex2d(scalePosition[0] + scaleSize[0], scalePosition[1] + i*step);
                glColor3fv(ColorMap[i+1].ptr());
                glVertex2d(scalePosition[0] + scaleSize[0], scalePosition[1] + (i+1)*step);
                glVertex2d(scalePosition[0], scalePosition[1] + (i+1)*step);
            }
            glEnd();

            //Draw color scale values
            unsigned int NUMBER_OF_VALUES = 10;
            glColor3f(0.5,0.5, 0.5);
            step = scaleSize[1]/(NUMBER_OF_VALUES - 1);
            double stepValue = (maximumVelocityAtTriangleVertex-minimumVelocityAtTriangleVertex)/(NUMBER_OF_VALUES - 1);
            double textScaleSize = 0.085;
            double xMarginWithScale = 5;

            for (unsigned int i=0 ; i<NUMBER_OF_VALUES  ; i++)
            {
                glPushMatrix();
                glTranslatef(scalePosition[0] + scaleSize[0] + xMarginWithScale, scalePosition[1] + i*step, 0);
                glScalef(textScaleSize, textScaleSize, textScaleSize);
                double intpart;
                double decpart = modf((minimumVelocityAtTriangleVertex + i*stepValue), &intpart);
                std::ostringstream oss;
                oss << intpart << "." << (ceil(decpart*100));
                std::string tmp = oss.str();


                const char* s = tmp.c_str();
                while(*s)
                {
                    glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
                    s++;
                }
                glPopMatrix();
            }
            glEnable(GL_DEPTH_TEST);
            glPopMatrix();
            glMatrixMode(GL_PROJECTION);
            glPopMatrix();
            glMatrixMode(GL_MODELVIEW);
        }
    }
    glDisable (GL_BLEND);
    glDepthMask(GL_TRUE);
}



}

}

}

#endif /* FLOWVISUALMODEL_INL_ */
