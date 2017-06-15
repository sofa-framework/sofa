/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef PLUGINS_PIM_COMPUTEMESHINTERSECTION_INL
#define PLUGINS_PIM_COMPUTEMESHINTERSECTION_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include "ComputeMeshIntersection.h"
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/defaulttype/Mat.h>

namespace plugins
{

namespace pim
{

using namespace sofa::helper;
using namespace sofa::defaulttype;
using namespace sofa::core::objectmodel;

template <class DataTypes>
ComputeMeshIntersection<DataTypes>::ComputeMeshIntersection():
    d_muscleLayerVertex(initData(&d_muscleLayerVertex, "muscleLayerVertex", "Muscle Layer vertex position") )
    , d_fatLayerVertex(initData(&d_fatLayerVertex, "fatLayerVertex", "Fat Layer vertex position") )
    , d_intersectionVertex(initData(&d_intersectionVertex, "intersectionVertex", "Intersection vertex position") )
    , d_muscleLayerTriangles(initData(&d_muscleLayerTriangles, "muscleLayerTriangles", "Muscle Layer triangles") )
    , d_fatLayerTriangles(initData(&d_fatLayerTriangles, "fatLayerTriangles", "Fat Layer triangles") )
    , d_intersectionTriangles(initData(&d_intersectionTriangles, "intersectionTriangles", "Intersection triangles") )
    , d_intersectionQuads(initData(&d_intersectionQuads, "intersectionQuads", "Intersection Quads") )
    , d_print_log(initData(&d_print_log, false,"print_log", "Print log") )
    , d_epsilon(initData(&d_epsilon, 0.0, "epsilon", "min dsitance betbeen the fat and the muscle") )
    , d_index(initData(&d_index, "index", "") )
{
}

template <class DataTypes>
void ComputeMeshIntersection<DataTypes>::init()
{
    if (d_print_log.getValue())
        std::cout << "init" << std::endl;

    addInput(&d_muscleLayerVertex);
    addInput(&d_fatLayerVertex);
    addInput(&d_muscleLayerTriangles);
    addInput(&d_fatLayerTriangles);
    addOutput(&d_intersectionVertex);
    addOutput(&d_intersectionTriangles);
    setDirtyValue();
}

///////////////////VERSION C /////////////////////////////////////////
template <class DataTypes>
void ComputeMeshIntersection<DataTypes>::computeIntersectionLayerVertex()
{
    if (d_print_log.getValue())
        std::cout << "computeIntersectionLayerVertex" << std::endl;

    unsigned int vertexIndex = 0;
    const VecCoord& XF = d_fatLayerVertex.getValue();
    const VecCoord& XM = d_muscleLayerVertex.getValue();
    vector<unsigned int>& index = *d_index.beginEdit();

    for (unsigned int i=0; i<XF.size(); ++i)
    {

//     if (d_print_log.getValue())
//         std::cout << (XF[i]-XM[i]).norm() << std::endl;

        if ((XF[i]-XM[i]).norm() > d_epsilon.getValue())
        {

//     if (d_print_log.getValue())
//         std::cout << "(XF[i]-XM[i]).norm() > d_epsilon.getValue()" << std::endl;

            topology.addPoint(XM[i].x(), XM[i].y(), XM[i].z());
            topology.addPoint(XF[i].x(), XF[i].y(), XF[i].z());
            index.push_back(i);
            intersectionIndices.insert(std::make_pair(i, vertexIndex*2));
            vertexIndex++;
        }
    }
    d_index.endEdit();
}

template <class DataTypes>
bool ComputeMeshIntersection<DataTypes>::isIntersectionLayerTriangle(const Triangle& ft, Triangle& fi)
{
    std::map<unsigned int, unsigned int>::const_iterator it0, it1, it2;

    it0 = intersectionIndices.find(ft[0]);
    it1 = intersectionIndices.find(ft[1]);
    it2 = intersectionIndices.find(ft[2]);

    if (it0 != intersectionIndices.end() && it1 != intersectionIndices.end() && it2 != intersectionIndices.end())
    {
        fi[0] = it0->second;
        fi[1] = it1->second;
        fi[2] = it2->second;
        return true;
    }
    return false;
}

template <class DataTypes>
void ComputeMeshIntersection<DataTypes>::computeIntersectionLayerTriangles()
{
    if (d_print_log.getValue())
        std::cout << "computeIntersectionLayerTriangles" << std::endl;

    const VecTriangles& fatLayerTriangles = d_fatLayerTriangles.getValue();
    for (unsigned int i=0; i<fatLayerTriangles.size(); ++i)
    {
        const Triangle ft = fatLayerTriangles[i];
        Triangle fi = Triangle(0,0,0);
        if (isIntersectionLayerTriangle(ft, fi))
        {
            topology.addTriangle(fi[0], fi[1], fi[2]);
            topology.addTriangle(fi[0]+1, fi[1]+1, fi[2]+1);
        }
    }
}

template <class DataTypes>
void ComputeMeshIntersection<DataTypes>::closeMesh()
{

    if (d_print_log.getValue())
        std::cout << "closeMesh" << std::endl;

    int count = 0;

    const vector<Edge>& eds = topology.getEdges();
    for (unsigned int i=0; i<eds.size(); i++)
    {
        if (topology.getTrianglesAroundEdge(i).size() < 2 && (fmod(eds[i][0], 2) == 0 ))
        {

//std::cout << eds[i][0] << " " << eds[i][0]+1 << " " << eds[i][1]+1 << " " << eds[i][1] << std::endl;

            topology.addQuad(eds[i][0], eds[i][0]+1, eds[i][1]+1, eds[i][1]);
            count++;
            /*            if (count > 10)
                            break;*/
        }
    }
}

template <class DataTypes>
void ComputeMeshIntersection<DataTypes>::update()
{
    if (d_print_log.getValue())
        std::cout << "update" << std::endl;

    cleanDirty();

    VecCoord& intersectionVertex = *(d_intersectionVertex.beginEdit());
    VecTriangles& intersectionTriangles = *(d_intersectionTriangles.beginEdit());
    vector<Quad>& intersectionQuads = *(d_intersectionQuads.beginEdit());

    computeIntersectionLayerVertex();

    computeIntersectionLayerTriangles();

    closeMesh();

    intersectionVertex.resize(topology.getNbPoints());
    for (int i=0; i<topology.getNbPoints(); ++i)
    {
        intersectionVertex[i][0] = topology.getPX(i);
        intersectionVertex[i][1] = topology.getPY(i);
        intersectionVertex[i][2] = topology.getPZ(i);
    }
    intersectionTriangles = topology.getTriangles();
    intersectionQuads = topology.getQuads();

    d_intersectionVertex.endEdit();
    d_intersectionTriangles.endEdit();
    d_intersectionQuads.endEdit();
}

template <class DataTypes>
void ComputeMeshIntersection<DataTypes>::draw()
{
    glPointSize(5);
    glColor4f(1,0,0,1);
    glBegin(GL_POINTS);
    const vector<Edge>& eds = topology.getEdges();
    for (unsigned int i=0; i<eds.size(); ++i)
    {
        if (topology.getTrianglesAroundEdge(i).size() < 2 )
        {
            glVertex3d(topology.getPX(eds[i][0]),topology.getPY(eds[i][0]),topology.getPZ(eds[i][0]));
            glVertex3d(topology.getPX(eds[i][1]),topology.getPY(eds[i][1]),topology.getPZ(eds[i][1]));
        }
    }
    glEnd();
}

} // namespace pim

} // namespace plugins

#endif
