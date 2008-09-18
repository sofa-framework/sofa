/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPING_INL
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPING_INL

#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>
#include <sofa/helper/gl/template.h>
#include <algorithm>
#include <iostream>

#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/component/topology/RegularGridTopology.h>
#include <sofa/component/topology/SparseGridTopology.h>

#include <sofa/component/topology/EdgeSetTopologyContainer.h>
#include <sofa/component/topology/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/QuadSetTopologyContainer.h>
#include <sofa/component/topology/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/HexahedronSetTopologyContainer.h>

#include <sofa/component/topology/EdgeSetGeometryAlgorithms.h>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.h>
#include <sofa/component/topology/QuadSetGeometryAlgorithms.h>
#include <sofa/component/topology/TetrahedronSetGeometryAlgorithms.h>
#include <sofa/component/topology/HexahedronSetGeometryAlgorithms.h>

#include <sofa/component/topology/PointData.h>
#include <sofa/component/topology/PointData.inl>

#include <sofa/component/topology/HexahedronData.h>
#include <sofa/component/topology/HexahedronData.inl>

using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;

template <class BasicMapping>
BarycentricMapping<BasicMapping>::BarycentricMapping(In* from, Out* to, BaseMeshTopology * topology )
    : Inherit(from, to), mapper(NULL)
{
    createMapperFromTopology(topology);
}


template <class In, class Out>
void BarycentricMapperRegularGridTopology<In,Out>::init(const typename Out::VecCoord& out, const typename In::VecCoord& in)
{
    if (this->map1d.size()+this->map2d.size()+this->map3d.size() != 0) return;
    int outside = 0;
    clear(out.size());

#ifdef SOFA_NEW_HEXA
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqHexas& cubes = topology->getHexas();
#else
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqCubes& cubes = topology->getCubes();
#endif

    if (cubes.size())
    {
        for (unsigned int i=0; i<out.size(); i++)
        {
            Vector3 coefs;
            int cube = topology->findCube(Vector3(out[i]), coefs[0], coefs[1], coefs[2]);
            if (cube==-1)
            {
                ++outside;
                cube = topology->findNearestCube(Vector3(out[i]), coefs[0], coefs[1], coefs[2]);
            }

            this->addPointInCube(cube, coefs.ptr());
        }
    }
    else
    {
        BarycentricMapperMeshTopology<In,Out>::init(out,in);
    }

}

template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::init(const typename Out::VecCoord& out, const typename In::VecCoord& in)
{
    if (this->map1d.size()+this->map2d.size()+this->map3d.size() != 0) return;
    int outside = 0;
    clear(out.size());


#ifdef SOFA_NEW_HEXA
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqHexas& cubes = topology->getHexas();
#else
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqCubes& cubes = topology->getCubes();
#endif

    if (cubes.size())
    {
        for (unsigned int i=0; i<out.size(); i++)
        {
            Vector3 coefs;
            int cube = topology->findCube(Vector3(out[i]), coefs[0], coefs[1], coefs[2]);
            if (cube==-1)
            {
                ++outside;
                cube = topology->findNearestCube(Vector3(out[i]), coefs[0], coefs[1], coefs[2]);
            }
            Vector3 baryCoords = coefs;
            this->addPointInCube(cube, baryCoords.ptr());
        }
    }
    else
    {
        BarycentricMapperMeshTopology<In,Out>::init(out,in);
    }
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::clear1d(int reserve)
{
    map1d.clear(); if (reserve>0) map1d.reserve(reserve);
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::clear2d(int reserve)
{
    map2d.clear(); if (reserve>0) map2d.reserve(reserve);
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::clear3d(int reserve)
{
    map3d.clear(); if (reserve>0) map3d.reserve(reserve);
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::clear(int reserve)
{
    map1d.clear(); if (reserve>0) map1d.reserve(reserve);
    map2d.clear(); if (reserve>0) map2d.reserve(reserve);
    map3d.clear(); if (reserve>0) map3d.reserve(reserve);
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::addPointInLine(int lineIndex, const SReal* baryCoords)
{
    map1d.resize(map1d.size()+1);
    MappingData1D& data = *map1d.rbegin();
    data.in_index = lineIndex;
    data.baryCoords[0] = (Real)baryCoords[0];
    return map1d.size()-1;
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::addPointInTriangle(int triangleIndex, const SReal* baryCoords)
{
    map2d.resize(map2d.size()+1);
    MappingData2D& data = *map2d.rbegin();
    data.in_index = triangleIndex;
    data.baryCoords[0] = (Real)baryCoords[0];
    data.baryCoords[1] = (Real)baryCoords[1];
    return map2d.size()-1;
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::addPointInQuad(int quadIndex, const SReal* baryCoords)
{
    map2d.resize(map2d.size()+1);
    MappingData2D& data = *map2d.rbegin();
    data.in_index = quadIndex + this->topology->getNbTriangles();
    data.baryCoords[0] = (Real)baryCoords[0];
    data.baryCoords[1] = (Real)baryCoords[1];
    return map2d.size()-1;
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::addPointInTetra(int tetraIndex, const SReal* baryCoords)
{
    map3d.resize(map3d.size()+1);
    MappingData3D& data = *map3d.rbegin();
    data.in_index = tetraIndex;
    data.baryCoords[0] = (Real)baryCoords[0];
    data.baryCoords[1] = (Real)baryCoords[1];
    data.baryCoords[2] = (Real)baryCoords[2];
    return map3d.size()-1;
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::addPointInCube(int cubeIndex, const SReal* baryCoords)
{
    map3d.resize(map3d.size()+1);
    MappingData3D& data = *map3d.rbegin();
    data.in_index = cubeIndex + this->topology->getNbTetras();
    data.baryCoords[0] = (Real)baryCoords[0];
    data.baryCoords[1] = (Real)baryCoords[1];
    data.baryCoords[2] = (Real)baryCoords[2];
    return map3d.size()-1;
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::createPointInLine(const typename Out::Coord& p, int lineIndex, const typename In::VecCoord* points)
{
    SReal baryCoords[1];
    const sofa::core::componentmodel::topology::BaseMeshTopology::Line& elem = this->topology->getLine(lineIndex);
    const typename In::Coord p0 = (*points)[elem[0]];
    const typename In::Coord pA = (*points)[elem[1]] - p0;
    typename In::Coord pos = p - p0;
    baryCoords[0] = ((pos*pA)/pA.norm2());
    return this->addPointInLine(lineIndex, baryCoords);
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::createPointInTriangle(const typename Out::Coord& p, int triangleIndex, const typename In::VecCoord* points)
{
    SReal baryCoords[2];
    const sofa::core::componentmodel::topology::BaseMeshTopology::Triangle& elem = this->topology->getTriangle(triangleIndex);
    const typename In::Coord p0 = (*points)[elem[0]];
    const typename In::Coord pA = (*points)[elem[1]] - p0;
    const typename In::Coord pB = (*points)[elem[2]] - p0;
    typename In::Coord pos = p - p0;
    // First project to plane
    typename In::Coord normal = cross(pA, pB);
    Real norm2 = normal.norm2();
    pos -= normal*((pos*normal)/norm2);
    baryCoords[0] = (Real)sqrt(cross(pB, pos).norm2() / norm2);
    baryCoords[1] = (Real)sqrt(cross(pA, pos).norm2() / norm2);
    return this->addPointInTriangle(triangleIndex, baryCoords);
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::createPointInQuad(const typename Out::Coord& p, int quadIndex, const typename In::VecCoord* points)
{
    SReal baryCoords[2];
    const sofa::core::componentmodel::topology::BaseMeshTopology::Quad& elem = this->topology->getQuad(quadIndex);
    const typename In::Coord p0 = (*points)[elem[0]];
    const typename In::Coord pA = (*points)[elem[1]] - p0;
    const typename In::Coord pB = (*points)[elem[3]] - p0;
    typename In::Coord pos = p - p0;
    Mat<3,3,typename In::Real> m,mt,base;
    m[0] = pA;
    m[1] = pB;
    m[2] = cross(pA, pB);
    mt.transpose(m);
    base.invert(mt);
    const typename In::Coord base0 = base[0];
    const typename In::Coord base1 = base[1];
    baryCoords[0] = base0 * pos;
    baryCoords[1] = base1 * pos;
    return this->addPointInQuad(quadIndex, baryCoords);
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::init(const typename Out::VecCoord& out, const typename In::VecCoord& in)
{
    int outside = 0;

    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqTetras& tetras = this->topology->getTetras();
#ifdef SOFA_NEW_HEXA
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqHexas& cubes = this->topology->getHexas();
#else
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqCubes& cubes = this->topology->getCubes();
#endif
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqTriangles& triangles = this->topology->getTriangles();
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqQuads& quads = this->topology->getQuads();
    sofa::helper::vector<Matrix3> bases;
    sofa::helper::vector<Vector3> centers;
    if (tetras.empty() && cubes.empty())
    {
        if (triangles.empty() && quads.empty())
        {
            //no 3D elements, nor 2D elements -> map on 1D elements

            const sofa::core::componentmodel::topology::BaseMeshTopology::SeqEdges& edges = this->topology->getEdges();
            if (edges.empty()) return;

            clear1d(out.size());

            sofa::helper::vector< SReal >   lengthEdges;
            sofa::helper::vector< Vector3 > unitaryVectors;

            unsigned int e;
            for (e=0; e<edges.size(); e++)
            {
                lengthEdges.push_back((in[edges[e][1]]-in[edges[e][0]]).norm());

                Vector3 V12 = (in[edges[e][1]]-in[edges[e][0]]); V12.normalize();
                unitaryVectors.push_back(V12);
            }

            for (unsigned int i=0; i<out.size(); i++)
            {
                SReal coef=0;
                for (e=0; e<edges.size(); e++)
                {
                    SReal lengthEdge = lengthEdges[e];
                    Vector3 V12 =unitaryVectors[e];

                    coef = 1-(V12)*(out[i]-in[edges[e][0]])/lengthEdge;
                    if (coef >= 0 && coef <= 1) {addPointInLine(e,&coef);  break; }
                }
                //If no good coefficient has been found, we add to the last element
                if (e == edges.size()) addPointInLine(edges.size()-1,&coef);

            }
        }
        else
        {
            // no 3D elements -> map on 2D elements
            clear2d(out.size()); // reserve space for 2D mapping
            int c0 = triangles.size();
            bases.resize(triangles.size()+quads.size());
            centers.resize(triangles.size()+quads.size());
            for (unsigned int t = 0; t < triangles.size(); t++)
            {
                Mat3x3d m,mt;
                m[0] = in[triangles[t][1]]-in[triangles[t][0]];
                m[1] = in[triangles[t][2]]-in[triangles[t][0]];
                m[2] = cross(m[0],m[1]);
                mt.transpose(m);
                bases[t].invert(mt);
                centers[t] = (in[triangles[t][0]]+in[triangles[t][1]]+in[triangles[t][2]])/3;
            }
            for (unsigned int c = 0; c < quads.size(); c++)
            {
                Mat3x3d m,mt;
                m[0] = in[quads[c][1]]-in[quads[c][0]];
                m[1] = in[quads[c][3]]-in[quads[c][0]];
                m[2] = cross(m[0],m[1]);
                mt.transpose(m);
                bases[c0+c].invert(mt);
                centers[c0+c] = (in[quads[c][0]]+in[quads[c][1]]+in[quads[c][2]]+in[quads[c][3]])*0.25;
            }
            for (unsigned int i=0; i<out.size(); i++)
            {
                Vector3 pos = out[i];
                Vector3 coefs;
                int index = -1;
                double distance = 1e10;
                for (unsigned int t = 0; t < triangles.size(); t++)
                {
                    Vec3d v = bases[t] * (pos - in[triangles[t][0]]);
                    double d = std::max(std::max(-v[0],-v[1]),std::max((v[2]<0?-v[2]:v[2])-0.01,v[0]+v[1]-1));
                    if (d>0) d = (pos-centers[t]).norm2();
                    if (d<distance) { coefs = v; distance = d; index = t; }
                }
                for (unsigned int c = 0; c < quads.size(); c++)
                {
                    Vec3d v = bases[c0+c] * (pos - in[quads[c][0]]);
                    double d = std::max(std::max(-v[0],-v[1]),std::max(std::max(v[1]-1,v[0]-1),std::max(v[2]-0.01,-v[2]-0.01)));
                    if (d>0) d = (pos-centers[c0+c]).norm2();
                    if (d<distance) { coefs = v; distance = d; index = c0+c; }
                }
                if (distance>0)
                {
                    ++outside;
                }
                if (index < c0)
                    addPointInTriangle(index, coefs.ptr());
                else
                    addPointInQuad(index-c0, coefs.ptr());
            }
        }
    }
    else
    {
        clear3d(out.size()); // reserve space for 3D mapping
        int c0 = tetras.size();
        bases.resize(tetras.size()+cubes.size());
        centers.resize(tetras.size()+cubes.size());
        for (unsigned int t = 0; t < tetras.size(); t++)
        {
            Mat3x3d m,mt;
            m[0] = in[tetras[t][1]]-in[tetras[t][0]];
            m[1] = in[tetras[t][2]]-in[tetras[t][0]];
            m[2] = in[tetras[t][3]]-in[tetras[t][0]];
            mt.transpose(m);
            bases[t].invert(mt);
            centers[t] = (in[tetras[t][0]]+in[tetras[t][1]]+in[tetras[t][2]]+in[tetras[t][3]])*0.25;
            //std::cout << "Tetra "<<t<<" center="<<centers[t]<<" base="<<m<<std::endl;
        }
        for (unsigned int c = 0; c < cubes.size(); c++)
        {
            Mat3x3d m,mt;
            m[0] = in[cubes[c][1]]-in[cubes[c][0]];
#ifdef SOFA_NEW_HEXA
            m[1] = in[cubes[c][3]]-in[cubes[c][0]];
#else
            m[1] = in[cubes[c][2]]-in[cubes[c][0]];
#endif
            m[2] = in[cubes[c][4]]-in[cubes[c][0]];
            mt.transpose(m);
            bases[c0+c].invert(mt);
            centers[c0+c] = (in[cubes[c][0]]+in[cubes[c][1]]+in[cubes[c][2]]+in[cubes[c][3]]+in[cubes[c][4]]+in[cubes[c][5]]+in[cubes[c][6]]+in[cubes[c][7]])*0.125;
        }
        for (unsigned int i=0; i<out.size(); i++)
        {
            Vector3 pos = out[i];
            Vector3 coefs;
            int index = -1;
            double distance = 1e10;
            for (unsigned int t = 0; t < tetras.size(); t++)
            {
                Vector3 v = bases[t] * (pos - in[tetras[t][0]]);
                double d = std::max(std::max(-v[0],-v[1]),std::max(-v[2],v[0]+v[1]+v[2]-1));
                if (d>0) d = (pos-centers[t]).norm2();
                if (d<distance) { coefs = v; distance = d; index = t; }
            }
            for (unsigned int c = 0; c < cubes.size(); c++)
            {
                Vector3 v = bases[c0+c] * (pos - in[cubes[c][0]]);
                double d = std::max(std::max(-v[0],-v[1]),std::max(std::max(-v[2],v[0]-1),std::max(v[1]-1,v[2]-1)));
                if (d>0) d = (pos-centers[c0+c]).norm2();
                if (d<distance) { coefs = v; distance = d; index = c0+c; }
            }
            if (distance>0)
            {
                ++outside;
            }
            if (index < c0)
                addPointInTetra(index, coefs.ptr());
            else
                addPointInCube(index-c0, coefs.ptr());
        }
    }
}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::clear(int reserve)
{
    map.clear(); if (reserve>0) map.reserve(reserve);
}


template <class In, class Out>
int BarycentricMapperEdgeSetTopology<In,Out>::addPointInLine(int edgeIndex, const SReal* baryCoords)
{
    map.resize(map.size()+1);
    MappingData& data = *map.rbegin();
    data.in_index = edgeIndex;
    data.baryCoords[0] = (Real)baryCoords[0];
    return map.size()-1;
}

template <class In, class Out>
int BarycentricMapperEdgeSetTopology<In,Out>::createPointInLine(const typename Out::Coord& p, int edgeIndex, const typename In::VecCoord* points)
{
    SReal baryCoords[1];
    const topology::Edge& elem = this->topology->getEdge(edgeIndex);
    const typename In::Coord p0 = (*points)[elem[0]];
    const typename In::Coord pA = (*points)[elem[1]] - p0;
    typename In::Coord pos = p - p0;
    baryCoords[0] = dot(pA,pos)/dot(pA,pA);
    return this->addPointInLine(edgeIndex, baryCoords);
}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::init(const typename Out::VecCoord& /*out*/, const typename In::VecCoord& /*in*/)
{
    _container->getContext()->get(_geomAlgo);
//	int outside = 0;
//	const sofa::helper::vector<topology::Edge>& edges = this->topology->getEdges();
    //TODO: implementation of BarycentricMapperEdgeSetTopology::init
}

template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::clear(int reserve)
{
    map.clear(); if (reserve>0) map.reserve(reserve);
}

template <class In, class Out>
int BarycentricMapperTriangleSetTopology<In,Out>::addPointInTriangle(int triangleIndex, const SReal* baryCoords)
{
    map.resize(map.size()+1);
    MappingData& data = *map.rbegin();
    data.in_index = triangleIndex;
    data.baryCoords[0] = (Real)baryCoords[0];
    data.baryCoords[1] = (Real)baryCoords[1];
    return map.size()-1;
}

template <class In, class Out>
int BarycentricMapperTriangleSetTopology<In,Out>::createPointInTriangle(const typename Out::Coord& p, int triangleIndex, const typename In::VecCoord* points)
{
    SReal baryCoords[2];
    const topology::Triangle& elem = this->topology->getTriangle(triangleIndex);
    const typename In::Coord p0 = (*points)[elem[0]];
    const typename In::Coord pA = (*points)[elem[1]] - p0;
    const typename In::Coord pB = (*points)[elem[2]] - p0;
    typename In::Coord pos = p - p0;
    // First project to plane
    typename In::Coord normal = cross(pA, pB);
    Real norm2 = normal.norm2();
    pos -= normal*((pos*normal)/norm2);
    baryCoords[0] = (Real)sqrt(cross(pB, pos).norm2() / norm2);
    baryCoords[1] = (Real)sqrt(cross(pA, pos).norm2() / norm2);
    return this->addPointInTriangle(triangleIndex, baryCoords);
}

template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::init(const typename Out::VecCoord& out, const typename In::VecCoord& in)
{
    _container->getContext()->get(_geomAlgo);

    int outside = 0;

    const sofa::helper::vector<topology::Triangle>& triangles = this->topology->getTriangles();
    sofa::helper::vector<Mat3x3d> bases;
    sofa::helper::vector<Vector3> centers;

    // no 3D elements -> map on 2D elements
    clear(out.size()); // reserve space for 2D mapping
    bases.resize(triangles.size());
    centers.resize(triangles.size());

    for (unsigned int t = 0; t < triangles.size(); t++)
    {
        Mat3x3d m,mt;
        m[0] = in[triangles[t][1]]-in[triangles[t][0]];
        m[1] = in[triangles[t][2]]-in[triangles[t][0]];
        m[2] = cross(m[0],m[1]);
        mt.transpose(m);
        bases[t].invert(mt);
        centers[t] = (in[triangles[t][0]]+in[triangles[t][1]]+in[triangles[t][2]])/3;
    }

    for (unsigned int i=0; i<out.size(); i++)
    {
        Vec3d pos = out[i];
        Vector3 coefs;
        int index = -1;
        double distance = 1e10;
        for (unsigned int t = 0; t < triangles.size(); t++)
        {
            Vec3d v = bases[t] * (pos - in[triangles[t][0]]);
            double d = std::max(std::max(-v[0],-v[1]),std::max((v[2]<0?-v[2]:v[2])-0.01,v[0]+v[1]-1));
            if (d>0) d = (pos-centers[t]).norm2();
            if (d<distance) { coefs = v; distance = d; index = t; }
        }
        if (distance>0)
        {
            ++outside;
        }
        addPointInTriangle(index, coefs.ptr());
    }
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::clear(int reserve)
{
    map.clear(); if (reserve>0) map.reserve(reserve);
}

template <class In, class Out>
int BarycentricMapperQuadSetTopology<In,Out>::addPointInQuad(int quadIndex, const SReal* baryCoords)
{
    map.resize(map.size()+1);
    MappingData& data = *map.rbegin();
    data.in_index = quadIndex;
    data.baryCoords[0] = (Real)baryCoords[0];
    data.baryCoords[1] = (Real)baryCoords[1];
    return map.size()-1;
}

template <class In, class Out>
int BarycentricMapperQuadSetTopology<In,Out>::createPointInQuad(const typename Out::Coord& p, int quadIndex, const typename In::VecCoord* points)
{
    SReal baryCoords[2];
    const topology::Quad& elem = this->topology->getQuad(quadIndex);
    const typename In::Coord p0 = (*points)[elem[0]];
    const typename In::Coord pA = (*points)[elem[1]] - p0;
    const typename In::Coord pB = (*points)[elem[3]] - p0;
    typename In::Coord pos = p - p0;
    Mat<3,3,typename In::Real> m,mt,base;
    m[0] = pA;
    m[1] = pB;
    m[2] = cross(pA, pB);
    mt.transpose(m);
    base.invert(mt);
    const typename In::Coord base0 = base[0];
    const typename In::Coord base1 = base[1];
    baryCoords[0] = base0 * pos;
    baryCoords[1] = base1 * pos;
    return this->addPointInQuad(quadIndex, baryCoords);
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::init(const typename Out::VecCoord& out, const typename In::VecCoord& in)
{
    _container->getContext()->get(_geomAlgo);

    int outside = 0;
    const sofa::helper::vector<topology::Quad>& quads = this->topology->getQuads();

    sofa::helper::vector<Matrix3> bases;
    sofa::helper::vector<Vector3> centers;

    clear(out.size());
    bases.resize(quads.size());
    centers.resize(quads.size());

    for (unsigned int c = 0; c < quads.size(); c++)
    {
        Mat3x3d m,mt;
        m[0] = in[quads[c][1]]-in[quads[c][0]];
        m[1] = in[quads[c][3]]-in[quads[c][0]];
        m[2] = cross(m[0],m[1]);
        mt.transpose(m);
        bases[c].invert(mt);
        centers[c] = (in[quads[c][0]]+in[quads[c][1]]+in[quads[c][2]]+in[quads[c][3]])*0.25;
    }

    for (unsigned int i=0; i<out.size(); i++)
    {
        Vec3d pos = out[i];
        Vector3 coefs;
        int index = -1;
        double distance = 1e10;
        for (unsigned int c = 0; c < quads.size(); c++)
        {
            Vec3d v = bases[c] * (pos - in[quads[c][0]]);
            double d = std::max(std::max(-v[0],-v[1]),std::max(std::max(v[1]-1,v[0]-1),std::max(v[2]-0.01,-v[2]-0.01)));
            if (d>0) d = (pos-centers[c]).norm2();
            if (d<distance) { coefs = v; distance = d; index = c; }
        }
        if (distance>0)
        {
            ++outside;
        }
        addPointInQuad(index, coefs.ptr());
    }
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::clear(int reserve)
{
    map.clear(); if (reserve>0) map.reserve(reserve);
}

template <class In, class Out>
int BarycentricMapperTetrahedronSetTopology<In,Out>::addPointInTetra(int tetraIndex, const SReal* baryCoords)
{
    map.resize(map.size()+1);
    MappingData& data = *map.rbegin();
    data.in_index = tetraIndex;
    data.baryCoords[0] = (Real)baryCoords[0];
    data.baryCoords[1] = (Real)baryCoords[1];
    data.baryCoords[2] = (Real)baryCoords[2];
    return map.size()-1;
}

//template <class In, class Out>
//int BarycentricMapperTetrahedronSetTopology<In,Out>::createPointInTetra(const typename Out::Coord& p, int index, const typename In::VecCoord* points)
//{
//	//TODO: add implementation
//}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::init(const typename Out::VecCoord& out, const typename In::VecCoord& in)
{
    _container->getContext()->get(_geomAlgo);

    int outside = 0;
    const sofa::helper::vector<topology::Tetrahedron>& tetras = this->topology->getTetras();

    sofa::helper::vector<Matrix3> bases;
    sofa::helper::vector<Vector3> centers;

    clear(out.size());
    bases.resize(tetras.size());
    centers.resize(tetras.size());
    for (unsigned int t = 0; t < tetras.size(); t++)
    {
        Mat3x3d m,mt;
        m[0] = in[tetras[t][1]]-in[tetras[t][0]];
        m[1] = in[tetras[t][2]]-in[tetras[t][0]];
        m[2] = in[tetras[t][3]]-in[tetras[t][0]];
        mt.transpose(m);
        bases[t].invert(mt);
        centers[t] = (in[tetras[t][0]]+in[tetras[t][1]]+in[tetras[t][2]]+in[tetras[t][3]])*0.25;
    }

    for (unsigned int i=0; i<out.size(); i++)
    {
        Vec3d pos = out[i];
        Vector3 coefs;
        int index = -1;
        double distance = 1e10;
        for (unsigned int t = 0; t < tetras.size(); t++)
        {
            Vec3d v = bases[t] * (pos - in[tetras[t][0]]);
            double d = std::max(std::max(-v[0],-v[1]),std::max(-v[2],v[0]+v[1]+v[2]-1));
            if (d>0) d = (pos-centers[t]).norm2();
            if (d<distance) { coefs = v; distance = d; index = t; }
        }
        if (distance>0)
        {
            ++outside;
        }
        addPointInTetra(index, coefs.ptr());
    }
}

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::clear(int reserve)
{
    map.clear();
    if (reserve>0) map.reserve(reserve);
}

template <class In, class Out>
int BarycentricMapperHexahedronSetTopology<In,Out>::addPointInCube(int cubeIndex, const SReal* baryCoords)
{
    map.resize(map.size()+1);
    MappingData& data = *map.rbegin();
    data.in_index = cubeIndex;
    data.baryCoords[0] = (Real)baryCoords[0];
    data.baryCoords[1] = (Real)baryCoords[1];
    data.baryCoords[2] = (Real)baryCoords[2];
    return map.size()-1;
}

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::init(const typename Out::VecCoord& out,
        const typename In::VecCoord& /*in*/)
{
    _container->getContext()->get(_geomAlgo);

    if(_geomAlgo == NULL)
    {
        std::cerr << "Error [BarycentricMapperHexahedronSetTopology::init] cannot find GeometryAlgorithms component." << endl;
    }

    clear(out.size());

    for (unsigned int i=0; i<out.size(); ++i)
    {
        Vector3 coefs;
        typename In::Real distance;
        const int index = _geomAlgo->findNearestElement(out[i], coefs, distance);

        if(index != -1)
            addPointInCube(index, coefs.ptr());
        else
            std::cerr << "Error [BarycentricMapperHexahedronSetTopology::init] cannot find a cell for barycentric mapping." << std::endl;
    }
}

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::createMapperFromTopology(BaseMeshTopology * topology)
{
    mapper = NULL;

    core::componentmodel::topology::TopologyContainer* topoCont2;
    this->fromModel->getContext()->get(topoCont2);

    if (topoCont2!=NULL)
    {

        topology::HexahedronSetTopologyContainer* t1 = dynamic_cast<topology::HexahedronSetTopologyContainer*>(topoCont2);
        if(t1 != NULL)
        {
            typedef BarycentricMapperHexahedronSetTopology<InDataTypes, OutDataTypes> HexahedronSetMapper;
            mapper = new HexahedronSetMapper(t1);
        }
        else
        {
            topology::TetrahedronSetTopologyContainer* t2 = dynamic_cast<topology::TetrahedronSetTopologyContainer*>(topoCont2);
            if(t2 != NULL)
            {
                typedef BarycentricMapperTetrahedronSetTopology<InDataTypes, OutDataTypes> TetrahedronSetMapper;
                mapper = new TetrahedronSetMapper(t2);
            }
            else
            {
                topology::QuadSetTopologyContainer* t3 = dynamic_cast<topology::QuadSetTopologyContainer*>(topoCont2);
                if(t3 != NULL)
                {
                    typedef BarycentricMapperQuadSetTopology<InDataTypes, OutDataTypes> QuadSetMapper;
                    mapper = new QuadSetMapper(t3);
                }
                else
                {
                    topology::TriangleSetTopologyContainer* t4 = dynamic_cast<topology::TriangleSetTopologyContainer*>(topoCont2);
                    if (t4 != NULL)
                    {
                        typedef BarycentricMapperTriangleSetTopology<InDataTypes, OutDataTypes> TriangleSetMapper;
                        mapper = new TriangleSetMapper(t4);
                    }
                    else
                    {
                        topology::EdgeSetTopologyContainer* t5 = dynamic_cast<topology::EdgeSetTopologyContainer*>(topoCont2);
                        if(t5 != NULL)
                        {
                            typedef BarycentricMapperEdgeSetTopology<InDataTypes, OutDataTypes> EdgeSetMapper;
                            mapper = new EdgeSetMapper(t5);
                        }
                    }
                }
            }
        }
    }
    else
    {

        topology::RegularGridTopology* t2 = dynamic_cast<topology::RegularGridTopology*>(topology);
        if (t2!=NULL)
        {
            typedef BarycentricMapperRegularGridTopology<InDataTypes, OutDataTypes> RegularGridMapper;
            if (f_grid->beginEdit()->isEmpty())
            {
                f_grid->setValue(RegularGridMapper(t2));
                mapper = f_grid->beginEdit();
            }
            else
            {
                f_grid->beginEdit()->setTopology(t2);
                this->mapper = f_grid->beginEdit();
            }

        }
        else
        {
            topology::SparseGridTopology* t4 = dynamic_cast<topology::SparseGridTopology*>(topology);
            if (t4!=NULL)
            {
                typedef BarycentricMapperSparseGridTopology<InDataTypes, OutDataTypes> SparseGridMapper;
                mapper = new SparseGridMapper(t4);
            }
            else // generic MeshTopology
            {
                typedef BarycentricMapperMeshTopology<InDataTypes, OutDataTypes> MeshMapper;
                topology::BaseMeshTopology* t3 = dynamic_cast<topology::BaseMeshTopology*>(topology);
                mapper = new MeshMapper(t3);
            }
        }
    }
}

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::init()
{
    topology_from = this->fromModel->getContext()->getMeshTopology();
    topology_to = this->toModel->getContext()->getMeshTopology();

    f_grid->beginEdit();
    if(mapper == NULL) // try to create a mapper according to the topology of the In model
    {
        if (topology_from!=NULL)
        {
            createMapperFromTopology(topology_from);
        }
    }


    if (mapper != NULL)
    {
        mapper->init(*this->toModel->getX(), *this->fromModel->getX());
    }
    else
    {
        std::cerr << "ERROR: Barycentric mapping does not understand topology."<<std::endl;
    }

    this->BasicMapping::init();

}

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    if (mapper!=NULL) mapper->apply(out, in);
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize(map1d.size()+map2d.size()+map3d.size());
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqLines& lines = this->topology->getLines();
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqTriangles& triangles = this->topology->getTriangles();
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqQuads& quads = this->topology->getQuads();
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqTetras& tetras = this->topology->getTetras();
#ifdef SOFA_NEW_HEXA
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqHexas& cubes = this->topology->getHexas();
#else
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqCubes& cubes = this->topology->getCubes();
#endif
    // 1D elements
    {
        for(unsigned int i=0; i<map1d.size(); i++)
        {
            const Real fx = map1d[i].baryCoords[0];
            int index = map1d[i].in_index;
            {
                const sofa::core::componentmodel::topology::BaseMeshTopology::Line& line = lines[index];
                out[i] = in[line[0]] * (1-fx)
                        + in[line[1]] * fx;
            }
        }
    }
    // 2D elements
    {
        const int i0 = map1d.size();
        const int c0 = triangles.size();
        for(unsigned int i=0; i<map2d.size(); i++)
        {
            const Real fx = map2d[i].baryCoords[0];
            const Real fy = map2d[i].baryCoords[1];
            int index = map2d[i].in_index;
            std::cout << index << " " << triangles.size() << " !!!!!!!\n";
            if (index<c0)
            {
                const sofa::core::componentmodel::topology::BaseMeshTopology::Triangle& triangle = triangles[index];
                out[i+i0] = in[triangle[0]] * (1-fx-fy)
                        + in[triangle[1]] * fx
                        + in[triangle[2]] * fy;
            }
            else
            {
                const sofa::core::componentmodel::topology::BaseMeshTopology::Quad& quad = quads[index-c0];
                out[i+i0] = in[quad[0]] * ((1-fx) * (1-fy))
                        + in[quad[1]] * ((  fx) * (1-fy))
                        + in[quad[3]] * ((1-fx) * (  fy))
                        + in[quad[2]] * ((  fx) * (  fy));
            }
        }
    }
    // 3D elements
    {
        const int i0 = map1d.size() + map2d.size();
        const int c0 = tetras.size();
        for(unsigned int i=0; i<map3d.size(); i++)
        {
            const Real fx = map3d[i].baryCoords[0];
            const Real fy = map3d[i].baryCoords[1];
            const Real fz = map3d[i].baryCoords[2];
            int index = map3d[i].in_index;
            if (index<c0)
            {
                const sofa::core::componentmodel::topology::BaseMeshTopology::Tetra& tetra = tetras[index];
                out[i+i0] = in[tetra[0]] * (1-fx-fy-fz)
                        + in[tetra[1]] * fx
                        + in[tetra[2]] * fy
                        + in[tetra[3]] * fz;
            }
            else
            {
#ifdef SOFA_NEW_HEXA
                const sofa::core::componentmodel::topology::BaseMeshTopology::Hexa& cube = cubes[index-c0];
#else
                const sofa::core::componentmodel::topology::BaseMeshTopology::Cube& cube = cubes[index-c0];
#endif
                out[i+i0] = in[cube[0]] * ((1-fx) * (1-fy) * (1-fz))
                        + in[cube[1]] * ((  fx) * (1-fy) * (1-fz))
#ifdef SOFA_NEW_HEXA
                        + in[cube[3]] * ((1-fx) * (  fy) * (1-fz))
                        + in[cube[2]] * ((  fx) * (  fy) * (1-fz))
#else
                        + in[cube[2]] * ((1-fx) * (  fy) * (1-fz))
                        + in[cube[3]] * ((  fx) * (  fy) * (1-fz))
#endif
                        + in[cube[4]] * ((1-fx) * (1-fy) * (  fz))
                        + in[cube[5]] * ((  fx) * (1-fy) * (  fz))
#ifdef SOFA_NEW_HEXA
                        + in[cube[7]] * ((1-fx) * (  fy) * (  fz))
                        + in[cube[6]] * ((  fx) * (  fy) * (  fz));
#else
                        + in[cube[6]] * ((1-fx) * (  fy) * (  fz))
                        + in[cube[7]] * ((  fx) * (  fy) * (  fz));
#endif
            }
        }
    }
}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize(map.size());
    const sofa::helper::vector<topology::Edge>& edges = this->topology->getEdges();
    // 2D elements
    for(unsigned int i=0; i<map.size(); i++)
    {
        const Real fx = map[i].baryCoords[0];
        int index = map[i].in_index;
        const topology::Edge& edge = edges[index];
        out[i] = in[edge[0]] * (1-fx)
                + in[edge[1]] * fx;
    }
}

template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize(map.size());
    const sofa::helper::vector<topology::Triangle>& triangles = this->topology->getTriangles();
    for(unsigned int i=0; i<map.size(); i++)
    {
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        int index = map[i].in_index;
        const topology::Triangle& triangle = triangles[index];
        out[i] = in[triangle[0]] * (1-fx-fy)
                + in[triangle[1]] * fx
                + in[triangle[2]] * fy;
    }
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize(map.size());
    const sofa::helper::vector<topology::Quad>& quads = this->topology->getQuads();
    for(unsigned int i=0; i<map.size(); i++)
    {
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        int index = map[i].in_index;
        const topology::Quad& quad = quads[index];
        out[i] = in[quad[0]] * ((1-fx) * (1-fy))
                + in[quad[1]] * ((  fx) * (1-fy))
                + in[quad[3]] * ((1-fx) * (  fy))
                + in[quad[2]] * ((  fx) * (  fy));
    }
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize(map.size());
    const sofa::helper::vector<topology::Tetrahedron>& tetras = this->topology->getTetras();
    for(unsigned int i=0; i<map.size(); i++)
    {
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        int index = map[i].in_index;
        const topology::Tetrahedron& tetra = tetras[index];
        out[i] = in[tetra[0]] * (1-fx-fy-fz)
                + in[tetra[1]] * fx
                + in[tetra[2]] * fy
                + in[tetra[3]] * fz;
    }
    //cerr<<"BarycentricMapperTetrahedronSetTopology<In,Out>::apply, in = "<<in<<endl;
    //cerr<<"BarycentricMapperTetrahedronSetTopology<In,Out>::apply, out = "<<out<<endl;
}

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize(map.size());
    const sofa::helper::vector<topology::Hexahedron>& cubes = this->topology->getHexas();
    for(unsigned int i=0; i<map.size(); i++)
    {
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        int index = map[i].in_index;
        const topology::Hexahedron& cube = cubes[index];
        out[i] = in[cube[0]] * ((1-fx) * (1-fy) * (1-fz))
                + in[cube[1]] * ((  fx) * (1-fy) * (1-fz))
                + in[cube[3]] * ((1-fx) * (  fy) * (1-fz))
                + in[cube[2]] * ((  fx) * (  fy) * (1-fz))
                + in[cube[4]] * ((1-fx) * (1-fy) * (  fz))
                + in[cube[5]] * ((  fx) * (1-fy) * (  fz))
                + in[cube[7]] * ((1-fx) * (  fy) * (  fz))
                + in[cube[6]] * ((  fx) * (  fy) * (  fz));
    }
}

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize(this->toModel->getX()->size());
    if (mapper!=NULL) mapper->applyJ(out, in);
}


template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    //cerr<<"    BarycentricMapping<BasicMapping>::MeshMapper::applyJ"<<endl;
    out.resize(map1d.size()+map2d.size()+map3d.size());
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqLines& lines = this->topology->getLines();
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqTriangles& triangles = this->topology->getTriangles();
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqQuads& quads = this->topology->getQuads();
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqTetras& tetras = this->topology->getTetras();
#ifdef SOFA_NEW_HEXA
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqHexas& cubes = this->topology->getHexas();
#else
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqCubes& cubes = this->topology->getCubes();
#endif
    // 1D elements
    {
        for(unsigned int i=0; i<map1d.size(); i++)
        {
            const Real fx = map1d[i].baryCoords[0];
            int index = map1d[i].in_index;
            {
                const sofa::core::componentmodel::topology::BaseMeshTopology::Line& line = lines[index];
                out[i] = in[line[0]] * (1-fx)
                        + in[line[1]] * fx;
            }
        }
    }
    // 2D elements
    {
        const int i0 = map1d.size();
        const int c0 = triangles.size();
        for(unsigned int i=0; i<map2d.size(); i++)
        {
            const Real fx = map2d[i].baryCoords[0];
            const Real fy = map2d[i].baryCoords[1];
            int index = map2d[i].in_index;
            if (index<c0)
            {
                const sofa::core::componentmodel::topology::BaseMeshTopology::Triangle& triangle = triangles[index];
                out[i+i0] = in[triangle[0]] * (1-fx-fy)
                        + in[triangle[1]] * fx
                        + in[triangle[2]] * fy;
            }
            else
            {
                const sofa::core::componentmodel::topology::BaseMeshTopology::Quad& quad = quads[index-c0];
                out[i+i0] = in[quad[0]] * ((1-fx) * (1-fy))
                        + in[quad[1]] * ((  fx) * (1-fy))
                        + in[quad[3]] * ((1-fx) * (  fy))
                        + in[quad[2]] * ((  fx) * (  fy));
            }
        }
    }
    // 3D elements
    {
        const int i0 = map1d.size() + map2d.size();
        const int c0 = tetras.size();
        for(unsigned int i=0; i<map3d.size(); i++)
        {
            const Real fx = map3d[i].baryCoords[0];
            const Real fy = map3d[i].baryCoords[1];
            const Real fz = map3d[i].baryCoords[2];
            int index = map3d[i].in_index;
            if (index<c0)
            {
                const sofa::core::componentmodel::topology::BaseMeshTopology::Tetra& tetra = tetras[index];
                out[i+i0] = in[tetra[0]] * (1-fx-fy-fz)
                        + in[tetra[1]] * fx
                        + in[tetra[2]] * fy
                        + in[tetra[3]] * fz;
            }
            else
            {
#ifdef SOFA_NEW_HEXA
                const sofa::core::componentmodel::topology::BaseMeshTopology::Hexa& cube = cubes[index-c0];
#else
                const sofa::core::componentmodel::topology::BaseMeshTopology::Cube& cube = cubes[index-c0];
#endif
                out[i+i0] = in[cube[0]] * ((1-fx) * (1-fy) * (1-fz))
                        + in[cube[1]] * ((  fx) * (1-fy) * (1-fz))
#ifdef SOFA_NEW_HEXA
                        + in[cube[3]] * ((1-fx) * (  fy) * (1-fz))
                        + in[cube[2]] * ((  fx) * (  fy) * (1-fz))
#else
                        + in[cube[2]] * ((1-fx) * (  fy) * (1-fz))
                        + in[cube[3]] * ((  fx) * (  fy) * (1-fz))
#endif
                        + in[cube[4]] * ((1-fx) * (1-fy) * (  fz))
                        + in[cube[5]] * ((  fx) * (1-fy) * (  fz))
#ifdef SOFA_NEW_HEXA
                        + in[cube[7]] * ((1-fx) * (  fy) * (  fz))
                        + in[cube[6]] * ((  fx) * (  fy) * (  fz));
#else
                        + in[cube[6]] * ((1-fx) * (  fy) * (  fz))
                        + in[cube[7]] * ((  fx) * (  fy) * (  fz));
#endif
            }
        }
    }
}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize(map.size());
    const sofa::helper::vector<topology::Edge>& edges = this->topology->getEdges();
    for(unsigned int i=0; i<map.size(); i++)
    {
        const Real fx = map[i].baryCoords[0];
        int index = map[i].in_index;
        const topology::Edge& edge = edges[index];
        out[i] = in[edge[0]] * (1-fx)
                + in[edge[1]] * fx;
    }
}

template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize(map.size());
    const sofa::helper::vector<topology::Triangle>& triangles = this->topology->getTriangles();
    for(unsigned int i=0; i<map.size(); i++)
    {
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        int index = map[i].in_index;
        const topology::Triangle& triangle = triangles[index];
        out[i] = in[triangle[0]] * (1-fx-fy)
                + in[triangle[1]] * fx
                + in[triangle[2]] * fy;
    }
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize(map.size());
    const sofa::helper::vector<topology::Quad>& quads = this->topology->getQuads();
    for(unsigned int i=0; i<map.size(); i++)
    {
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        int index = map[i].in_index;
        const topology::Quad& quad = quads[index];
        out[i] = in[quad[0]] * ((1-fx) * (1-fy))
                + in[quad[1]] * ((  fx) * (1-fy))
                + in[quad[3]] * ((1-fx) * (  fy))
                + in[quad[2]] * ((  fx) * (  fy));
    }
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize(map.size());
    const sofa::helper::vector<topology::Tetrahedron>& tetras = this->topology->getTetras();
    for(unsigned int i=0; i<map.size(); i++)
    {
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        int index = map[i].in_index;
        const topology::Tetrahedron& tetra = tetras[index];
        out[i] = in[tetra[0]] * (1-fx-fy-fz)
                + in[tetra[1]] * fx
                + in[tetra[2]] * fy
                + in[tetra[3]] * fz;
    }
}

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize(map.size());
    const sofa::helper::vector<topology::Hexahedron>& cubes = this->topology->getHexas();
    for(unsigned int i=0; i<map.size(); i++)
    {
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        int index = map[i].in_index;
        const topology::Hexahedron& cube = cubes[index];
        out[i] = in[cube[0]] * ((1-fx) * (1-fy) * (1-fz))
                + in[cube[1]] * ((  fx) * (1-fy) * (1-fz))
                + in[cube[3]] * ((1-fx) * (  fy) * (1-fz))
                + in[cube[2]] * ((  fx) * (  fy) * (1-fz))
                + in[cube[4]] * ((1-fx) * (1-fy) * (  fz))
                + in[cube[5]] * ((  fx) * (1-fy) * (  fz))
                + in[cube[7]] * ((1-fx) * (  fy) * (  fz))
                + in[cube[6]] * ((  fx) * (  fy) * (  fz));
    }
}

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    if (mapper!=NULL) mapper->applyJT(out, in);
}


template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqLines& lines = this->topology->getLines();
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqTriangles& triangles = this->topology->getTriangles();
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqQuads& quads = this->topology->getQuads();
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqTetras& tetras = this->topology->getTetras();
#ifdef SOFA_NEW_HEXA
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqHexas& cubes = this->topology->getHexas();
#else
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqCubes& cubes = this->topology->getCubes();
#endif
    // 1D elements
    {
        for(unsigned int i=0; i<map1d.size(); i++)
        {
            const typename Out::Deriv v = in[i];
            const OutReal fx = (OutReal)map1d[i].baryCoords[0];
            int index = map1d[i].in_index;
            {
                const sofa::core::componentmodel::topology::BaseMeshTopology::Line& line = lines[index];
                out[line[0]] += v * (1-fx);
                out[line[1]] += v * fx;
            }
        }
    }
    // 2D elements
    {
        const int i0 = map1d.size();
        const int c0 = triangles.size();
        for(unsigned int i=0; i<map2d.size(); i++)
        {
            const typename Out::Deriv v = in[i+i0];
            const OutReal fx = (OutReal)map2d[i].baryCoords[0];
            const OutReal fy = (OutReal)map2d[i].baryCoords[1];
            int index = map2d[i].in_index;
            if (index<c0)
            {
                const sofa::core::componentmodel::topology::BaseMeshTopology::Triangle& triangle = triangles[index];
                out[triangle[0]] += v * (1-fx-fy);
                out[triangle[1]] += v * fx;
                out[triangle[2]] += v * fy;
            }
            else
            {
                const sofa::core::componentmodel::topology::BaseMeshTopology::Quad& quad = quads[index-c0];
                out[quad[0]] += v * ((1-fx) * (1-fy));
                out[quad[1]] += v * ((  fx) * (1-fy));
                out[quad[3]] += v * ((1-fx) * (  fy));
                out[quad[2]] += v * ((  fx) * (  fy));
            }
        }
    }
    // 3D elements
    {
        const int i0 = map1d.size() + map2d.size();
        const int c0 = tetras.size();
        for(unsigned int i=0; i<map3d.size(); i++)
        {
            const typename Out::Deriv v = in[i+i0];
            const OutReal fx = (OutReal)map3d[i].baryCoords[0];
            const OutReal fy = (OutReal)map3d[i].baryCoords[1];
            const OutReal fz = (OutReal)map3d[i].baryCoords[2];
            int index = map3d[i].in_index;
            if (index<c0)
            {
                const sofa::core::componentmodel::topology::BaseMeshTopology::Tetra& tetra = tetras[index];
                out[tetra[0]] += v * (1-fx-fy-fz);
                out[tetra[1]] += v * fx;
                out[tetra[2]] += v * fy;
                out[tetra[3]] += v * fz;
            }
            else
            {
#ifdef SOFA_NEW_HEXA
                const sofa::core::componentmodel::topology::BaseMeshTopology::Hexa& cube = cubes[index-c0];
#else
                const sofa::core::componentmodel::topology::BaseMeshTopology::Cube& cube = cubes[index-c0];
#endif
                out[cube[0]] += v * ((1-fx) * (1-fy) * (1-fz));
                out[cube[1]] += v * ((  fx) * (1-fy) * (1-fz));
#ifdef SOFA_NEW_HEXA
                out[cube[3]] += v * ((1-fx) * (  fy) * (1-fz));
                out[cube[2]] += v * ((  fx) * (  fy) * (1-fz));
#else
                out[cube[2]] += v * ((1-fx) * (  fy) * (1-fz));
                out[cube[3]] += v * ((  fx) * (  fy) * (1-fz));
#endif
                out[cube[4]] += v * ((1-fx) * (1-fy) * (  fz));
                out[cube[5]] += v * ((  fx) * (1-fy) * (  fz));
#ifdef SOFA_NEW_HEXA
                out[cube[7]] += v * ((1-fx) * (  fy) * (  fz));
                out[cube[6]] += v * ((  fx) * (  fy) * (  fz));
#else
                out[cube[6]] += v * ((1-fx) * (  fy) * (  fz));
                out[cube[7]] += v * ((  fx) * (  fy) * (  fz));
#endif
            }
        }
    }
}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const sofa::helper::vector<topology::Edge>& edges = this->topology->getEdges();
    for(unsigned int i=0; i<map.size(); i++)
    {
        const typename Out::Deriv v = in[i];
        const OutReal fx = (OutReal)map[i].baryCoords[0];
        int index = map[i].in_index;
        const topology::Edge& edge = edges[index];
        out[edge[0]] += v * (1-fx);
        out[edge[1]] += v * fx;
    }
}

template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const sofa::helper::vector<topology::Triangle>& triangles = this->topology->getTriangles();
    for(unsigned int i=0; i<map.size(); i++)
    {
        const typename Out::Deriv v = in[i];
        const OutReal fx = (OutReal)map[i].baryCoords[0];
        const OutReal fy = (OutReal)map[i].baryCoords[1];
        int index = map[i].in_index;
        const topology::Triangle& triangle = triangles[index];
        out[triangle[0]] += v * (1-fx-fy);
        out[triangle[1]] += v * fx;
        out[triangle[2]] += v * fy;
    }
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const sofa::helper::vector<topology::Quad>& quads = this->topology->getQuads();
    for(unsigned int i=0; i<map.size(); i++)
    {
        const typename Out::Deriv v = in[i];
        const OutReal fx = (OutReal)map[i].baryCoords[0];
        const OutReal fy = (OutReal)map[i].baryCoords[1];
        int index = map[i].in_index;
        const topology::Quad& quad = quads[index];
        out[quad[0]] += v * ((1-fx) * (1-fy));
        out[quad[1]] += v * ((  fx) * (1-fy));
        out[quad[3]] += v * ((1-fx) * (  fy));
        out[quad[2]] += v * ((  fx) * (  fy));
    }
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const sofa::helper::vector<topology::Tetrahedron>& tetras = this->topology->getTetras();
    for(unsigned int i=0; i<map.size(); i++)
    {
        const typename Out::Deriv v = in[i];
        const OutReal fx = (OutReal)map[i].baryCoords[0];
        const OutReal fy = (OutReal)map[i].baryCoords[1];
        const OutReal fz = (OutReal)map[i].baryCoords[2];
        int index = map[i].in_index;
        const topology::Tetrahedron& tetra = tetras[index];
        out[tetra[0]] += v * (1-fx-fy-fz);
        out[tetra[1]] += v * fx;
        out[tetra[2]] += v * fy;
        out[tetra[3]] += v * fz;
    }
}

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const sofa::helper::vector<topology::Hexahedron>& cubes = this->topology->getHexas();
    for(unsigned int i=0; i<map.size(); i++)
    {
        const typename Out::Deriv v = in[i];
        const OutReal fx = (OutReal)map[i].baryCoords[0];
        const OutReal fy = (OutReal)map[i].baryCoords[1];
        const OutReal fz = (OutReal)map[i].baryCoords[2];
        int index = map[i].in_index;
        const topology::Hexahedron& cube = cubes[index];
        out[cube[0]] += v * ((1-fx) * (1-fy) * (1-fz));
        out[cube[1]] += v * ((  fx) * (1-fy) * (1-fz));
        out[cube[3]] += v * ((1-fx) * (  fy) * (1-fz));
        out[cube[2]] += v * ((  fx) * (  fy) * (1-fz));
        out[cube[4]] += v * ((1-fx) * (1-fy) * (  fz));
        out[cube[5]] += v * ((  fx) * (1-fy) * (  fz));
        out[cube[7]] += v * ((1-fx) * (  fy) * (  fz));
        out[cube[6]] += v * ((  fx) * (  fy) * (  fz));
    }
}

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::draw()
{
    if (!this->getShow()) return;
    glDisable (GL_LIGHTING);
    glPointSize(7);
    glColor4f (1,1,0,1);
    const OutVecCoord& out = *this->toModel->getX();
    glBegin (GL_POINTS);
    for (unsigned int i=0; i<out.size(); i++)
    {
        helper::gl::glVertexT(out[i]);
    }
    glEnd();
    const InVecCoord& in = *this->fromModel->getX();
    if (mapper!=NULL) mapper->draw(out, in);
    glPointSize(1);
}



template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::getJ(unsigned int Idx, sofa::helper::vector< double > &factor, sofa::helper::vector< unsigned int > &indices)
{
    if (Idx < map1d.size())
    {
        int index = map1d[Idx].in_index;

        const sofa::core::componentmodel::topology::BaseMeshTopology::SeqLines& lines = this->topology->getLines();
        const sofa::core::componentmodel::topology::BaseMeshTopology::Line& line = lines[index];

        factor.push_back( map1d[Idx].baryCoords[0]);  indices.push_back( line[0] );
        factor.push_back( 1-map1d[Idx].baryCoords[0]); indices.push_back( line[1] );
    }
    else if (Idx < map1d.size()+map2d.size())
    {
        Idx -= map1d.size();
        int index = map2d[Idx].in_index;

        const sofa::core::componentmodel::topology::BaseMeshTopology::SeqTriangles& triangles = this->topology->getTriangles();
        const sofa::core::componentmodel::topology::BaseMeshTopology::SeqQuads& quads = this->topology->getQuads();
        const int c0 = triangles.size();
        if (index < c0)
        {
            const sofa::core::componentmodel::topology::BaseMeshTopology::Triangle& triangle = triangles[index];

            factor.push_back( 1-map2d[Idx].baryCoords[0]-map1d[Idx].baryCoords[1]);  indices.push_back( triangle[0] );
            factor.push_back( map2d[Idx].baryCoords[0]);                             indices.push_back( triangle[1] );
            factor.push_back( map2d[Idx].baryCoords[1]);                             indices.push_back( triangle[2] );
        }
        else
        {
            const sofa::core::componentmodel::topology::BaseMeshTopology::Quad& quad = quads[index-c0];

            factor.push_back( (1-map1d[Idx].baryCoords[0])*(1-map1d[Idx].baryCoords[1]));  indices.push_back( quad[0] );
            factor.push_back( (  map1d[Idx].baryCoords[0])*(1-map1d[Idx].baryCoords[1]));  indices.push_back( quad[1] );
            factor.push_back( (1-map1d[Idx].baryCoords[0])*(  map1d[Idx].baryCoords[1]));  indices.push_back( quad[2] );
            factor.push_back( (  map1d[Idx].baryCoords[0])*(  map1d[Idx].baryCoords[1]));  indices.push_back( quad[3] );
        }
    }
    else if (Idx < map1d.size()+map2d.size()+map3d.size())
    {

        Idx -= (map1d.size() + map2d.size());
        const sofa::core::componentmodel::topology::BaseMeshTopology::SeqTetras& tetras = this->topology->getTetras();
#ifdef SOFA_NEW_HEXA
        const sofa::core::componentmodel::topology::BaseMeshTopology::SeqHexas& cubes = this->topology->getHexas();
#else
        const sofa::core::componentmodel::topology::BaseMeshTopology::SeqCubes& cubes = this->topology->getCubes();
#endif

        const int c0 = tetras.size();
        int index = map3d[Idx].in_index;

        if (index < c0)
        {
            const sofa::core::componentmodel::topology::BaseMeshTopology::Tetra& tetra = tetras[index];

            factor.push_back( map2d[Idx].baryCoords[0]);                             indices.push_back( tetra[0] );
            factor.push_back( map2d[Idx].baryCoords[1]);                             indices.push_back( tetra[1] );
            factor.push_back( map2d[Idx].baryCoords[2]);                             indices.push_back( tetra[2] );
        }
        else
        {

#ifdef SOFA_NEW_HEXA
            const sofa::core::componentmodel::topology::BaseMeshTopology::Hexa& cube = cubes[index-c0];
#else
            const sofa::core::componentmodel::topology::BaseMeshTopology::Cube& cube = cubes[index-c0];
#endif

            factor.push_back( (1-map1d[Idx].baryCoords[0])*(1-map1d[Idx].baryCoords[1])*(1-map1d[Idx].baryCoords[2]));  indices.push_back( cube[0] );
            factor.push_back( (  map1d[Idx].baryCoords[0])*(1-map1d[Idx].baryCoords[1])*(1-map1d[Idx].baryCoords[2]));  indices.push_back( cube[1] );
#ifdef SOFA_NEW_HEXA
            factor.push_back( (  map1d[Idx].baryCoords[0])*(  map1d[Idx].baryCoords[1])*(1-map1d[Idx].baryCoords[2]));  indices.push_back( cube[2] );
            factor.push_back( (1-map1d[Idx].baryCoords[0])*(  map1d[Idx].baryCoords[1])*(1-map1d[Idx].baryCoords[2]));  indices.push_back( cube[3] );
#else
            factor.push_back( (1-map1d[Idx].baryCoords[0])*(  map1d[Idx].baryCoords[1])*(1-map1d[Idx].baryCoords[2]));  indices.push_back( cube[2] );
            factor.push_back( (  map1d[Idx].baryCoords[0])*(  map1d[Idx].baryCoords[1])*(1-map1d[Idx].baryCoords[2]));  indices.push_back( cube[3] );
#endif
            factor.push_back( (1-map1d[Idx].baryCoords[0])*(1-map1d[Idx].baryCoords[1])*(  map1d[Idx].baryCoords[2]));  indices.push_back( cube[4] );
            factor.push_back( (  map1d[Idx].baryCoords[0])*(1-map1d[Idx].baryCoords[1])*(  map1d[Idx].baryCoords[2]));  indices.push_back( cube[5] );
#ifdef SOFA_NEW_HEXA
            factor.push_back( (  map1d[Idx].baryCoords[0])*(  map1d[Idx].baryCoords[1])*(  map1d[Idx].baryCoords[2]));  indices.push_back( cube[6] );
            factor.push_back( (1-map1d[Idx].baryCoords[0])*(  map1d[Idx].baryCoords[1])*(  map1d[Idx].baryCoords[2]));  indices.push_back( cube[7] );
#else
            factor.push_back( (1-map1d[Idx].baryCoords[0])*(  map1d[Idx].baryCoords[1])*(  map1d[Idx].baryCoords[2]));  indices.push_back( cube[6] );
            factor.push_back( (  map1d[Idx].baryCoords[0])*(  map1d[Idx].baryCoords[1])*(  map1d[Idx].baryCoords[2]));  indices.push_back( cube[7] );
#endif
        }
    }
    else
    {
    }
}


template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::draw(const typename Out::VecCoord& out, const typename In::VecCoord& in)
{
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqLines& lines = this->topology->getLines();
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqTriangles& triangles = this->topology->getTriangles();
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqQuads& quads = this->topology->getQuads();
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqTetras& tetras = this->topology->getTetras();
#ifdef SOFA_NEW_HEXA
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqHexas& cubes = this->topology->getHexas();
#else
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqCubes& cubes = this->topology->getCubes();
#endif
    glBegin (GL_LINES);
    // 1D elements
    {
        const int i0 = 0;
        for(unsigned int i=0; i<map1d.size(); i++)
        {
            const Real fx = map1d[i].baryCoords[0];
            int index = map1d[i].in_index;
            {
                const sofa::core::componentmodel::topology::BaseMeshTopology::Line& line = lines[index];
                Real f[2];
                f[0] = (1-fx);
                f[1] = fx;
                for (int j=0; j<2; j++)
                {
                    if (f[j]<=-0.0001 || f[j]>=0.0001)
                    {
                        glColor3f((float)f[j],1,(float)f[j]);
                        helper::gl::glVertexT(out[i+i0]);
                        helper::gl::glVertexT(in[line[j]]);
                    }
                }
            }
        }
    }
    // 2D elements
    {
        const int i0 = map1d.size();
        const int c0 = triangles.size();
        for(unsigned int i=0; i<map2d.size(); i++)
        {
            const Real fx = map2d[i].baryCoords[0];
            const Real fy = map2d[i].baryCoords[1];
            int index = map2d[i].in_index;
            if (index<c0)
            {
                const sofa::core::componentmodel::topology::BaseMeshTopology::Triangle& triangle = triangles[index];
                Real f[3];
                f[0] = (1-fx-fy);
                f[1] = fx;
                f[2] = fy;
                for (int j=0; j<3; j++)
                {
                    if (f[j]<=-0.0001 || f[j]>=0.0001)
                    {
                        glColor3f((float)f[j],1,(float)f[j]);
                        helper::gl::glVertexT(out[i+i0]);
                        helper::gl::glVertexT(in[triangle[j]]);
                    }
                }
            }
            else
            {
                const sofa::core::componentmodel::topology::BaseMeshTopology::Quad& quad = quads[index-c0];
                Real f[4];
                f[0] = ((1-fx) * (1-fy));
                f[1] = ((  fx) * (1-fy));
                f[3] = ((1-fx) * (  fy));
                f[2] = ((  fx) * (  fy));
                for (int j=0; j<4; j++)
                {
                    if (f[j]<=-0.0001 || f[j]>=0.0001)
                    {
                        glColor3f((float)f[j],1,(float)f[j]);
                        helper::gl::glVertexT(out[i+i0]);
                        helper::gl::glVertexT(in[quad[j]]);
                    }
                }
            }
        }
    }
    // 3D elements
    {
        const int i0 = map1d.size()+map2d.size();
        const int c0 = tetras.size();
        for (unsigned int i=0; i<map3d.size(); i++)
        {
            const Real fx = map3d[i].baryCoords[0];
            const Real fy = map3d[i].baryCoords[1];
            const Real fz = map3d[i].baryCoords[2];
            int index = map3d[i].in_index;
            if (index<c0)
            {
                const sofa::core::componentmodel::topology::BaseMeshTopology::Tetra& tetra = tetras[index];
                Real f[4];
                f[0] = (1-fx-fy-fz);
                f[1] = fx;
                f[2] = fy;
                f[3] = fz;
                for (int j=0; j<4; j++)
                {
                    if (f[j]<=-0.0001 || f[j]>=0.0001)
                    {
                        glColor3f((float)f[j],1,(float)f[j]);
                        helper::gl::glVertexT(out[i+i0]);
                        helper::gl::glVertexT(in[tetra[j]]);
                    }
                }
            }
            else
            {
#ifdef SOFA_NEW_HEXA
                const sofa::core::componentmodel::topology::BaseMeshTopology::Hexa& cube = cubes[index-c0];
#else
                const sofa::core::componentmodel::topology::BaseMeshTopology::Cube& cube = cubes[index-c0];
#endif
                Real f[8];
                f[0] = (1-fx) * (1-fy) * (1-fz);
                f[1] = (  fx) * (1-fy) * (1-fz);
#ifdef SOFA_NEW_HEXA
                f[3] = (1-fx) * (  fy) * (1-fz);
                f[2] = (  fx) * (  fy) * (1-fz);
#else
                f[2] = (1-fx) * (  fy) * (1-fz);
                f[3] = (  fx) * (  fy) * (1-fz);
#endif
                f[4] = (1-fx) * (1-fy) * (  fz);
                f[5] = (  fx) * (1-fy) * (  fz);
#ifdef SOFA_NEW_HEXA
                f[7] = (1-fx) * (  fy) * (  fz);
                f[6] = (  fx) * (  fy) * (  fz);
#else
                f[6] = (1-fx) * (  fy) * (  fz);
                f[7] = (  fx) * (  fy) * (  fz);
#endif
                for (int j=0; j<8; j++)
                {
                    if (f[j]<=-0.0001 || f[j]>=0.0001)
                    {
                        glColor3f((float)f[j],1,1);
                        helper::gl::glVertexT(out[i+i0]);
                        helper::gl::glVertexT(in[cube[j]]);
                    }
                }
            }
        }
    }
    glEnd();
}


template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::draw(const typename Out::VecCoord& out, const typename In::VecCoord& in)
{
    const sofa::helper::vector<topology::Edge>& edges = this->topology->getEdges();

    glBegin (GL_LINES);
    {
        for(unsigned int i=0; i<map.size(); i++)
        {
            const Real fx = map[i].baryCoords[0];
            int index = map[i].in_index;
            const topology::Edge& edge = edges[index];
            {
                const Real f = Real(1.0)-fx;
                if (f<=-0.0001 || f>=0.0001)
                {
                    glColor3f((float)f,1,(float)f);
                    helper::gl::glVertexT(out[i]);
                    helper::gl::glVertexT(in[edge[0]]);
                }
            }
            {
                const Real f = fx;
                if (f<=-0.0001 || f>=0.0001)
                {
                    glColor3f((float)f,1,(float)f);
                    helper::gl::glVertexT(out[i]);
                    helper::gl::glVertexT(in[edge[1]]);
                }
            }
        }
    }
    glEnd();
}

template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::draw(const typename Out::VecCoord& out, const typename In::VecCoord& in)
{
    const sofa::helper::vector<topology::Triangle>& triangles = this->topology->getTriangles();

    glBegin (GL_LINES);
    {
        for(unsigned int i=0; i<map.size(); i++)
        {
            const Real fx = map[i].baryCoords[0];
            const Real fy = map[i].baryCoords[1];
            int index = map[i].in_index;
            const topology::Triangle& triangle = triangles[index];
            Real f[3];
            f[0] = (1-fx-fy);
            f[1] = fx;
            f[2] = fy;
            for (int j=0; j<3; j++)
            {
                if (f[j]<=-0.0001 || f[j]>=0.0001)
                {
                    glColor3f((float)f[j],1,(float)f[j]);
                    helper::gl::glVertexT(out[i]);
                    helper::gl::glVertexT(in[triangle[j]]);
                }
            }
        }
    }
    glEnd();
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::draw(const typename Out::VecCoord& out, const typename In::VecCoord& in)
{
    const sofa::helper::vector<topology::Quad>& quads = this->topology->getQuads();
    glBegin (GL_LINES);
    {
        for(unsigned int i=0; i<map.size(); i++)
        {
            const Real fx = map[i].baryCoords[0];
            const Real fy = map[i].baryCoords[1];
            int index = map[i].in_index;
            const topology::Quad& quad = quads[index];
            Real f[4];
            f[0] = ((1-fx) * (1-fy));
            f[1] = ((  fx) * (1-fy));
            f[3] = ((1-fx) * (  fy));
            f[2] = ((  fx) * (  fy));
            for (int j=0; j<4; j++)
            {
                if (f[j]<=-0.0001 || f[j]>=0.0001)
                {
                    glColor3f((float)f[j],1,(float)f[j]);
                    helper::gl::glVertexT(out[i]);
                    helper::gl::glVertexT(in[quad[j]]);
                }
            }
        }
    }
    glEnd();
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::draw(const typename Out::VecCoord& out, const typename In::VecCoord& in)
{
    const sofa::helper::vector<topology::Tetrahedron>& tetras = this->topology->getTetras();
    glBegin (GL_LINES);
    {
        for(unsigned int i=0; i<map.size(); i++)
        {
            const Real fx = map[i].baryCoords[0];
            const Real fy = map[i].baryCoords[1];
            const Real fz = map[i].baryCoords[2];
            int index = map[i].in_index;
            const topology::Tetrahedron& tetra = tetras[index];
            Real f[4];
            f[0] = (1-fx-fy-fz);
            f[1] = fx;
            f[2] = fy;
            f[3] = fz;
            for (int j=0; j<4; j++)
            {
                if (f[j]<=-0.0001 || f[j]>=0.0001)
                {
                    glColor3f((float)f[j],1,(float)f[j]);
                    helper::gl::glVertexT(out[i]);
                    helper::gl::glVertexT(in[tetra[j]]);
                }
            }
        }
    }
    glEnd();
}

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::draw(const typename Out::VecCoord& out, const typename In::VecCoord& in)
{
    const sofa::helper::vector<topology::Hexahedron>& cubes = this->topology->getHexas();
    glBegin (GL_LINES);
    {
        for(unsigned int i=0; i<map.size(); i++)
        {
            const Real fx = map[i].baryCoords[0];
            const Real fy = map[i].baryCoords[1];
            const Real fz = map[i].baryCoords[2];
            int index = map[i].in_index;
            const topology::Hexahedron& cube = cubes[index];
            Real f[8];
            f[0] = (1-fx) * (1-fy) * (1-fz);
            f[1] = (  fx) * (1-fy) * (1-fz);
            f[3] = (1-fx) * (  fy) * (1-fz);
            f[2] = (  fx) * (  fy) * (1-fz);
            f[4] = (1-fx) * (1-fy) * (  fz);
            f[5] = (  fx) * (1-fy) * (  fz);
            f[7] = (1-fx) * (  fy) * (  fz);
            f[6] = (  fx) * (  fy) * (  fz);
            for (int j=0; j<8; j++)
            {
                if (f[j]<=-0.0001 || f[j]>=0.0001)
                {
                    glColor3f((float)f[j],1,1);
                    helper::gl::glVertexT(out[i]);
                    helper::gl::glVertexT(in[cube[j]]);
                }
            }
        }
    }
    glEnd();
}

/************************************* PropagateConstraint ***********************************/


template <class BasicMapping>
void BarycentricMapping<BasicMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{
    if (mapper!=NULL)
    {
        mapper->applyJT(out, in);
    }
}


template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{
//    printf("\n applyJT() in BaricentricMapping  [MeshMapper] \n");
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqLines& lines = this->topology->getLines();
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqTriangles& triangles = this->topology->getTriangles();
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqQuads& quads = this->topology->getQuads();
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqTetras& tetras = this->topology->getTetras();
#ifdef SOFA_NEW_HEXA
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqHexas& cubes = this->topology->getHexas();
#else
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqCubes& cubes = this->topology->getCubes();
#endif
    //const int iLine = lines.size();
    const int iTri = triangles.size();
    //const int iQuad = quads.size();
    const int iTetra= tetras.size();
    //const int iCube = cubes.size();

    const int i1d = map1d.size();
    const int i2d = map2d.size();
    const int i3d = map3d.size();

    int indexIn;

    int offset = out.size();
    out.resize(offset+in.size());

    for(unsigned int i=0; i<in.size(); i++)
    {
        for(unsigned int j=0; j<in[i].size(); j++)
        {
            const typename Out::SparseDeriv cIn = in[i][j];
            indexIn = cIn.index;
            // 1D elements
            if (indexIn < i1d)
            {
                const OutReal fx = (OutReal)map1d[indexIn].baryCoords[0];
                int index = map1d[indexIn].in_index;
                {
                    const sofa::core::componentmodel::topology::BaseMeshTopology::Line& line = lines[index];
                    out[i+offset].push_back(typename In::SparseDeriv((unsigned) line[0], (typename In::Deriv) cIn.data * (1-fx)));
                    out[i+offset].push_back(typename In::SparseDeriv(line[1], (typename In::Deriv) cIn.data * fx));
                }
            }
            // 2D elements : triangle or quad
            else if (indexIn < i2d)
            {
                const OutReal fx = (OutReal)map2d[indexIn].baryCoords[0];
                const OutReal fy = (OutReal)map2d[indexIn].baryCoords[1];
                int index = map2d[indexIn].in_index;
                if (index < iTri) // triangle
                {
                    const sofa::core::componentmodel::topology::BaseMeshTopology::Triangle& triangle = triangles[index];
                    out[i+offset].push_back(typename In::SparseDeriv(triangle[0], (typename In::Deriv) cIn.data * (1-fx-fy)));
                    out[i+offset].push_back(typename In::SparseDeriv(triangle[1], (typename In::Deriv) cIn.data * fx));
                    out[i+offset].push_back(typename In::SparseDeriv(triangle[2], (typename In::Deriv) cIn.data * fy));
                }
                else // 2D element : Quad
                {
                    const sofa::core::componentmodel::topology::BaseMeshTopology::Quad& quad = quads[index - iTri];
                    out[i+offset].push_back(typename In::SparseDeriv(quad[0], (typename In::Deriv) cIn.data * ((1-fx) * (1-fy))));
                    out[i+offset].push_back(typename In::SparseDeriv(quad[1], (typename In::Deriv) cIn.data * ((  fx) * (1-fy))));
                    out[i+offset].push_back(typename In::SparseDeriv(quad[3], (typename In::Deriv) cIn.data * ((1-fx) * (  fy))));
                    out[i+offset].push_back(typename In::SparseDeriv(quad[2], (typename In::Deriv) cIn.data * ((  fx) * (  fy))));
                }
            }
            // 3D elements
            else if (indexIn < i3d)
            {
                const OutReal fx = (OutReal)map3d[indexIn].baryCoords[0];
                const OutReal fy = (OutReal)map3d[indexIn].baryCoords[1];
                const OutReal fz = (OutReal)map3d[indexIn].baryCoords[2];
                int index = map3d[indexIn].in_index;
                if (index < iTetra) // tetra
                {
                    const sofa::core::componentmodel::topology::BaseMeshTopology::Tetra& tetra = tetras[index];
                    out[i+offset].push_back(typename In::SparseDeriv(tetra[0], (typename In::Deriv) cIn.data * (1-fx-fy-fz)));
                    out[i+offset].push_back(typename In::SparseDeriv(tetra[1], (typename In::Deriv) cIn.data * fx));
                    out[i+offset].push_back(typename In::SparseDeriv(tetra[2], (typename In::Deriv) cIn.data * fy));
                    out[i+offset].push_back(typename In::SparseDeriv(tetra[3], (typename In::Deriv) cIn.data * fz));
                }
                else // cube
                {
#ifdef SOFA_NEW_HEXA
                    const sofa::core::componentmodel::topology::BaseMeshTopology::Hexa& cube = cubes[index-iTetra];
#else
                    const sofa::core::componentmodel::topology::BaseMeshTopology::Cube& cube = cubes[index-iTetra];
#endif
                    out[i+offset].push_back(typename In::SparseDeriv(cube[0], (typename In::Deriv) cIn.data * ((1-fx) * (1-fy) * (1-fz))));
                    out[i+offset].push_back(typename In::SparseDeriv(cube[1], (typename In::Deriv) cIn.data * ((  fx) * (1-fy) * (1-fz))));
#ifdef SOFA_NEW_HEXA
                    out[i+offset].push_back(typename In::SparseDeriv(cube[3], (typename In::Deriv) cIn.data * ((1-fx) * (  fy) * (1-fz))));
                    out[i+offset].push_back(typename In::SparseDeriv(cube[2], (typename In::Deriv) cIn.data * ((  fx) * (  fy) * (1-fz))));
#else
                    out[i+offset].push_back(typename In::SparseDeriv(cube[2], (typename In::Deriv) cIn.data * ((1-fx) * (  fy) * (1-fz))));
                    out[i+offset].push_back(typename In::SparseDeriv(cube[3], (typename In::Deriv) cIn.data * ((  fx) * (  fy) * (1-fz))));
#endif
                    out[i+offset].push_back(typename In::SparseDeriv(cube[4], (typename In::Deriv) cIn.data * ((1-fx) * (1-fy) * (  fz))));
                    out[i+offset].push_back(typename In::SparseDeriv(cube[5], (typename In::Deriv) cIn.data * ((  fx) * (1-fy) * (  fz))));
#ifdef SOFA_NEW_HEXA
                    out[i+offset].push_back(typename In::SparseDeriv(cube[7], (typename In::Deriv) cIn.data * ((1-fx) * (  fy) * (  fz))));
                    out[i+offset].push_back(typename In::SparseDeriv(cube[6], (typename In::Deriv) cIn.data * ((  fx) * (  fy) * (  fz))));
#else
                    out[i+offset].push_back(typename In::SparseDeriv(cube[6], (typename In::Deriv) cIn.data * ((1-fx) * (  fy) * (  fz))));
                    out[i+offset].push_back(typename In::SparseDeriv(cube[7], (typename In::Deriv) cIn.data * ((  fx) * (  fy) * (  fz))));
#endif
                }
            }
        }
    }
}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{
    int offset = out.size();
    out.resize(offset+in.size());
    const sofa::helper::vector<topology::Edge>& edges = this->topology->getEdges();

    for(unsigned int i=0; i<in.size(); i++)
    {
        for(unsigned int j=0; j<in[i].size(); j++)
        {
            const typename Out::SparseDeriv cIn = in[i][j];
            const topology::Edge edge = edges[this->map[cIn.index].in_index];
            const OutReal fx = (OutReal)map[cIn.index].baryCoords[0];

            out[i+offset].push_back(typename In::SparseDeriv(edge[0], (typename In::Deriv) (cIn.data * (1-fx))));
            out[i+offset].push_back(typename In::SparseDeriv(edge[1], (typename In::Deriv) (cIn.data * (fx))));
        }
    }
}

template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{
    int offset = out.size();
    out.resize(offset+in.size());
    const sofa::helper::vector<topology::Triangle>& triangles = this->topology->getTriangles();

    for(unsigned int i=0; i<in.size(); i++)
    {
        for(unsigned int j=0; j<in[i].size(); j++)
        {
            const typename Out::SparseDeriv cIn = in[i][j];
            const topology::Triangle triangle = triangles[this->map[cIn.index].in_index];
            const OutReal fx = (OutReal)map[cIn.index].baryCoords[0];
            const OutReal fy = (OutReal)map[cIn.index].baryCoords[1];

            out[i+offset].push_back(typename In::SparseDeriv(triangle[0], (typename In::Deriv) (cIn.data * (1-fx-fy))));
            out[i+offset].push_back(typename In::SparseDeriv(triangle[1], (typename In::Deriv) (cIn.data * (fx))));
            out[i+offset].push_back(typename In::SparseDeriv(triangle[2], (typename In::Deriv) (cIn.data * (fy))));
        }
    }
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{
    int offset = out.size();
    out.resize(offset+in.size());
    const sofa::helper::vector<topology::Quad>& quads = this->topology->getQuads();

    for(unsigned int i=0; i<in.size(); i++)
    {
        for(unsigned int j=0; j<in[i].size(); j++)
        {
            const typename Out::SparseDeriv cIn = in[i][j];
            const int indexIn = cIn.index;
            const OutReal fx = (OutReal)map[cIn.index].baryCoords[0];
            const OutReal fy = (OutReal)map[cIn.index].baryCoords[1];

            const sofa::core::componentmodel::topology::BaseMeshTopology::Quad& quad = quads[map[indexIn].in_index];
            out[i+offset].push_back(typename In::SparseDeriv(quad[0], (typename In::Deriv) cIn.data * ((1-fx) * (1-fy))));
            out[i+offset].push_back(typename In::SparseDeriv(quad[1], (typename In::Deriv) cIn.data * ((  fx) * (1-fy))));
            out[i+offset].push_back(typename In::SparseDeriv(quad[3], (typename In::Deriv) cIn.data * ((1-fx) * (  fy))));
            out[i+offset].push_back(typename In::SparseDeriv(quad[2], (typename In::Deriv) cIn.data * ((  fx) * (  fy))));
        }
    }
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{
    int offset = out.size();
    out.resize(offset+in.size());
    const sofa::helper::vector<topology::Tetrahedron>& tetras = this->topology->getTetras();

    for(unsigned int i=0; i<in.size(); i++)
    {
        for(unsigned int j=0; j<in[i].size(); j++)
        {
            const typename Out::SparseDeriv cIn = in[i][j];
            const int indexIn = cIn.index;
            const OutReal fx = (OutReal)map[indexIn].baryCoords[0];
            const OutReal fy = (OutReal)map[indexIn].baryCoords[1];
            const OutReal fz = (OutReal)map[indexIn].baryCoords[2];
            int index = map[indexIn].in_index;
            const topology::Tetrahedron& tetra = tetras[index];
            out[i+offset].push_back(typename In::SparseDeriv(tetra[0], (typename In::Deriv) cIn.data * (1-fx-fy-fz)));
            out[i+offset].push_back(typename In::SparseDeriv(tetra[1], (typename In::Deriv) cIn.data * fx));
            out[i+offset].push_back(typename In::SparseDeriv(tetra[2], (typename In::Deriv) cIn.data * fy));
            out[i+offset].push_back(typename In::SparseDeriv(tetra[3], (typename In::Deriv) cIn.data * fz));
        }
    }
}

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{
    int offset = out.size();
    out.resize(offset+in.size());
    const sofa::helper::vector<topology::Hexahedron>& cubes = this->topology->getHexas();

    for(unsigned int i=0; i<in.size(); i++)
    {
        for(unsigned int j=0; j<in[i].size(); j++)
        {
            const typename Out::SparseDeriv cIn = in[i][j];
            const int indexIn = cIn.index;
            const OutReal fx = (OutReal)map[indexIn].baryCoords[0];
            const OutReal fy = (OutReal)map[indexIn].baryCoords[1];
            const OutReal fz = (OutReal)map[indexIn].baryCoords[2];
            int index = map[indexIn].in_index;
            const topology::Hexahedron& cube = cubes[index];
            out[i+offset].push_back(typename In::SparseDeriv(cube[0], (typename In::Deriv) cIn.data * ((1-fx) * (1-fy) * (1-fz))));
            out[i+offset].push_back(typename In::SparseDeriv(cube[1], (typename In::Deriv) cIn.data * ((  fx) * (1-fy) * (1-fz))));
            out[i+offset].push_back(typename In::SparseDeriv(cube[3], (typename In::Deriv) cIn.data * ((1-fx) * (  fy) * (1-fz))));
            out[i+offset].push_back(typename In::SparseDeriv(cube[2], (typename In::Deriv) cIn.data * ((  fx) * (  fy) * (1-fz))));
            out[i+offset].push_back(typename In::SparseDeriv(cube[4], (typename In::Deriv) cIn.data * ((1-fx) * (1-fy) * (  fz))));
            out[i+offset].push_back(typename In::SparseDeriv(cube[5], (typename In::Deriv) cIn.data * ((  fx) * (1-fy) * (  fz))));
            out[i+offset].push_back(typename In::SparseDeriv(cube[7], (typename In::Deriv) cIn.data * ((1-fx) * (  fy) * (  fz))));
            out[i+offset].push_back(typename In::SparseDeriv(cube[6], (typename In::Deriv) cIn.data * ((  fx) * (  fy) * (  fz))));
        }
    }
}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::handleTopologyChange()
{
    std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itBegin = this->topology->firstChange();
    std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itEnd = this->topology->lastChange();

    for(std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator changeIt = itBegin;
        changeIt != itEnd; ++changeIt)
    {
        const core::componentmodel::topology::TopologyChangeType changeType = (*changeIt)->getChangeType();
        switch(changeType)
        {
            //TODO: implementation of BarycentricMapperEdgeSetTopology<In,Out>::handleTopologyChange()
        case core::componentmodel::topology::ENDING_EVENT:       ///< To notify the end for the current sequence of topological change events
            break;
        case core::componentmodel::topology::POINTSINDICESSWAP:  ///< For PointsIndicesSwap.
            break;
        case core::componentmodel::topology::POINTSADDED:        ///< For PointsAdded.
            break;
        case core::componentmodel::topology::POINTSREMOVED:      ///< For PointsRemoved.
            break;
        case core::componentmodel::topology::POINTSRENUMBERING:  ///< For PointsRenumbering.
            break;
        case core::componentmodel::topology::EDGESADDED:         ///< For EdgesAdded.
            break;
        case core::componentmodel::topology::EDGESREMOVED:       ///< For EdgesRemoved.
            break;
        case core::componentmodel::topology::EDGESRENUMBERING:    ///< For EdgesRenumbering.
            break;
        default:
            break;
        }
    }
}

template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::handleTopologyChange()
{
    std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itBegin = this->topology->firstChange();
    std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itEnd = this->topology->lastChange();

    for(std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator changeIt = itBegin;
        changeIt != itEnd; ++changeIt)
    {
        const core::componentmodel::topology::TopologyChangeType changeType = (*changeIt)->getChangeType();
        switch(changeType)
        {
            //TODO: implementation of BarycentricMapperTriangleSetTopology<In,Out>::handleTopologyChange()
        case core::componentmodel::topology::ENDING_EVENT:       ///< To notify the end for the current sequence of topological change events
            break;
        case core::componentmodel::topology::POINTSINDICESSWAP:  ///< For PointsIndicesSwap.
            break;
        case core::componentmodel::topology::POINTSADDED:        ///< For PointsAdded.
            break;
        case core::componentmodel::topology::POINTSREMOVED:      ///< For PointsRemoved.
            break;
        case core::componentmodel::topology::POINTSRENUMBERING:  ///< For PointsRenumbering.
            break;
        case core::componentmodel::topology::TRIANGLESADDED:     ///< For TrianglesAdded.
            break;
        case core::componentmodel::topology::TRIANGLESREMOVED:   ///< For TrianglesRemoved.
            break;
        case core::componentmodel::topology::TRIANGLESRENUMBERING: ///< For TrianglesRenumbering.
            break;
        default:
            break;
        }
    }
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::handleTopologyChange()
{
    std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itBegin = this->topology->firstChange();
    std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itEnd = this->topology->lastChange();

    for(std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator changeIt = itBegin;
        changeIt != itEnd; ++changeIt)
    {
        const core::componentmodel::topology::TopologyChangeType changeType = (*changeIt)->getChangeType();
        switch(changeType)
        {
            //TODO: implementation of BarycentricMapperQuadSetTopology<In,Out>::handleTopologyChange()
        case core::componentmodel::topology::ENDING_EVENT:       ///< To notify the end for the current sequence of topological change events
            break;
        case core::componentmodel::topology::POINTSINDICESSWAP:  ///< For PointsIndicesSwap.
            break;
        case core::componentmodel::topology::POINTSADDED:        ///< For PointsAdded.
            break;
        case core::componentmodel::topology::POINTSREMOVED:      ///< For PointsRemoved.
            break;
        case core::componentmodel::topology::POINTSRENUMBERING:  ///< For PointsRenumbering.
            break;
        case core::componentmodel::topology::QUADSADDED:     ///< For QuadsAdded.
            break;
        case core::componentmodel::topology::QUADSREMOVED:   ///< For QuadsRemoved.
            break;
        case core::componentmodel::topology::QUADSRENUMBERING: ///< For QuadsRenumbering.
            break;
        default:
            break;
        }
    }
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::handleTopologyChange()
{
    std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itBegin = this->topology->firstChange();
    std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itEnd = this->topology->lastChange();

    for(std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator changeIt = itBegin;
        changeIt != itEnd; ++changeIt)
    {
        const core::componentmodel::topology::TopologyChangeType changeType = (*changeIt)->getChangeType();
        switch(changeType)
        {
            //TODO: implementation of BarycentricMapperTetrahedronSetTopology<In,Out>::handleTopologyChange()
        case core::componentmodel::topology::ENDING_EVENT:       ///< To notify the end for the current sequence of topological change events
            break;
        case core::componentmodel::topology::POINTSINDICESSWAP:  ///< For PointsIndicesSwap.
            break;
        case core::componentmodel::topology::POINTSADDED:        ///< For PointsAdded.
            break;
        case core::componentmodel::topology::POINTSREMOVED:      ///< For PointsRemoved.
            break;
        case core::componentmodel::topology::POINTSRENUMBERING:  ///< For PointsRenumbering.
            break;
        case core::componentmodel::topology::TETRAHEDRAADDED:     ///< For TetrahedraAdded.
            break;
        case core::componentmodel::topology::TETRAHEDRAREMOVED:   ///< For TetrahedraRemoved.
            break;
        case core::componentmodel::topology::TETRAHEDRARENUMBERING: ///< For TetrahedraRenumbering.
            break;
        default:
            break;
        }
    }
}

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::handleTopologyChange()
{
    if(this->topology->firstChange() == this->topology->lastChange())
        return;

    std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itBegin = this->topology->firstChange();
    std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itEnd = this->topology->lastChange();

    for(std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator changeIt = itBegin;
        changeIt != itEnd; ++changeIt)
    {
        const core::componentmodel::topology::TopologyChangeType changeType = (*changeIt)->getChangeType();
        switch(changeType)
        {
            //TODO: implementation of BarycentricMapperHexahedronSetTopology<In,Out>::handleTopologyChange()
        case core::componentmodel::topology::ENDING_EVENT:       ///< To notify the end for the current sequence of topological change events
            break;
        case core::componentmodel::topology::POINTSINDICESSWAP:  ///< For PointsIndicesSwap.
            break;
        case core::componentmodel::topology::POINTSADDED:        ///< For PointsAdded.
            break;
        case core::componentmodel::topology::POINTSREMOVED:      ///< For PointsRemoved.
            break;
        case core::componentmodel::topology::POINTSRENUMBERING:  ///< For PointsRenumbering.
            break;
        case core::componentmodel::topology::HEXAHEDRAADDED:     ///< For HexahedraAdded.
        {
        }
        break;
        case core::componentmodel::topology::HEXAHEDRAREMOVED:   ///< For HexahedraRemoved.
        {
            const unsigned int nbHexas = this->topology->getNbHexas();

            const sofa::helper::vector<unsigned int> &tab = (static_cast< const component::topology::HexahedraRemoved *> (*changeIt))->getArray();
            sofa::helper::vector<unsigned int> hexahedra(tab);

            for(unsigned int i=0; i<hexahedra.size(); ++i)
            {
                // remove all references to the removed cubes from the mapping data
                unsigned int cubeId = hexahedra[i];
                for(unsigned int j=0; j<map.size(); ++j)
                {
                    if(map[j].in_index == (int) cubeId) // compute new mapping
                    {
                        Vector3 coefs;
                        coefs[0] = map[j].baryCoords[0];
                        coefs[1] = map[j].baryCoords[1];
                        coefs[2] = map[j].baryCoords[2];

                        typename In::Coord pos = _geomAlgo->getRestPointPositionInHexahedron(cubeId, coefs);

                        // find nearest cell and barycentric coords
                        int index=-1;

                        typename In::Real distance = 1e10;
                        for (unsigned int c=0; c<nbHexas; ++c)
                        {
                            bool validC = true;
                            // don't search in cubes that are being removed
                            for(unsigned int k=0; k<hexahedra.size(); ++k)
                            {
                                if(c == hexahedra[k])
                                {
                                    validC = false;
                                    break;
                                }
                            }

                            if(validC)

                            {
                                const Real d = _geomAlgo->computeElementRestDistanceMeasure(c, pos);

                                if (d<distance)
                                {
                                    distance = d;
                                    index = c;
                                }
                            }
                        }

                        if(index != -1)
                        {
                            const Vector3 bc = _geomAlgo->computeHexahedronRestBarycentricCoeficients(index, pos);

                            map[j].baryCoords[0] = (Real) bc[0];
                            map[j].baryCoords[1] = (Real) bc[1];
                            map[j].baryCoords[2] = (Real) bc[2];
                            map[j].in_index = index;
                        }
                    }
                }
            }

            // renumber
            unsigned int lastCubeId = nbHexas-1;
            for(unsigned int i=0; i<hexahedra.size(); ++i, --lastCubeId)
            {
                unsigned int cubeId = hexahedra[i];
                for(unsigned int j=0; j<map.size(); ++j)
                {
                    if(map[j].in_index == (int) lastCubeId)
                        map[j].in_index = cubeId;
                }
            }
        }
        break;
        case core::componentmodel::topology::HEXAHEDRARENUMBERING: ///< For HexahedraRenumbering.
            break;
        default:
            break;
        }
    }
}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::handlePointEvents(std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator itBegin,
        std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator itEnd)
{
    map.handleTopologyEvents(itBegin, itEnd);
}

template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::handlePointEvents(std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator itBegin,
        std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator itEnd)
{
    map.handleTopologyEvents(itBegin, itEnd);
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::handlePointEvents(std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator itBegin,
        std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator itEnd)
{
    map.handleTopologyEvents(itBegin, itEnd);
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::handlePointEvents(std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator itBegin,
        std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator itEnd)
{
    map.handleTopologyEvents(itBegin, itEnd);
}

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::handlePointEvents(std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator itBegin,
        std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator itEnd)
{
    map.handleTopologyEvents(itBegin, itEnd);
}

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::handleTopologyChange()
{
    BarycentricMapperDynamicTopology* topoMapper = dynamic_cast<BarycentricMapperDynamicTopology*>(mapper);

    if(topoMapper != NULL)
    {
        // handle changes in the From topology
        topoMapper->handleTopologyChange();

        // handle changes in the To topology
        if (topology_to != NULL)
        {
            if(topology_to->firstChange() != topology_to->lastChange()) // may not be necessary
            {
                const std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itBegin = topology_to->firstChange();
                const std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itEnd = topology_to->lastChange();

                topoMapper->handlePointEvents(itBegin, itEnd);
            }
        }
    }
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
