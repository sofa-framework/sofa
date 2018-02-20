/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPING_INL
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPING_INL

#include <SofaBaseMechanics/BarycentricMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/MechanicalState.h>

#include <SofaBaseTopology/RegularGridTopology.h>
#include <SofaBaseTopology/SparseGridTopology.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/QuadSetTopologyContainer.h>
#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>
#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <SofaBaseTopology/QuadSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.h>
#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TopologyData.inl>

#include <sofa/helper/vector.h>
#include <sofa/helper/system/config.h>

#include <sofa/simulation/Simulation.h>

#include <algorithm>
#include <iostream>

//#include <SofaMeshCollision/MeshIntTool.h>


namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
BarycentricMapping<TIn, TOut>::BarycentricMapping()
    : Inherit()
    , mapper(initLink("mapper","Internal mapper created depending on the type of topology"))
    , useRestPosition(core::objectmodel::Base::initData(&useRestPosition, false, "useRestPosition", "Use the rest position of the input and output models to initialize the mapping"))
#ifdef SOFA_DEV
    , sleeping(core::objectmodel::Base::initData(&sleeping, false, "sleeping", "is the mapping sleeping (not computed)"))
#endif
{
}

template <class TIn, class TOut>
BarycentricMapping<TIn, TOut>::BarycentricMapping(core::State<In>* from, core::State<Out>* to, typename Mapper::SPtr mapper)
    : Inherit ( from, to )
    , mapper(initLink("mapper","Internal mapper created depending on the type of topology"), mapper)
#ifdef SOFA_DEV
    , sleeping(core::objectmodel::Base::initData(&sleeping, false, "sleeping", "is the mapping sleeping (not computed)"))
#endif
{
    if (mapper)
        this->addSlave(mapper.get());
}

template <class TIn, class TOut>
BarycentricMapping<TIn, TOut>::BarycentricMapping (core::State<In>* from, core::State<Out>* to, BaseMeshTopology * topology )
    : Inherit ( from, to )
    , mapper (initLink("mapper","Internal mapper created depending on the type of topology"))
#ifdef SOFA_DEV
    , sleeping(core::objectmodel::Base::initData(&sleeping, false, "sleeping", "is the mapping sleeping (not computed)"))
#endif
{
    if (topology)
    {
        createMapperFromTopology ( topology );
    }
}

template <class TIn, class TOut>
BarycentricMapping<TIn, TOut>::~BarycentricMapping()
{
}

template <class In, class Out>
void BarycentricMapperRegularGridTopology<In,Out>::clear ( int reserve )
{
    updateJ = true;
    map.clear();
    if ( reserve>0 ) map.reserve ( reserve );
}

template <class In, class Out>
int BarycentricMapperRegularGridTopology<In,Out>::addPointInCube ( const int cubeIndex, const SReal* baryCoords )
{
    map.resize ( map.size() +1 );
    CubeData& data = *map.rbegin();
    data.in_index = cubeIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    data.baryCoords[2] = ( Real ) baryCoords[2];
    return (int)map.size()-1;
}

template <class In, class Out>
void BarycentricMapperRegularGridTopology<In,Out>::init ( const typename Out::VecCoord& out, const typename In::VecCoord& /*in*/ )
{
    //if ( map.size() != 0 ) return;
    updateJ = true;

    int outside = 0;
    clear ( (int)out.size() );
    if ( fromTopology->isVolume() )
    {
        for ( unsigned int i=0; i<out.size(); i++ )
        {
            sofa::defaulttype::Vector3 coefs;
            int cube = fromTopology->findCube ( sofa::defaulttype::Vector3 ( Out::getCPos(out[i]) ), coefs[0], coefs[1], coefs[2] );
            if ( cube==-1 )
            {
                ++outside;
                cube = fromTopology->findNearestCube ( sofa::defaulttype::Vector3 ( Out::getCPos(out[i]) ), coefs[0], coefs[1], coefs[2] );
            }

            this->addPointInCube ( cube, coefs.ptr() );
        }
    }
}

template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::clear ( int reserve )
{
    updateJ = true;
    map.clear();
    if ( reserve>0 ) map.reserve ( reserve );
}

template <class In, class Out>
int BarycentricMapperSparseGridTopology<In,Out>::addPointInCube ( const int cubeIndex, const SReal* baryCoords )
{
    map.resize ( map.size() +1 );
    CubeData& data = *map.rbegin();
    data.in_index = cubeIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    data.baryCoords[2] = ( Real ) baryCoords[2];
    return (int)map.size()-1;
}

template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::init ( const typename Out::VecCoord& out, const typename In::VecCoord& /*in*/ )
{
    if ( this->map.size() != 0 ) return;
    updateJ = true;
    int outside = 0;
    clear ( (int)out.size() );

    if ( fromTopology->isVolume() )
    {
        for ( unsigned int i=0; i<out.size(); i++ )
        {
            sofa::defaulttype::Vector3 coefs;
            int cube = fromTopology->findCube ( sofa::defaulttype::Vector3 ( Out::getCPos(out[i]) ), coefs[0], coefs[1], coefs[2] );
            if ( cube==-1 )
            {
                ++outside;
                cube = fromTopology->findNearestCube ( sofa::defaulttype::Vector3 ( Out::getCPos(out[i]) ), coefs[0], coefs[1], coefs[2] );
            }
            sofa::defaulttype::Vector3 baryCoords = coefs;
            this->addPointInCube ( cube, baryCoords.ptr() );
        }
    }
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::clear1d ( int reserve )
{
    updateJ = true;
    map1d.clear(); if ( reserve>0 ) map1d.reserve ( reserve );
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::clear2d ( int reserve )
{
    updateJ = true;
    map2d.clear(); if ( reserve>0 ) map2d.reserve ( reserve );
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::clear3d ( int reserve )
{
    updateJ = true;
    map3d.clear(); if ( reserve>0 ) map3d.reserve ( reserve );
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::clear ( int reserve )
{
    updateJ = true;
    map1d.clear(); if ( reserve>0 ) map1d.reserve ( reserve );
    map2d.clear(); if ( reserve>0 ) map2d.reserve ( reserve );
    map3d.clear(); if ( reserve>0 ) map3d.reserve ( reserve );
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::addPointInLine ( const int lineIndex, const SReal* baryCoords )
{
    map1d.resize ( map1d.size() +1 );
    MappingData1D& data = *map1d.rbegin();
    data.in_index = lineIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    return (int)map1d.size()-1;
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::addPointInTriangle ( const int triangleIndex, const SReal* baryCoords )
{
    map2d.resize ( map2d.size() +1 );
    MappingData2D& data = *map2d.rbegin();
    data.in_index = triangleIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    return (int)map2d.size()-1;
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::addPointInQuad ( const int quadIndex, const SReal* baryCoords )
{
    map2d.resize ( map2d.size() +1 );
    MappingData2D& data = *map2d.rbegin();
    data.in_index = quadIndex + this->fromTopology->getNbTriangles();
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    return (int)map2d.size()-1;
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::addPointInTetra ( const int tetraIndex, const SReal* baryCoords )
{
    map3d.resize ( map3d.size() +1 );
    MappingData3D& data = *map3d.rbegin();
    data.in_index = tetraIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    data.baryCoords[2] = ( Real ) baryCoords[2];
    return (int)map3d.size()-1;
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::addPointInCube ( const int cubeIndex, const SReal* baryCoords )
{
    map3d.resize ( map3d.size() +1 );
    MappingData3D& data = *map3d.rbegin();
    data.in_index = cubeIndex + this->fromTopology->getNbTetrahedra();
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    data.baryCoords[2] = ( Real ) baryCoords[2];
    return (int)map3d.size()-1;
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::createPointInLine ( const typename Out::Coord& p, int lineIndex, const typename In::VecCoord* points )
{
    SReal baryCoords[1];
    const sofa::core::topology::BaseMeshTopology::Line& elem = this->fromTopology->getLine ( lineIndex );
    const typename In::Coord p0 = ( *points ) [elem[0]];
    const typename In::Coord pA = ( *points ) [elem[1]] - p0;
    typename In::Coord pos = Out::getCPos(p) - p0;
    baryCoords[0] = ( ( pos*pA ) /pA.norm2() );
    return this->addPointInLine ( lineIndex, baryCoords );
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::createPointInTriangle ( const typename Out::Coord& p, int triangleIndex, const typename In::VecCoord* points )
{
    SReal baryCoords[2];
    const sofa::core::topology::BaseMeshTopology::Triangle& elem = this->fromTopology->getTriangle ( triangleIndex );

    const typename In::Coord & p1 = ( *points ) [elem[0]];
    const typename In::Coord & p2 = ( *points ) [elem[1]];
    const typename In::Coord & p3 = ( *points ) [elem[2]];
    const typename In::Coord & to_be_projected = Out::getCPos(p);

    const typename In::Coord AB = p2-p1;
    const typename In::Coord AC = p3-p1;
    const typename In::Coord AQ = to_be_projected -p1;
    sofa::defaulttype::Mat<2,2,typename In::Real> A;
    sofa::defaulttype::Vec<2,typename In::Real> b;
    A[0][0] = AB*AB;
    A[1][1] = AC*AC;
    A[0][1] = A[1][0] = AB*AC;
    b[0] = AQ*AB;
    b[1] = AQ*AC;
    const typename In::Real det = sofa::defaulttype::determinant(A);

    baryCoords[0] = (b[0]*A[1][1] - b[1]*A[0][1])/det;
    baryCoords[1]  = (b[1]*A[0][0] - b[0]*A[1][0])/det;

    if (baryCoords[0] < 0 || baryCoords[1] < 0 || baryCoords[0] + baryCoords[1] > 1)
    {
        // nearest point is on an edge or corner
        // barycentric coordinate on AB
        SReal pAB = b[0] / A[0][0]; // AQ*AB / AB*AB
        // barycentric coordinate on AC
        SReal pAC = b[1] / A[1][1]; // AQ*AC / AB*AB
        if (pAB < 0 && pAC < 0)
        {
            // closest point is A
            baryCoords[0] = 0.0;
            baryCoords[1] = 0.0;
        }
        else if (pAB < 1 && baryCoords[1] < 0)
        {
            // closest point is on AB
            baryCoords[0] = pAB;
            baryCoords[1] = 0.0;
        }
        else if (pAC < 1 && baryCoords[0] < 0)
        {
            // closest point is on AC
            baryCoords[0] = 0.0;
            baryCoords[1] = pAC;
        }
        else
        {
            // barycentric coordinate on BC
            // BQ*BC / BC*BC = (AQ-AB)*(AC-AB) / (AC-AB)*(AC-AB) = (AQ*AC-AQ*AB + AB*AB-AB*AC) / (AB*AB+AC*AC-2AB*AC)
            SReal pBC = (b[1] - b[0] + A[0][0] - A[0][1]) / (A[0][0] + A[1][1] - 2*A[0][1]); // BQ*BC / BC*BC
            if (pBC < 0)
            {
                // closest point is B
                baryCoords[0] = 1.0;
                baryCoords[1] = 0.0;
            }
            else if (pBC > 1)
            {
                // closest point is C
                baryCoords[0] = 0.0;
                baryCoords[1] = 1.0;
            }
            else
            {
                // closest point is on BC
                baryCoords[0] = 1.0-pBC;
                baryCoords[1] = pBC;
            }
        }
    }

    return this->addPointInTriangle ( triangleIndex, baryCoords );
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::createPointInQuad ( const typename Out::Coord& p, int quadIndex, const typename In::VecCoord* points )
{
    SReal baryCoords[2];
    const sofa::core::topology::BaseMeshTopology::Quad& elem = this->fromTopology->getQuad ( quadIndex );
    const typename In::Coord p0 = ( *points ) [elem[0]];
    const typename In::Coord pA = ( *points ) [elem[1]] - p0;
    const typename In::Coord pB = ( *points ) [elem[3]] - p0;
    typename In::Coord pos = Out::getCPos(p) - p0;
    sofa::defaulttype::Mat<3,3,typename In::Real> m,mt,base;
    m[0] = pA;
    m[1] = pB;
    m[2] = cross ( pA, pB );
    mt.transpose ( m );
    base.invert ( mt );
    const typename In::Coord base0 = base[0];
    const typename In::Coord base1 = base[1];
    baryCoords[0] = base0 * pos;
    baryCoords[1] = base1 * pos;
    return this->addPointInQuad ( quadIndex, baryCoords );
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::init ( const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    int outside = 0;
    updateJ = true;

    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = this->fromTopology->getTetrahedra();
#ifdef SOFA_NEW_HEXA
    const sofa::core::topology::BaseMeshTopology::SeqHexahedra& cubes = this->fromTopology->getHexahedra();
#else
    const sofa::core::topology::BaseMeshTopology::SeqCubes& cubes = this->fromTopology->getCubes();
#endif
    const sofa::core::topology::BaseMeshTopology::SeqTriangles& triangles = this->fromTopology->getTriangles();
    const sofa::core::topology::BaseMeshTopology::SeqQuads& quads = this->fromTopology->getQuads();
    sofa::helper::vector<sofa::defaulttype::Matrix3> bases;
    sofa::helper::vector<sofa::defaulttype::Vector3> centers;
    if ( tetrahedra.empty() && cubes.empty() )
    {
        if ( triangles.empty() && quads.empty() )
        {
            //no 3D elements, nor 2D elements -> map on 1D elements

            const sofa::core::topology::BaseMeshTopology::SeqEdges& edges = this->fromTopology->getEdges();
            if ( edges.empty() ) return;

            clear1d ( (int)out.size() );

            sofa::helper::vector< SReal >   lengthEdges;
            sofa::helper::vector< sofa::defaulttype::Vector3 > unitaryVectors;

            unsigned int e;
            for ( e=0; e<edges.size(); e++ )
            {
                lengthEdges.push_back ( ( in[edges[e][1]]-in[edges[e][0]] ).norm() );

                sofa::defaulttype::Vector3 V12 = ( in[edges[e][1]]-in[edges[e][0]] ); V12.normalize();
                unitaryVectors.push_back ( V12 );
            }

            for ( unsigned int i=0; i<out.size(); i++ )
            {
                SReal coef=0;
                for ( e=0; e<edges.size(); e++ )
                {
                    SReal lengthEdge = lengthEdges[e];
                    sofa::defaulttype::Vector3 V12 =unitaryVectors[e];

                    coef = ( V12 ) * sofa::defaulttype::Vector3 ( Out::getCPos(out[i])-in[edges[e][0]] ) /lengthEdge;
                    if ( coef >= 0 && coef <= 1 ) {addPointInLine ( e,&coef );  break; }

                }
                //If no good coefficient has been found, we add to the last element
                if ( e == edges.size() ) addPointInLine ( (int)edges.size()-1,&coef );

            }
        }
        else
        {
            // no 3D elements -> map on 2D elements
            clear2d ( (int)out.size() ); // reserve space for 2D mapping
            size_t c0 = triangles.size();
            bases.resize ( triangles.size() +quads.size() );
            centers.resize ( triangles.size() +quads.size() );
            for ( unsigned int t = 0; t < triangles.size(); t++ )
            {
                sofa::defaulttype::Mat3x3d m,mt;
                m[0] = in[triangles[t][1]]-in[triangles[t][0]];
                m[1] = in[triangles[t][2]]-in[triangles[t][0]];
                m[2] = cross ( m[0],m[1] );
                mt.transpose ( m );
                bases[t].invert ( mt );
                centers[t] = ( in[triangles[t][0]]+in[triangles[t][1]]+in[triangles[t][2]] ) /3;
            }
            for ( unsigned int c = 0; c < quads.size(); c++ )
            {
                sofa::defaulttype::Mat3x3d m,mt;
                m[0] = in[quads[c][1]]-in[quads[c][0]];
                m[1] = in[quads[c][3]]-in[quads[c][0]];
                m[2] = cross ( m[0],m[1] );
                mt.transpose ( m );
                bases[c0+c].invert ( mt );
                centers[c0+c] = ( in[quads[c][0]]+in[quads[c][1]]+in[quads[c][2]]+in[quads[c][3]] ) *0.25;
            }
            for ( unsigned int i=0; i<out.size(); i++ )
            {
                sofa::defaulttype::Vector3 pos = Out::getCPos(out[i]);
                sofa::defaulttype::Vector3 coefs;
                int index = -1;
                double distance = 1e10;
                for ( unsigned int t = 0; t < triangles.size(); t++ )
                {
                    sofa::defaulttype::Vec3d v = bases[t] * ( pos - in[triangles[t][0]] );
                    double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( ( v[2]<0?-v[2]:v[2] )-0.01,v[0]+v[1]-1 ) );
                    if ( d>0 ) d = ( pos-centers[t] ).norm2();
                    if ( d<distance ) { coefs = v; distance = d; index = t; }
                }
                for ( unsigned int c = 0; c < quads.size(); c++ )
                {
                    sofa::defaulttype::Vec3d v = bases[c0+c] * ( pos - in[quads[c][0]] );
                    double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( v[1]-1,v[0]-1 ),std::max ( v[2]-0.01,-v[2]-0.01 ) ) );
                    if ( d>0 ) d = ( pos-centers[c0+c] ).norm2();
                    if ( d<distance ) { coefs = v; distance = d; index = c0+c; }
                }
                if ( distance>0 )
                {
                    ++outside;
                }
                if ( index < (int)c0 )
                    addPointInTriangle ( index, coefs.ptr() );
                else
                    addPointInQuad ( index-c0, coefs.ptr() );
            }
        }
    }
    else
    {
        clear3d ( out.size() ); // reserve space for 3D mapping
        int c0 = tetrahedra.size();
        bases.resize ( tetrahedra.size() +cubes.size() );
        centers.resize ( tetrahedra.size() +cubes.size() );
        for ( unsigned int t = 0; t < tetrahedra.size(); t++ )
        {
            sofa::defaulttype::Mat3x3d m,mt;
            m[0] = in[tetrahedra[t][1]]-in[tetrahedra[t][0]];
            m[1] = in[tetrahedra[t][2]]-in[tetrahedra[t][0]];
            m[2] = in[tetrahedra[t][3]]-in[tetrahedra[t][0]];
            mt.transpose ( m );
            bases[t].invert ( mt );
            centers[t] = ( in[tetrahedra[t][0]]+in[tetrahedra[t][1]]+in[tetrahedra[t][2]]+in[tetrahedra[t][3]] ) *0.25;
        }
        for ( unsigned int c = 0; c < cubes.size(); c++ )
        {
            sofa::defaulttype::Mat3x3d m,mt;
            m[0] = in[cubes[c][1]]-in[cubes[c][0]];
#ifdef SOFA_NEW_HEXA
            m[1] = in[cubes[c][3]]-in[cubes[c][0]];
#else
            m[1] = in[cubes[c][2]]-in[cubes[c][0]];
#endif
            m[2] = in[cubes[c][4]]-in[cubes[c][0]];
            mt.transpose ( m );
            bases[c0+c].invert ( mt );
            centers[c0+c] = ( in[cubes[c][0]]+in[cubes[c][1]]+in[cubes[c][2]]+in[cubes[c][3]]+in[cubes[c][4]]+in[cubes[c][5]]+in[cubes[c][6]]+in[cubes[c][7]] ) *0.125;
        }
        for ( unsigned int i=0; i<out.size(); i++ )
        {
            sofa::defaulttype::Vector3 pos = Out::getCPos(out[i]);
            sofa::defaulttype::Vector3 coefs;
            int index = -1;
            double distance = 1e10;
            for ( unsigned int t = 0; t < tetrahedra.size(); t++ )
            {
                sofa::defaulttype::Vector3 v = bases[t] * ( pos - in[tetrahedra[t][0]] );
                double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( -v[2],v[0]+v[1]+v[2]-1 ) );
                if ( d>0 ) d = ( pos-centers[t] ).norm2();
                if ( d<distance ) { coefs = v; distance = d; index = t; }
            }
            for ( unsigned int c = 0; c < cubes.size(); c++ )
            {
                sofa::defaulttype::Vector3 v = bases[c0+c] * ( pos - in[cubes[c][0]] );
                double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( -v[2],v[0]-1 ),std::max ( v[1]-1,v[2]-1 ) ) );
                if ( d>0 ) d = ( pos-centers[c0+c] ).norm2();
                if ( d<distance ) { coefs = v; distance = d; index = c0+c; }
            }
            if ( distance>0 )
            {
                ++outside;
            }
            if ( index < c0 )
                addPointInTetra ( index, coefs.ptr() );
            else
                addPointInCube ( index-c0, coefs.ptr() );
        }
    }
}





template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::clear ( int reserve )
{
    helper::vector<MappingData>& vectorData = *(map.beginEdit());
    vectorData.clear(); if ( reserve>0 ) vectorData.reserve ( reserve );
    map.endEdit();
}


template <class In, class Out>
int BarycentricMapperEdgeSetTopology<In,Out>::addPointInLine ( const int edgeIndex, const SReal* baryCoords )
{
    helper::vector<MappingData>& vectorData = *(map.beginEdit());
    vectorData.resize ( map.getValue().size() +1 );
    map.endEdit();
    MappingData& data = *vectorData.rbegin();
    data.in_index = edgeIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    return (int)map.getValue().size()-1;
}

template <class In, class Out>
int BarycentricMapperEdgeSetTopology<In,Out>::createPointInLine ( const typename Out::Coord& p, int edgeIndex, const typename In::VecCoord* points )
{
    SReal baryCoords[1];
    const core::topology::BaseMeshTopology::Edge& elem = this->fromTopology->getEdge ( edgeIndex );
    const typename In::Coord p0 = ( *points ) [elem[0]];
    const typename In::Coord pA = ( *points ) [elem[1]] - p0;
    typename In::Coord pos = Out::getCPos(p) - p0;
    baryCoords[0] = dot ( pA,pos ) /dot ( pA,pA );
    return this->addPointInLine ( edgeIndex, baryCoords );
}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::init ( const typename Out::VecCoord& /*out*/, const typename In::VecCoord& /*in*/ )
{
    _fromContainer->getContext()->get ( _fromGeomAlgo );
    // Why do we need that ? is reset the map in case of topology change
//    if (this->toTopology)
//    {
//        map.createTopologicalEngine(this->toTopology);
//        map.registerTopologicalData();
//    }

    //  int outside = 0;
    //  const sofa::helper::vector<topology::Edge>& edges = this->fromTopology->getEdges();
    //TODO: implementation of BarycentricMapperEdgeSetTopology::init
}

template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::clear ( int reserve )
{
    helper::vector<MappingData>& vectorData = *(map.beginEdit());
    vectorData.clear(); if ( reserve>0 ) vectorData.reserve ( reserve );
    map.endEdit();
}

template <class In, class Out>
int BarycentricMapperTriangleSetTopology<In,Out>::addPointInTriangle ( const int triangleIndex, const SReal* baryCoords )
{
    helper::vector<MappingData>& vectorData = *(map.beginEdit());
    vectorData.resize ( map.getValue().size() +1 );
    MappingData& data = *vectorData.rbegin();
    map.endEdit();
    data.in_index = triangleIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    return (int)map.getValue().size()-1;
}

template <class In, class Out>
int BarycentricMapperTriangleSetTopology<In,Out>::createPointInTriangle ( const typename Out::Coord& p, int triangleIndex, const typename In::VecCoord* points )
{
    SReal baryCoords[2];
    const core::topology::BaseMeshTopology::Triangle& elem = this->fromTopology->getTriangle ( triangleIndex );
    const typename In::Coord p0 = ( *points ) [elem[0]];
    const typename In::Coord pA = ( *points ) [elem[1]] - p0;
    const typename In::Coord pB = ( *points ) [elem[2]] - p0;
    typename In::Coord pos = Out::getCPos(p) - p0;
    // First project to plane
    typename In::Coord normal = cross ( pA, pB );
    Real norm2 = normal.norm2();
    pos -= normal* ( ( pos*normal ) /norm2 );
    baryCoords[0] = ( Real ) sqrt ( cross ( pB, pos ).norm2() / norm2 );
    baryCoords[1] = ( Real ) sqrt ( cross ( pA, pos ).norm2() / norm2 );
    return this->addPointInTriangle ( triangleIndex, baryCoords );
}

template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::init ( const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    _fromContainer->getContext()->get ( _fromGeomAlgo );

    // Why do we need that ? it reset the map in case of topology change
//    if (this->toTopology)
//    {
//        map.createTopologicalEngine(this->toTopology);
//        map.registerTopologicalData();
//    }

    int outside = 0;

    const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle>& triangles = this->fromTopology->getTriangles();
    sofa::helper::vector<sofa::defaulttype::Mat3x3d> bases;
    sofa::helper::vector<sofa::defaulttype::Vector3> centers;

    // no 3D elements -> map on 2D elements
    clear ( (int)out.size() ); // reserve space for 2D mapping
    bases.resize ( triangles.size() );
    centers.resize ( triangles.size() );

    for ( unsigned int t = 0; t < triangles.size(); t++ )
    {
        sofa::defaulttype::Mat3x3d m,mt;
        m[0] = in[triangles[t][1]]-in[triangles[t][0]];
        m[1] = in[triangles[t][2]]-in[triangles[t][0]];
        m[2] = cross ( m[0],m[1] );
        mt.transpose ( m );
        bases[t].invert ( mt );
        centers[t] = ( in[triangles[t][0]]+in[triangles[t][1]]+in[triangles[t][2]] ) /3;
    }

    for ( unsigned int i=0; i<out.size(); i++ )
    {
        sofa::defaulttype::Vec3d pos = Out::getCPos(out[i]);
        sofa::defaulttype::Vector3 coefs;
        int index = -1;
        double distance = 1e10;
        for ( unsigned int t = 0; t < triangles.size(); t++ )
        {
            sofa::defaulttype::Vec3d v = bases[t] * ( pos - in[triangles[t][0]] );
            double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( ( v[2]<0?-v[2]:v[2] )-0.01,v[0]+v[1]-1 ) );
            if ( d>0 ) d = ( pos-centers[t] ).norm2();
            if ( d<distance ) { coefs = v; distance = d; index = t; }
        }
        if ( distance>0 )
        {
            ++outside;
        }
        addPointInTriangle ( index, coefs.ptr() );
    }
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::clear ( int reserve )
{
    helper::vector<MappingData>& vectorData = *(map.beginEdit());
    vectorData.clear(); if ( reserve>0 ) vectorData.reserve ( reserve );
    map.beginEdit();
}

template <class In, class Out>
int BarycentricMapperQuadSetTopology<In,Out>::addPointInQuad ( const int quadIndex, const SReal* baryCoords )
{
    helper::vector<MappingData>& vectorData = *(map.beginEdit());
    vectorData.resize ( map.getValue().size() +1 );
    MappingData& data = *vectorData.rbegin();
    map.endEdit();
    data.in_index = quadIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    return (int)map.getValue().size()-1;
}

template <class In, class Out>
int BarycentricMapperQuadSetTopology<In,Out>::createPointInQuad ( const typename Out::Coord& p, int quadIndex, const typename In::VecCoord* points )
{
    SReal baryCoords[2];
    const core::topology::BaseMeshTopology::Quad& elem = this->fromTopology->getQuad ( quadIndex );
    const typename In::Coord p0 = ( *points ) [elem[0]];
    const typename In::Coord pA = ( *points ) [elem[1]] - p0;
    const typename In::Coord pB = ( *points ) [elem[3]] - p0;
    typename In::Coord pos = Out::getCPos(p) - p0;
    sofa::defaulttype::Mat<3,3,typename In::Real> m,mt,base;
    m[0] = pA;
    m[1] = pB;
    m[2] = cross ( pA, pB );
    mt.transpose ( m );
    base.invert ( mt );
    const typename In::Coord base0 = base[0];
    const typename In::Coord base1 = base[1];
    baryCoords[0] = base0 * pos;
    baryCoords[1] = base1 * pos;
    return this->addPointInQuad ( quadIndex, baryCoords );
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::init ( const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    _fromContainer->getContext()->get ( _fromGeomAlgo );
    // Why do we need that ? is reset the map in case of topology change
//    if (this->toTopology)
//    {
//        map.createTopologicalEngine(this->toTopology);
//        map.registerTopologicalData();
//    }

    int outside = 0;
    const sofa::helper::vector<core::topology::BaseMeshTopology::Quad>& quads = this->fromTopology->getQuads();

    sofa::helper::vector< sofa::defaulttype::Matrix3> bases;
    sofa::helper::vector< sofa::defaulttype::Vector3> centers;

    clear ( (int)out.size() );
    bases.resize ( quads.size() );
    centers.resize ( quads.size() );

    for ( unsigned int c = 0; c < quads.size(); c++ )
    {
        sofa::defaulttype::Mat3x3d m,mt;
        m[0] = in[quads[c][1]]-in[quads[c][0]];
        m[1] = in[quads[c][3]]-in[quads[c][0]];
        m[2] = cross ( m[0],m[1] );
        mt.transpose ( m );
        bases[c].invert ( mt );
        centers[c] = ( in[quads[c][0]]+in[quads[c][1]]+in[quads[c][2]]+in[quads[c][3]] ) *0.25;
    }

    for ( unsigned int i=0; i<out.size(); i++ )
    {
        sofa::defaulttype::Vec3d pos = Out::getCPos(out[i]);
        sofa::defaulttype::Vector3 coefs;
        int index = -1;
        double distance = 1e10;
        for ( unsigned int c = 0; c < quads.size(); c++ )
        {
            sofa::defaulttype::Vec3d v = bases[c] * ( pos - in[quads[c][0]] );
            double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( v[1]-1,v[0]-1 ),std::max ( v[2]-0.01,-v[2]-0.01 ) ) );
            if ( d>0 ) d = ( pos-centers[c] ).norm2();
            if ( d<distance ) { coefs = v; distance = d; index = c; }
        }
        if ( distance>0 )
        {
            ++outside;
        }
        addPointInQuad ( index, coefs.ptr() );
    }
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::clear ( int reserve )
{
    helper::vector<MappingData>& vectorData = *(map.beginEdit());
    vectorData.clear(); if ( reserve>0 ) vectorData.reserve ( reserve );
    map.endEdit();
}

template <class In, class Out>
int BarycentricMapperTetrahedronSetTopology<In,Out>::addPointInTetra ( const int tetraIndex, const SReal* baryCoords )
{
    helper::vector<MappingData>& vectorData = *(map.beginEdit());
    vectorData.resize ( map.getValue().size() +1 );
    MappingData& data = *vectorData.rbegin();
    map.endEdit();
    data.in_index = tetraIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    data.baryCoords[2] = ( Real ) baryCoords[2];
    return (int)map.getValue().size()-1;
}

//template <class In, class Out>
//int BarycentricMapperTetrahedronSetTopology<In,Out>::createPointInTetra(const typename Out::Coord& p, int index, const typename In::VecCoord* points)
//{
//  //TODO: add implementation
//}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::init ( const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    _fromContainer->getContext()->get ( _fromGeomAlgo );
    // Why do we need that ? is reset the map in case of topology change
//    if (this->toTopology)
//    {
//        map.createTopologicalEngine(this->toTopology);
//        map.registerTopologicalData();
//    }

    int outside = 0;
    const sofa::helper::vector<core::topology::BaseMeshTopology::Tetrahedron>& tetrahedra = this->fromTopology->getTetrahedra();

    sofa::helper::vector< sofa::defaulttype::Matrix3> bases;
    sofa::helper::vector< sofa::defaulttype::Vector3> centers;

    clear ( (int)out.size() );
    bases.resize ( tetrahedra.size() );
    centers.resize ( tetrahedra.size() );
    for ( unsigned int t = 0; t < tetrahedra.size(); t++ )
    {
        sofa::defaulttype::Mat3x3d m,mt;
        m[0] = in[tetrahedra[t][1]]-in[tetrahedra[t][0]];
        m[1] = in[tetrahedra[t][2]]-in[tetrahedra[t][0]];
        m[2] = in[tetrahedra[t][3]]-in[tetrahedra[t][0]];
        mt.transpose ( m );
        bases[t].invert ( mt );
        centers[t] = ( in[tetrahedra[t][0]]+in[tetrahedra[t][1]]+in[tetrahedra[t][2]]+in[tetrahedra[t][3]] ) *0.25;
    }

    for ( unsigned int i=0; i<out.size(); i++ )
    {
        sofa::defaulttype::Vec3d pos = Out::getCPos(out[i]);
        sofa::defaulttype::Vector3 coefs;
        int index = -1;
        double distance = 1e10;
        for ( unsigned int t = 0; t < tetrahedra.size(); t++ )
        {
            sofa::defaulttype::Vec3d v = bases[t] * ( pos - in[tetrahedra[t][0]] );
            double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( -v[2],v[0]+v[1]+v[2]-1 ) );
            if ( d>0 ) d = ( pos-centers[t] ).norm2();
            if ( d<distance ) { coefs = v; distance = d; index = t; }
        }
        if ( distance>0 )
        {
            ++outside;
        }
        addPointInTetra ( index, coefs.ptr() );
    }
}





template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::clear ( int reserve )
{
    helper::vector<MappingData>& vectorData = *(map.beginEdit());
    vectorData.clear();
    if ( reserve>0 ) vectorData.reserve ( reserve );
    map.endEdit();
}

template <class In, class Out>
int BarycentricMapperHexahedronSetTopology<In,Out>::addPointInCube ( const int cubeIndex, const SReal* baryCoords )
{
    helper::vector<MappingData>& vectorData = *(map.beginEdit());
    vectorData.resize ( map.getValue().size() +1 );
    MappingData& data = *vectorData.rbegin();
    map.endEdit();
    data.in_index = cubeIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    data.baryCoords[2] = ( Real ) baryCoords[2];
    return (int)map.getValue().size()-1;
}

template <class In, class Out>
int BarycentricMapperHexahedronSetTopology<In,Out>::setPointInCube ( const int pointIndex,
        const int cubeIndex,
        const SReal* baryCoords )
{
    if ( pointIndex >= ( int ) map.getValue().size() )
        return -1;

    helper::vector<MappingData>& vectorData = *(map.beginEdit());
    MappingData& data = vectorData[pointIndex];
    data.in_index = cubeIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    data.baryCoords[2] = ( Real ) baryCoords[2];
    map.endEdit();

    if(cubeIndex == -1)
        _invalidIndex.insert(pointIndex);
    else
        _invalidIndex.erase(pointIndex);

    return pointIndex;
}

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::init ( const typename Out::VecCoord& out,
        const typename In::VecCoord& /*in*/ )
{
    _fromContainer->getContext()->get ( _fromGeomAlgo );

    if ( _fromGeomAlgo == NULL )
    {
        msg_error() << "Cannot find GeometryAlgorithms component at init." ;
    }

    if ( !map.getValue().empty() )
        return;


    clear ( (int)out.size() );

    typename In::VecCoord coord;
    helper::vector<int>   elements ( out.size() );
    helper::vector<sofa::defaulttype::Vector3> coefs ( out.size() );
    helper::vector<Real>  distances ( out.size() );

    coord.resize ( out.size() );
    for ( unsigned int i=0; i<out.size(); ++i ) coord[i] = Out::getCPos(out[i]);

    _fromGeomAlgo->findNearestElementsInRestPos ( coord, elements, coefs, distances );

    for ( unsigned int i=0; i<elements.size(); ++i )
    {
        if ( elements[i] != -1 )
            addPointInCube ( elements[i], coefs[i].ptr() );
        else
            msg_error() << "Cannot find a cell for barycentric mapping at init." ;
    }
}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::createMapperFromTopology ( BaseMeshTopology * topology )
{
    using sofa::core::behavior::BaseMechanicalState;

    mapper = NULL;

    topology::PointSetTopologyContainer* toTopoCont;
    this->toModel->getContext()->get(toTopoCont);

    core::topology::TopologyContainer* fromTopoCont = 0;

    if (dynamic_cast< core::topology::TopologyContainer* >(topology) != 0)
    {
        fromTopoCont = dynamic_cast< core::topology::TopologyContainer* >(topology);
    }
    else if (topology == 0)
    {
        this->fromModel->getContext()->get(fromTopoCont);
    }


    if (fromTopoCont != NULL)
    {
        topology::HexahedronSetTopologyContainer* t1 = dynamic_cast< topology::HexahedronSetTopologyContainer* >(fromTopoCont);
        if (t1 != NULL)
        {
            typedef BarycentricMapperHexahedronSetTopology<InDataTypes, OutDataTypes> HexahedronSetMapper;
            mapper = sofa::core::objectmodel::New<HexahedronSetMapper>(t1, toTopoCont);
        }
        else
        {
            topology::TetrahedronSetTopologyContainer* t2 = dynamic_cast<topology::TetrahedronSetTopologyContainer*>(fromTopoCont);
            if (t2 != NULL)
            {
                typedef BarycentricMapperTetrahedronSetTopology<InDataTypes, OutDataTypes> TetrahedronSetMapper;
                mapper = sofa::core::objectmodel::New<TetrahedronSetMapper>(t2, toTopoCont);
            }
            else
            {
                topology::QuadSetTopologyContainer* t3 = dynamic_cast<topology::QuadSetTopologyContainer*>(fromTopoCont);
                if (t3 != NULL)
                {
                    typedef BarycentricMapperQuadSetTopology<InDataTypes, OutDataTypes> QuadSetMapper;
                    mapper = sofa::core::objectmodel::New<QuadSetMapper>(t3, toTopoCont);
                }
                else
                {
                    topology::TriangleSetTopologyContainer* t4 = dynamic_cast<topology::TriangleSetTopologyContainer*>(fromTopoCont);
                    if (t4 != NULL)
                    {
                        typedef BarycentricMapperTriangleSetTopology<InDataTypes, OutDataTypes> TriangleSetMapper;
                        mapper = sofa::core::objectmodel::New<TriangleSetMapper>(t4, toTopoCont);
                    }
                    else
                    {
                        topology::EdgeSetTopologyContainer* t5 = dynamic_cast<topology::EdgeSetTopologyContainer*>(fromTopoCont);
                        if ( t5 != NULL )
                        {
                            typedef BarycentricMapperEdgeSetTopology<InDataTypes, OutDataTypes> EdgeSetMapper;
                            mapper = sofa::core::objectmodel::New<EdgeSetMapper>(t5, toTopoCont);
                        }
                    }
                }
            }
        }
    }
    else
    {
        using sofa::component::topology::RegularGridTopology;

        RegularGridTopology* rgt = dynamic_cast< RegularGridTopology* >(topology);

        if (rgt != NULL && rgt->isVolume())
        {
            typedef BarycentricMapperRegularGridTopology< InDataTypes, OutDataTypes > RegularGridMapper;

            mapper = sofa::core::objectmodel::New<RegularGridMapper>(rgt, toTopoCont);
        }
        else
        {
            using sofa::component::topology::SparseGridTopology;

            SparseGridTopology* sgt = dynamic_cast< SparseGridTopology* >(topology);
            if (sgt != NULL && sgt->isVolume())
            {
                typedef BarycentricMapperSparseGridTopology< InDataTypes, OutDataTypes > SparseGridMapper;
                mapper = sofa::core::objectmodel::New<SparseGridMapper>(sgt, toTopoCont);
            }
            else // generic MeshTopology
            {
                using sofa::core::topology::BaseMeshTopology;

                typedef BarycentricMapperMeshTopology< InDataTypes, OutDataTypes > MeshMapper;
                BaseMeshTopology* bmt = dynamic_cast< BaseMeshTopology* >(topology);
                mapper = sofa::core::objectmodel::New<MeshMapper>(bmt, toTopoCont);
            }
        }
    }
    if (mapper)
    {
        mapper->setName("mapper");
        this->addSlave(mapper.get());
        mapper->maskFrom = this->maskFrom;
        mapper->maskTo = this->maskTo;
    }
}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::init()
{
    topology_from = this->fromModel->getContext()->getMeshTopology();
    topology_to = this->toModel->getContext()->getMeshTopology();

    //IPB
    //core::objectmodel::BaseContext* context = this->fromModel->getContext();
    //->get(tetForceField);
    //serr << "!!!!!!!!!!!! getDT = " <<  this->fromModel->getContext()->getDt() << sendl;
    //IPE

    Inherit::init();

    if ( mapper == NULL ) // try to create a mapper according to the topology of the In model
    {
        if ( topology_from!=NULL )
        {
            createMapperFromTopology ( topology_from );
        }
    }

    if ( mapper != NULL )
    {
        if (useRestPosition.getValue())
            mapper->init ( ((const core::State<Out> *)this->toModel)->read(core::ConstVecCoordId::restPosition())->getValue(), ((const core::State<In> *)this->fromModel)->read(core::ConstVecCoordId::restPosition())->getValue() );
        else
            mapper->init (((const core::State<Out> *)this->toModel)->read(core::ConstVecCoordId::position())->getValue(), ((const core::State<In> *)this->fromModel)->read(core::ConstVecCoordId::position())->getValue() );
    }
    else
    {
        serr << "ERROR: Barycentric mapping does not understand topology."<<sendl;
    }

}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::reinit()
{
    if ( mapper != NULL )
    {
        mapper->clear();
        mapper->init (((const core::State<Out> *)this->toModel)->read(core::ConstVecCoordId::position())->getValue(), ((const core::State<In> *)this->fromModel)->read(core::ConstVecCoordId::position())->getValue() );
    }
}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/, Data< typename Out::VecCoord >& out, const Data< typename In::VecCoord >& in)
{
    if (
#ifdef SOFA_DEV
        sleeping.getValue()==false &&
#endif
        mapper != NULL)
    {
        mapper->resize( this->toModel );
        mapper->apply(*out.beginWriteOnly(), in.getValue());
        out.endEdit();
    }
}


template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::resize( core::State<Out>* toModel )
{
    toModel->resize(map1d.size() +map2d.size() +map3d.size());
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize( map1d.size() +map2d.size() +map3d.size() );

    const sofa::core::topology::BaseMeshTopology::SeqLines& lines = this->fromTopology->getLines();
    const sofa::core::topology::BaseMeshTopology::SeqTriangles& triangles = this->fromTopology->getTriangles();
    const sofa::core::topology::BaseMeshTopology::SeqQuads& quads = this->fromTopology->getQuads();
    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = this->fromTopology->getTetrahedra();
#ifdef SOFA_NEW_HEXA
    const sofa::core::topology::BaseMeshTopology::SeqHexahedra& cubes = this->fromTopology->getHexahedra();
#else
    const sofa::core::topology::BaseMeshTopology::SeqCubes& cubes = this->fromTopology->getCubes();
#endif
    // 1D elements
    {
        for ( unsigned int i=0; i<map1d.size(); i++ )
        {
            const Real fx = map1d[i].baryCoords[0];
            int index = map1d[i].in_index;
            {
                const sofa::core::topology::BaseMeshTopology::Line& line = lines[index];
                Out::setCPos(out[i] , in[line[0]] * ( 1-fx )
                        + in[line[1]] * fx );
            }
        }
    }
    // 2D elements
    {
        const int i0 = map1d.size();
        const int c0 = triangles.size();
        for ( unsigned int i=0; i<map2d.size(); i++ )
        {
            const Real fx = map2d[i].baryCoords[0];
            const Real fy = map2d[i].baryCoords[1];
            int index = map2d[i].in_index;
            if ( index<c0 )
            {
                const sofa::core::topology::BaseMeshTopology::Triangle& triangle = triangles[index];
                Out::setCPos(out[i+i0] , in[triangle[0]] * ( 1-fx-fy )
                        + in[triangle[1]] * fx
                        + in[triangle[2]] * fy );
            }
            else
            {
                if (quads.size())
                {
                    const sofa::core::topology::BaseMeshTopology::Quad& quad = quads[index-c0];
                    Out::setCPos(out[i+i0] , in[quad[0]] * ( ( 1-fx ) * ( 1-fy ) )
                            + in[quad[1]] * ( ( fx ) * ( 1-fy ) )
                            + in[quad[3]] * ( ( 1-fx ) * ( fy ) )
                            + in[quad[2]] * ( ( fx ) * ( fy ) ) );
                }
            }
        }
    }
    // 3D elements
    {
        const int i0 = map1d.size() + map2d.size();
        const int c0 = tetrahedra.size();
        for ( unsigned int i=0; i<map3d.size(); i++ )
        {
            const Real fx = map3d[i].baryCoords[0];
            const Real fy = map3d[i].baryCoords[1];
            const Real fz = map3d[i].baryCoords[2];
            int index = map3d[i].in_index;
            if ( index<c0 )
            {
                const sofa::core::topology::BaseMeshTopology::Tetra& tetra = tetrahedra[index];
                Out::setCPos(out[i+i0] , in[tetra[0]] * ( 1-fx-fy-fz )
                        + in[tetra[1]] * fx
                        + in[tetra[2]] * fy
                        + in[tetra[3]] * fz );
            }
            else
            {
#ifdef SOFA_NEW_HEXA
                const sofa::core::topology::BaseMeshTopology::Hexa& cube = cubes[index-c0];
#else
                const sofa::core::topology::BaseMeshTopology::Cube& cube = cubes[index-c0];
#endif
                Out::setCPos(out[i+i0] , in[cube[0]] * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) )
                        + in[cube[1]] * ( ( fx ) * ( 1-fy ) * ( 1-fz ) )
#ifdef SOFA_NEW_HEXA
                        + in[cube[3]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                        + in[cube[2]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
#else
                        + in[cube[2]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                        + in[cube[3]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
#endif
                        + in[cube[4]] * ( ( 1-fx ) * ( 1-fy ) * ( fz ) )
                        + in[cube[5]] * ( ( fx ) * ( 1-fy ) * ( fz ) )
#ifdef SOFA_NEW_HEXA
                        + in[cube[7]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                        + in[cube[6]] * ( ( fx ) * ( fy ) * ( fz ) ) );
#else
                        + in[cube[6]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                        + in[cube[7]] * ( ( fx ) * ( fy ) * ( fz ) ) );
#endif
            }
        }
    }
}


template <class In, class Out>
void BarycentricMapperRegularGridTopology<In,Out>::resize( core::State<Out>* toModel )
{
    toModel->resize(map.size());
}


template <class In, class Out>
void BarycentricMapperRegularGridTopology<In,Out>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize( map.size() );

    for ( unsigned int i=0; i<map.size(); i++ )
    {
#ifdef SOFA_NEW_HEXA
        const topology::RegularGridTopology::Hexa cube = this->fromTopology->getHexaCopy ( this->map[i].in_index );
#else
        const topology::RegularGridTopology::Cube cube = this->fromTopology->getCubeCopy ( this->map[i].in_index );
#endif
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        Out::setCPos(out[i] , in[cube[0]] * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) )
                + in[cube[1]] * ( ( fx ) * ( 1-fy ) * ( 1-fz ) )
#ifdef SOFA_NEW_HEXA
                + in[cube[3]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                + in[cube[2]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
#else
                + in[cube[2]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                + in[cube[3]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
#endif
                + in[cube[4]] * ( ( 1-fx ) * ( 1-fy ) * ( fz ) )
                + in[cube[5]] * ( ( fx ) * ( 1-fy ) * ( fz ) )
#ifdef SOFA_NEW_HEXA
                + in[cube[7]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                + in[cube[6]] * ( ( fx ) * ( fy ) * ( fz ) ) );
#else
                + in[cube[6]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                + in[cube[7]] * ( ( fx ) * ( fy ) * ( fz ) ) );
#endif
    }
}

template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::resize( core::State<Out>* toModel )
{
    toModel->resize(map.size());
}

template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize( map.size() );

    typedef sofa::helper::vector< CubeData > CubeDataVector;
    typedef typename CubeDataVector::const_iterator CubeDataVectorIt;

    CubeDataVectorIt it = map.begin();
    CubeDataVectorIt itEnd = map.end();

    unsigned int i = 0;

    while (it != itEnd)
    {
#ifdef SOFA_NEW_HEXA
        const topology::SparseGridTopology::Hexa cube = this->fromTopology->getHexahedron( it->in_index );
#else
        const topology::SparseGridTopology::Cube cube = this->fromTopology->getCube ( it->in_index );
#endif
        const Real fx = it->baryCoords[0];
        const Real fy = it->baryCoords[1];
        const Real fz = it->baryCoords[2];

        Out::setCPos(out[i] , in[cube[0]] * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) )
                + in[cube[1]] * ( ( fx ) * ( 1-fy ) * ( 1-fz ) )
#ifdef SOFA_NEW_HEXA
                + in[cube[3]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                + in[cube[2]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
#else
                + in[cube[2]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                + in[cube[3]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
#endif
                + in[cube[4]] * ( ( 1-fx ) * ( 1-fy ) * ( fz ) )
                + in[cube[5]] * ( ( fx ) * ( 1-fy ) * ( fz ) )
#ifdef SOFA_NEW_HEXA
                + in[cube[7]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                + in[cube[6]] * ( ( fx ) * ( fy ) * ( fz ) ) );
#else
                + in[cube[6]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                + in[cube[7]] * ( ( fx ) * ( fy ) * ( fz ) ) );
#endif
        ++it;
        ++i;
    }
}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::resize( core::State<Out>* toModel )
{
    toModel->resize(map.getValue().size());
}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize( map.getValue().size() );

    const sofa::helper::vector<core::topology::BaseMeshTopology::Edge>& edges = this->fromTopology->getEdges();
    // 2D elements
    helper::vector<MappingData>& vectorData = *(map.beginEdit());

    for ( unsigned int i=0; i<map.getValue().size(); i++ )
    {
        const Real fx = vectorData[i].baryCoords[0];
        int index = vectorData[i].in_index;
        const core::topology::BaseMeshTopology::Edge& edge = edges[index];
        Out::setCPos(out[i] , in[edge[0]] * ( 1-fx )
                + in[edge[1]] * fx );
    }
}

template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::resize( core::State<Out>* toModel )
{
    toModel->resize(map.getValue().size());
}

template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize( map.getValue().size() );

    const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle>& triangles = this->fromTopology->getTriangles();
    for ( unsigned int i=0; i<map.getValue().size(); i++ )
    {
        const Real fx = map.getValue()[i].baryCoords[0];
        const Real fy = map.getValue()[i].baryCoords[1];
        int index = map.getValue()[i].in_index;
        const core::topology::BaseMeshTopology::Triangle& triangle = triangles[index];
        Out::setCPos(out[i] , in[triangle[0]] * ( 1-fx-fy )
                + in[triangle[1]] * fx
                + in[triangle[2]] * fy );
    }
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::resize( core::State<Out>* toModel )
{
    toModel->resize(map.getValue().size());
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize( map.getValue().size() );

    const sofa::helper::vector<core::topology::BaseMeshTopology::Quad>& quads = this->fromTopology->getQuads();
    for ( unsigned int i=0; i<map.getValue().size(); i++ )
    {
        const Real fx = map.getValue()[i].baryCoords[0];
        const Real fy = map.getValue()[i].baryCoords[1];
        int index = map.getValue()[i].in_index;
        const core::topology::BaseMeshTopology::Quad& quad = quads[index];
        Out::setCPos(out[i] , in[quad[0]] * ( ( 1-fx ) * ( 1-fy ) )
                + in[quad[1]] * ( ( fx ) * ( 1-fy ) )
                + in[quad[3]] * ( ( 1-fx ) * ( fy ) )
                + in[quad[2]] * ( ( fx ) * ( fy ) ) );
    }
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::resize( core::State<Out>* toModel )
{
    toModel->resize(map.getValue().size());
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize( map.getValue().size() );

    const sofa::helper::vector<core::topology::BaseMeshTopology::Tetrahedron>& tetrahedra = this->fromTopology->getTetrahedra();
    for ( unsigned int i=0; i<map.getValue().size(); i++ )
    {
        const Real fx = map.getValue()[i].baryCoords[0];
        const Real fy = map.getValue()[i].baryCoords[1];
        const Real fz = map.getValue()[i].baryCoords[2];
        int index = map.getValue()[i].in_index;
        const core::topology::BaseMeshTopology::Tetrahedron& tetra = tetrahedra[index];
        Out::setCPos(out[i] , in[tetra[0]] * ( 1-fx-fy-fz )
                + in[tetra[1]] * fx
                + in[tetra[2]] * fy
                + in[tetra[3]] * fz );
    }
    //serr<<"BarycentricMapperTetrahedronSetTopology<In,Out>::apply, in = "<<in<<sendl;
    //serr<<"BarycentricMapperTetrahedronSetTopology<In,Out>::apply, out = "<<out<<sendl;
}

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::resize( core::State<Out>* toModel )
{
    toModel->resize(map.getValue().size());
}

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize( map.getValue().size() );

    const sofa::helper::vector<core::topology::BaseMeshTopology::Hexahedron>& cubes = this->fromTopology->getHexahedra();
    for ( unsigned int i=0; i<map.getValue().size(); i++ )
    {
        const Real fx = map.getValue()[i].baryCoords[0];
        const Real fy = map.getValue()[i].baryCoords[1];
        const Real fz = map.getValue()[i].baryCoords[2];
        int index = map.getValue()[i].in_index;
        const core::topology::BaseMeshTopology::Hexahedron& cube = cubes[index];
        Out::setCPos(out[i] , in[cube[0]] * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) )
                + in[cube[1]] * ( ( fx ) * ( 1-fy ) * ( 1-fz ) )
                + in[cube[3]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                + in[cube[2]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
                + in[cube[4]] * ( ( 1-fx ) * ( 1-fy ) * ( fz ) )
                + in[cube[5]] * ( ( fx ) * ( 1-fy ) * ( fz ) )
                + in[cube[7]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                + in[cube[6]] * ( ( fx ) * ( fy ) * ( fz ) ) );
    }
}


//-- test mapping partiel
template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::applyOnePoint( const unsigned int& hexaPointId,typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    const sofa::helper::vector<core::topology::BaseMeshTopology::Hexahedron>& cubes = this->fromTopology->getHexahedra();
    const Real fx = map.getValue()[hexaPointId].baryCoords[0];
    const Real fy = map.getValue()[hexaPointId].baryCoords[1];
    const Real fz = map.getValue()[hexaPointId].baryCoords[2];
    int index = map.getValue()[hexaPointId].in_index;
    const core::topology::BaseMeshTopology::Hexahedron& cube = cubes[index];
    Out::setCPos(out[hexaPointId] , in[cube[0]] * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) )
            + in[cube[1]] * ( ( fx ) * ( 1-fy ) * ( 1-fz ) )
            + in[cube[3]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
            + in[cube[2]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
            + in[cube[4]] * ( ( 1-fx ) * ( 1-fy ) * ( fz ) )
            + in[cube[5]] * ( ( fx ) * ( 1-fy ) * ( fz ) )
            + in[cube[7]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
            + in[cube[6]] * ( ( fx ) * ( fy ) * ( fz ) ) );
}
//--

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::applyJ (const core::MechanicalParams * /*mparams*/, Data< typename Out::VecDeriv >& _out, const Data< typename In::VecDeriv >& in)
{
#ifdef SOFA_DEV
    if ( sleeping.getValue()==false)
    {
#endif
        typename Out::VecDeriv* out = _out.beginEdit();
        if (mapper != NULL)
        {
            mapper->applyJ(*out, in.getValue());
        }
        _out.endEdit();
#ifdef SOFA_DEV
    }
#endif
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize( map1d.size() +map2d.size() +map3d.size() );

    const sofa::core::topology::BaseMeshTopology::SeqLines& lines = this->fromTopology->getLines();
    const sofa::core::topology::BaseMeshTopology::SeqTriangles& triangles = this->fromTopology->getTriangles();
    const sofa::core::topology::BaseMeshTopology::SeqQuads& quads = this->fromTopology->getQuads();
    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = this->fromTopology->getTetrahedra();
#ifdef SOFA_NEW_HEXA
    const sofa::core::topology::BaseMeshTopology::SeqHexahedra& cubes = this->fromTopology->getHexahedra();
#else
    const sofa::core::topology::BaseMeshTopology::SeqCubes& cubes = this->fromTopology->getCubes();
#endif


    //std::cout << "BarycentricMapper: applyJ with masks" << std::endl;

    const size_t sizeMap1d=map1d.size();
    const size_t sizeMap2d=map2d.size();
    const size_t sizeMap3d=map3d.size();

    const size_t idxStart1=sizeMap1d;
    const size_t idxStart2=sizeMap1d+sizeMap2d;
    const size_t idxStart3=sizeMap1d+sizeMap2d+sizeMap3d;

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( this->maskTo->isActivated() && !this->maskTo->getEntry(i) ) continue;

        // 1D elements
        if (i < idxStart1)
        {
            const Real fx = map1d[i].baryCoords[0];
            int index = map1d[i].in_index;
            {
                const sofa::core::topology::BaseMeshTopology::Line& line = lines[index];
                Out::setDPos(out[i] , in[line[0]] * ( 1-fx )
                        + in[line[1]] * fx );
            }
        }
        // 2D elements
        else if (i < idxStart2)
        {
            const size_t i0 = idxStart1;
            const size_t c0 = triangles.size();

            const Real fx = map2d[i-i0].baryCoords[0];
            const Real fy = map2d[i-i0].baryCoords[1];
            size_t index = map2d[i-i0].in_index;

            if ( index<c0 )
            {
                const sofa::core::topology::BaseMeshTopology::Triangle& triangle = triangles[index];
                Out::setDPos(out[i] , in[triangle[0]] * ( 1-fx-fy )
                        + in[triangle[1]] * fx
                        + in[triangle[2]] * fy );
            }
            else
            {
                const sofa::core::topology::BaseMeshTopology::Quad& quad = quads[index-c0];
                Out::setDPos(out[i] , in[quad[0]] * ( ( 1-fx ) * ( 1-fy ) )
                        + in[quad[1]] * ( ( fx ) * ( 1-fy ) )
                        + in[quad[3]] * ( ( 1-fx ) * ( fy ) )
                        + in[quad[2]] * ( ( fx ) * ( fy ) ) );
            }
        }
        // 3D elements
        else if (i < idxStart3)
        {
            const size_t i0 = idxStart2;
            const size_t c0 = tetrahedra.size();
            const Real fx = map3d[i-i0].baryCoords[0];
            const Real fy = map3d[i-i0].baryCoords[1];
            const Real fz = map3d[i-i0].baryCoords[2];
            size_t index = map3d[i-i0].in_index;
            if ( index<c0 )
            {
                const sofa::core::topology::BaseMeshTopology::Tetra& tetra = tetrahedra[index];
                Out::setDPos(out[i] , in[tetra[0]] * ( 1-fx-fy-fz )
                        + in[tetra[1]] * fx
                        + in[tetra[2]] * fy
                        + in[tetra[3]] * fz );
            }
            else
            {
#ifdef SOFA_NEW_HEXA
                const sofa::core::topology::BaseMeshTopology::Hexa& cube = cubes[index-c0];
#else
                const sofa::core::topology::BaseMeshTopology::Cube& cube = cubes[index-c0];
#endif
                Out::setDPos(out[i] , in[cube[0]] * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) )
                        + in[cube[1]] * ( ( fx ) * ( 1-fy ) * ( 1-fz ) )
#ifdef SOFA_NEW_HEXA
                        + in[cube[3]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                        + in[cube[2]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
#else
                        + in[cube[2]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                        + in[cube[3]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
#endif
                        + in[cube[4]] * ( ( 1-fx ) * ( 1-fy ) * ( fz ) )
                        + in[cube[5]] * ( ( fx ) * ( 1-fy ) * ( fz ) )
#ifdef SOFA_NEW_HEXA
                        + in[cube[7]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                        + in[cube[6]] * ( ( fx ) * ( fy ) * ( fz ) ) );
#else
                        + in[cube[6]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                        + in[cube[7]] * ( ( fx ) * ( fy ) * ( fz ) ) );
#endif
            }
        }
    }

}

template <class In, class Out>
void BarycentricMapperRegularGridTopology<In,Out>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize( map.size() );

    for( size_t index=0 ; index<this->maskTo->size() ; ++index)
    {
        if( this->maskTo->isActivated() && !this->maskTo->getEntry(index) ) continue;

#ifdef SOFA_NEW_HEXA
        const topology::RegularGridTopology::Hexa cube = this->fromTopology->getHexaCopy ( this->map[index].in_index );
#else
        const topology::RegularGridTopology::Cube cube = this->fromTopology->getCubeCopy ( this->map[index].in_index );
#endif
        const Real fx = map[index].baryCoords[0];
        const Real fy = map[index].baryCoords[1];
        const Real fz = map[index].baryCoords[2];
        Out::setDPos(out[index] , in[cube[0]] * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) )
                + in[cube[1]] * ( ( fx ) * ( 1-fy ) * ( 1-fz ) )
#ifdef SOFA_NEW_HEXA
                + in[cube[3]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                + in[cube[2]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
#else
                + in[cube[2]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                + in[cube[3]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
#endif
                + in[cube[4]] * ( ( 1-fx ) * ( 1-fy ) * ( fz ) )
                + in[cube[5]] * ( ( fx ) * ( 1-fy ) * ( fz ) )
#ifdef SOFA_NEW_HEXA
                + in[cube[7]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                + in[cube[6]] * ( ( fx ) * ( fy ) * ( fz ) ) );
#else
                + in[cube[6]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                + in[cube[7]] * ( ( fx ) * ( fy ) * ( fz ) ) );
#endif
    }
}

template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize( map.size() );

    for( size_t index=0 ; index<this->maskTo->size() ; ++index)
    {
        if( this->maskTo->isActivated() && !this->maskTo->getEntry(index) ) continue;

#ifdef SOFA_NEW_HEXA
        const topology::SparseGridTopology::Hexa cube = this->fromTopology->getHexahedron ( this->map[index].in_index );
#else
        const topology::SparseGridTopology::Cube cube = this->fromTopology->getCube ( this->map[index].in_index );
#endif
        const Real fx = map[index].baryCoords[0];
        const Real fy = map[index].baryCoords[1];
        const Real fz = map[index].baryCoords[2];
        Out::setDPos(out[index] , in[cube[0]] * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) )
                + in[cube[1]] * ( ( fx ) * ( 1-fy ) * ( 1-fz ) )
#ifdef SOFA_NEW_HEXA
                + in[cube[3]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                + in[cube[2]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
#else
                + in[cube[2]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                + in[cube[3]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
#endif
                + in[cube[4]] * ( ( 1-fx ) * ( 1-fy ) * ( fz ) )
                + in[cube[5]] * ( ( fx ) * ( 1-fy ) * ( fz ) )
#ifdef SOFA_NEW_HEXA
                + in[cube[7]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                + in[cube[6]] * ( ( fx ) * ( fy ) * ( fz ) ) );
#else
                + in[cube[6]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                + in[cube[7]] * ( ( fx ) * ( fy ) * ( fz ) ) );
#endif
    }

}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize( map.getValue().size() );

    const sofa::helper::vector<core::topology::BaseMeshTopology::Edge>& edges = this->fromTopology->getEdges();

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( this->maskTo->isActivated() && !this->maskTo->getEntry(i) ) continue;

        const Real fx = map.getValue()[i].baryCoords[0];
        int index = map.getValue()[i].in_index;
        const core::topology::BaseMeshTopology::Edge& edge = edges[index];
        Out::setDPos(out[i] , in[edge[0]] * ( 1-fx )
                + in[edge[1]] * fx);
    }
}

template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize( map.getValue().size() );

    const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle>& triangles = this->fromTopology->getTriangles();

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( this->maskTo->isActivated() && !this->maskTo->getEntry(i) ) continue;

        const Real fx = map.getValue()[i].baryCoords[0];
        const Real fy = map.getValue()[i].baryCoords[1];
        int index = map.getValue()[i].in_index;
        const core::topology::BaseMeshTopology::Triangle& triangle = triangles[index];
        Out::setDPos(out[i] , in[triangle[0]] * ( 1-fx-fy )
                + in[triangle[1]] * fx
                + in[triangle[2]] * fy);
    }
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize( map.getValue().size() );

    const sofa::helper::vector<core::topology::BaseMeshTopology::Quad>& quads = this->fromTopology->getQuads();

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( this->maskTo->isActivated() && !this->maskTo->getEntry(i) ) continue;

        const Real fx = map.getValue()[i].baryCoords[0];
        const Real fy = map.getValue()[i].baryCoords[1];
        int index = map.getValue()[i].in_index;
        const core::topology::BaseMeshTopology::Quad& quad = quads[index];
        Out::setDPos(out[i] , in[quad[0]] * ( ( 1-fx ) * ( 1-fy ) )
                + in[quad[1]] * ( ( fx ) * ( 1-fy ) )
                + in[quad[3]] * ( ( 1-fx ) * ( fy ) )
                + in[quad[2]] * ( ( fx ) * ( fy ) ) );
    }
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize( map.getValue().size() );

    const sofa::helper::vector<core::topology::BaseMeshTopology::Tetrahedron>& tetrahedra = this->fromTopology->getTetrahedra();


    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( this->maskTo->isActivated() && !this->maskTo->getEntry(i) ) continue;

        const Real fx = map.getValue()[i].baryCoords[0];
        const Real fy = map.getValue()[i].baryCoords[1];
        const Real fz = map.getValue()[i].baryCoords[2];
        int index = map.getValue()[i].in_index;
        const core::topology::BaseMeshTopology::Tetrahedron& tetra = tetrahedra[index];
        Out::setDPos(out[i] , in[tetra[0]] * ( 1-fx-fy-fz )
                + in[tetra[1]] * fx
                + in[tetra[2]] * fy
                + in[tetra[3]] * fz );
    }
}

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize( map.getValue().size() );

    const sofa::helper::vector<core::topology::BaseMeshTopology::Hexahedron>& cubes = this->fromTopology->getHexahedra();

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( this->maskTo->isActivated() && !this->maskTo->getEntry(i) ) continue;

        const Real fx = map.getValue()[i].baryCoords[0];
        const Real fy = map.getValue()[i].baryCoords[1];
        const Real fz = map.getValue()[i].baryCoords[2];
        int index = map.getValue()[i].in_index;
        const core::topology::BaseMeshTopology::Hexahedron& cube = cubes[index];
        Out::setDPos(out[i] ,
                in[cube[0]] * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) )
                + in[cube[1]] * ( ( fx ) * ( 1-fy ) * ( 1-fz ) )
                + in[cube[3]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                + in[cube[2]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
                + in[cube[4]] * ( ( 1-fx ) * ( 1-fy ) * ( fz ) )
                + in[cube[5]] * ( ( fx ) * ( 1-fy ) * ( fz ) )
                + in[cube[7]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                + in[cube[6]] * ( ( fx ) * ( fy ) * ( fz ) ) );
    }
}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::applyJT (const core::MechanicalParams * /*mparams*/, Data< typename In::VecDeriv >& out, const Data< typename Out::VecDeriv >& in)
{
    if (
#ifdef SOFA_DEV
        sleeping.getValue()==false &&
#endif
        mapper != NULL)
    {
        mapper->applyJT(*out.beginEdit(), in.getValue());
        out.endEdit();
    }
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const sofa::core::topology::BaseMeshTopology::SeqLines& lines = this->fromTopology->getLines();
    const sofa::core::topology::BaseMeshTopology::SeqTriangles& triangles = this->fromTopology->getTriangles();
    const sofa::core::topology::BaseMeshTopology::SeqQuads& quads = this->fromTopology->getQuads();
    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = this->fromTopology->getTetrahedra();
#ifdef SOFA_NEW_HEXA
    const sofa::core::topology::BaseMeshTopology::SeqHexahedra& cubes = this->fromTopology->getHexahedra();
#else
    const sofa::core::topology::BaseMeshTopology::SeqCubes& cubes = this->fromTopology->getCubes();
#endif

    const size_t i1d = map1d.size();
    const size_t i2d = map2d.size();
    const size_t i3d = map3d.size();

    ForceMask& mask = *this->maskFrom;

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( !this->maskTo->getEntry(i) ) continue;

        // 1D elements
        if (i < i1d)
        {
            const typename Out::DPos v = Out::getDPos(in[i]);
            const OutReal fx = ( OutReal ) map1d[i].baryCoords[0];
            size_t index = map1d[i].in_index;
            {
                const sofa::core::topology::BaseMeshTopology::Line& line = lines[index];
                out[line[0]] += v * ( 1-fx );
                out[line[1]] += v * fx;
                mask.insertEntry(line[0]);
                mask.insertEntry(line[1]);
            }
        }
        // 2D elements
        else if (i < i1d+i2d)
        {
            const size_t i0 = map1d.size();
            const size_t c0 = triangles.size();
            const typename Out::DPos v = Out::getDPos(in[i]);
            const OutReal fx = ( OutReal ) map2d[i-i0].baryCoords[0];
            const OutReal fy = ( OutReal ) map2d[i-i0].baryCoords[1];
            size_t index = map2d[i-i0].in_index;
            if ( index<c0 )
            {
                const sofa::core::topology::BaseMeshTopology::Triangle& triangle = triangles[index];
                out[triangle[0]] += v * ( 1-fx-fy );
                out[triangle[1]] += v * fx;
                out[triangle[2]] += v * fy;
                mask.insertEntry(triangle[0]);
                mask.insertEntry(triangle[1]);
                mask.insertEntry(triangle[2]);
            }
            else
            {
                const sofa::core::topology::BaseMeshTopology::Quad& quad = quads[index-c0];
                out[quad[0]] += v * ( ( 1-fx ) * ( 1-fy ) );
                out[quad[1]] += v * ( ( fx ) * ( 1-fy ) );
                out[quad[3]] += v * ( ( 1-fx ) * ( fy ) );
                out[quad[2]] += v * ( ( fx ) * ( fy ) );
                mask.insertEntry(quad[0]);
                mask.insertEntry(quad[1]);
                mask.insertEntry(quad[2]);
                mask.insertEntry(quad[3]);
            }
        }
        // 3D elements
        else if (i < i1d+i2d+i3d)
        {
            const size_t i0 = map1d.size() + map2d.size();
            const size_t c0 = tetrahedra.size();
            const typename Out::DPos v = Out::getDPos(in[i]);
            const OutReal fx = ( OutReal ) map3d[i-i0].baryCoords[0];
            const OutReal fy = ( OutReal ) map3d[i-i0].baryCoords[1];
            const OutReal fz = ( OutReal ) map3d[i-i0].baryCoords[2];
            size_t index = map3d[i-i0].in_index;
            if ( index<c0 )
            {
                const sofa::core::topology::BaseMeshTopology::Tetra& tetra = tetrahedra[index];
                out[tetra[0]] += v * ( 1-fx-fy-fz );
                out[tetra[1]] += v * fx;
                out[tetra[2]] += v * fy;
                out[tetra[3]] += v * fz;
                mask.insertEntry(tetra[0]);
                mask.insertEntry(tetra[1]);
                mask.insertEntry(tetra[2]);
                mask.insertEntry(tetra[3]);
            }
            else
            {
#ifdef SOFA_NEW_HEXA
                const sofa::core::topology::BaseMeshTopology::Hexa& cube = cubes[index-c0];
#else
                const sofa::core::topology::BaseMeshTopology::Cube& cube = cubes[index-c0];
#endif
                out[cube[0]] += v * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) );
                out[cube[1]] += v * ( ( fx ) * ( 1-fy ) * ( 1-fz ) );
#ifdef SOFA_NEW_HEXA
                out[cube[3]] += v * ( ( 1-fx ) * ( fy ) * ( 1-fz ) );
                out[cube[2]] += v * ( ( fx ) * ( fy ) * ( 1-fz ) );
#else
                out[cube[2]] += v * ( ( 1-fx ) * ( fy ) * ( 1-fz ) );
                out[cube[3]] += v * ( ( fx ) * ( fy ) * ( 1-fz ) );
#endif
                out[cube[4]] += v * ( ( 1-fx ) * ( 1-fy ) * ( fz ) );
                out[cube[5]] += v * ( ( fx ) * ( 1-fy ) * ( fz ) );
#ifdef SOFA_NEW_HEXA
                out[cube[7]] += v * ( ( 1-fx ) * ( fy ) * ( fz ) );
                out[cube[6]] += v * ( ( fx ) * ( fy ) * ( fz ) );
#else
                out[cube[6]] += v * ( ( 1-fx ) * ( fy ) * ( fz ) );
                out[cube[7]] += v * ( ( fx ) * ( fy ) * ( fz ) );
#endif

                mask.insertEntry(cube[0]);
                mask.insertEntry(cube[1]);
                mask.insertEntry(cube[2]);
                mask.insertEntry(cube[3]);
                mask.insertEntry(cube[4]);
                mask.insertEntry(cube[5]);
                mask.insertEntry(cube[6]);
                mask.insertEntry(cube[7]);
            }
        }
    }
}




template <class In, class Out>
void BarycentricMapperRegularGridTopology<In,Out>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    ForceMask& mask = *this->maskFrom;

    for( size_t index=0 ; index<this->maskTo->size() ; ++index)
    {
        if( !this->maskTo->getEntry(index) ) continue;

        const typename Out::DPos v = Out::getDPos(in[index]);
#ifdef SOFA_NEW_HEXA
        const topology::RegularGridTopology::Hexa cube = this->fromTopology->getHexaCopy ( this->map[index].in_index );
#else
        const topology::RegularGridTopology::Cube cube = this->fromTopology->getCubeCopy ( this->map[index].in_index );
#endif
        const OutReal fx = ( OutReal ) map[index].baryCoords[0];
        const OutReal fy = ( OutReal ) map[index].baryCoords[1];
        const OutReal fz = ( OutReal ) map[index].baryCoords[2];
        out[cube[0]] += v * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) );
        out[cube[1]] += v * ( ( fx ) * ( 1-fy ) * ( 1-fz ) );
#ifdef SOFA_NEW_HEXA
        out[cube[3]] += v * ( ( 1-fx ) * ( fy ) * ( 1-fz ) );
        out[cube[2]] += v * ( ( fx ) * ( fy ) * ( 1-fz ) );
#else
        out[cube[2]] += v * ( ( 1-fx ) * ( fy ) * ( 1-fz ) );
        out[cube[3]] += v * ( ( fx ) * ( fy ) * ( 1-fz ) );
#endif
        out[cube[4]] += v * ( ( 1-fx ) * ( 1-fy ) * ( fz ) );
        out[cube[5]] += v * ( ( fx ) * ( 1-fy ) * ( fz ) );
#ifdef SOFA_NEW_HEXA
        out[cube[7]] += v * ( ( 1-fx ) * ( fy ) * ( fz ) );
        out[cube[6]] += v * ( ( fx ) * ( fy ) * ( fz ) );
#else
        out[cube[6]] += v * ( ( 1-fx ) * ( fy ) * ( fz ) );
        out[cube[7]] += v * ( ( fx ) * ( fy ) * ( fz ) );
#endif
        mask.insertEntry(cube[0]);
        mask.insertEntry(cube[1]);
        mask.insertEntry(cube[2]);
        mask.insertEntry(cube[3]);
        mask.insertEntry(cube[4]);
        mask.insertEntry(cube[5]);
        mask.insertEntry(cube[6]);
        mask.insertEntry(cube[7]);
    }

}

template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    ForceMask& mask = *this->maskFrom;

    for( size_t index=0 ; index<this->maskTo->size() ; ++index)
    {
        if( !this->maskTo->getEntry(index) ) continue;

        const typename Out::DPos v = Out::getDPos(in[index]);
#ifdef SOFA_NEW_HEXA
        const topology::SparseGridTopology::Hexa cube = this->fromTopology->getHexahedron ( this->map[index].in_index );
#else
        const topology::SparseGridTopology::Cube cube = this->fromTopology->getCube ( this->map[index].in_index );
#endif
        const OutReal fx = ( OutReal ) map[index].baryCoords[0];
        const OutReal fy = ( OutReal ) map[index].baryCoords[1];
        const OutReal fz = ( OutReal ) map[index].baryCoords[2];
        out[cube[0]] += v * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) );
        out[cube[1]] += v * ( ( fx ) * ( 1-fy ) * ( 1-fz ) );
#ifdef SOFA_NEW_HEXA
        out[cube[3]] += v * ( ( 1-fx ) * ( fy ) * ( 1-fz ) );
        out[cube[2]] += v * ( ( fx ) * ( fy ) * ( 1-fz ) );
#else
        out[cube[2]] += v * ( ( 1-fx ) * ( fy ) * ( 1-fz ) );
        out[cube[3]] += v * ( ( fx ) * ( fy ) * ( 1-fz ) );
#endif
        out[cube[4]] += v * ( ( 1-fx ) * ( 1-fy ) * ( fz ) );
        out[cube[5]] += v * ( ( fx ) * ( 1-fy ) * ( fz ) );
#ifdef SOFA_NEW_HEXA
        out[cube[7]] += v * ( ( 1-fx ) * ( fy ) * ( fz ) );
        out[cube[6]] += v * ( ( fx ) * ( fy ) * ( fz ) );
#else
        out[cube[6]] += v * ( ( 1-fx ) * ( fy ) * ( fz ) );
        out[cube[7]] += v * ( ( fx ) * ( fy ) * ( fz ) );
#endif

        mask.insertEntry(cube[0]);
        mask.insertEntry(cube[1]);
        mask.insertEntry(cube[2]);
        mask.insertEntry(cube[3]);
        mask.insertEntry(cube[4]);
        mask.insertEntry(cube[5]);
        mask.insertEntry(cube[6]);
        mask.insertEntry(cube[7]);
    }
}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const sofa::helper::vector<core::topology::BaseMeshTopology::Edge>& edges = this->fromTopology->getEdges();

    ForceMask& mask = *this->maskFrom;

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( !this->maskTo->getEntry(i) ) continue;

        const typename Out::DPos v = Out::getDPos(in[i]);
        const OutReal fx = ( OutReal ) map.getValue()[i].baryCoords[0];
        int index = map.getValue()[i].in_index;
        const core::topology::BaseMeshTopology::Edge& edge = edges[index];
        out[edge[0]] += v * ( 1-fx );
        out[edge[1]] += v * fx;

        mask.insertEntry(edge[0]);
        mask.insertEntry(edge[1]);
    }

}

template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle>& triangles = this->fromTopology->getTriangles();

    ForceMask& mask = *this->maskFrom;

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( !this->maskTo->getEntry(i) ) continue;

        const typename Out::DPos v = Out::getDPos(in[i]);
        const OutReal fx = ( OutReal ) map.getValue()[i].baryCoords[0];
        const OutReal fy = ( OutReal ) map.getValue()[i].baryCoords[1];
        int index = map.getValue()[i].in_index;
        const core::topology::BaseMeshTopology::Triangle& triangle = triangles[index];
        out[triangle[0]] += v * ( 1-fx-fy );
        out[triangle[1]] += v * fx;
        out[triangle[2]] += v * fy;
        mask.insertEntry(triangle[0]);
        mask.insertEntry(triangle[1]);
        mask.insertEntry(triangle[2]);
    }

}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const sofa::helper::vector<core::topology::BaseMeshTopology::Quad>& quads = this->fromTopology->getQuads();

    ForceMask& mask = *this->maskFrom;

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( !this->maskTo->getEntry(i) ) continue;

        const typename Out::DPos v = Out::getDPos(in[i]);
        const OutReal fx = ( OutReal ) map.getValue()[i].baryCoords[0];
        const OutReal fy = ( OutReal ) map.getValue()[i].baryCoords[1];
        int index = map.getValue()[i].in_index;
        const core::topology::BaseMeshTopology::Quad& quad = quads[index];
        out[quad[0]] += v * ( ( 1-fx ) * ( 1-fy ) );
        out[quad[1]] += v * ( ( fx ) * ( 1-fy ) );
        out[quad[3]] += v * ( ( 1-fx ) * ( fy ) );
        out[quad[2]] += v * ( ( fx ) * ( fy ) );
        mask.insertEntry(quad[0]);
        mask.insertEntry(quad[1]);
        mask.insertEntry(quad[2]);
        mask.insertEntry(quad[3]);
    }
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const sofa::helper::vector<core::topology::BaseMeshTopology::Tetrahedron>& tetrahedra = this->fromTopology->getTetrahedra();

    ForceMask& mask = *this->maskFrom;

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( !this->maskTo->getEntry(i) ) continue;

        const typename Out::DPos v = Out::getDPos(in[i]);
        const OutReal fx = ( OutReal ) map.getValue()[i].baryCoords[0];
        const OutReal fy = ( OutReal ) map.getValue()[i].baryCoords[1];
        const OutReal fz = ( OutReal ) map.getValue()[i].baryCoords[2];
        int index = map.getValue()[i].in_index;
        const core::topology::BaseMeshTopology::Tetrahedron& tetra = tetrahedra[index];
        out[tetra[0]] += v * ( 1-fx-fy-fz );
        out[tetra[1]] += v * fx;
        out[tetra[2]] += v * fy;
        out[tetra[3]] += v * fz;
        mask.insertEntry(tetra[0]);
        mask.insertEntry(tetra[1]);
        mask.insertEntry(tetra[2]);
        mask.insertEntry(tetra[3]);
    }
}

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const sofa::helper::vector<core::topology::BaseMeshTopology::Hexahedron>& cubes = this->fromTopology->getHexahedra();

    ForceMask& mask = *this->maskFrom;

    //////////////  DEBUG  /////////////
    // unsigned int mapSize = map.size();
    // std::cout << "Map size: " << mapSize << std::endl;
    // for(unsigned int i=0;i<map.size();i++)
    // {
    //   std::cout << "index " << map[i].in_index << ", baryCoord ( " << (OutReal)map[i].baryCoords[0] << ", " << (OutReal)map[i].baryCoords[1] << ", " << (OutReal)map[i].baryCoords[2] << ")." << std::endl;
    // }
    ////////////////////////////////////

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( !this->maskTo->getEntry(i) ) continue;

        const typename Out::DPos v = Out::getDPos(in[i]);
        const OutReal fx = ( OutReal ) map.getValue()[i].baryCoords[0];
        const OutReal fy = ( OutReal ) map.getValue()[i].baryCoords[1];
        const OutReal fz = ( OutReal ) map.getValue()[i].baryCoords[2];
        int index = map.getValue()[i].in_index;
        const core::topology::BaseMeshTopology::Hexahedron& cube = cubes[index];
        out[cube[0]] += v * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) );
        out[cube[1]] += v * ( ( fx ) * ( 1-fy ) * ( 1-fz ) );
        out[cube[3]] += v * ( ( 1-fx ) * ( fy ) * ( 1-fz ) );
        out[cube[2]] += v * ( ( fx ) * ( fy ) * ( 1-fz ) );
        out[cube[4]] += v * ( ( 1-fx ) * ( 1-fy ) * ( fz ) );
        out[cube[5]] += v * ( ( fx ) * ( 1-fy ) * ( fz ) );
        out[cube[7]] += v * ( ( 1-fx ) * ( fy ) * ( fz ) );
        out[cube[6]] += v * ( ( fx ) * ( fy ) * ( fz ) );

        mask.insertEntry(cube[0]);
        mask.insertEntry(cube[1]);
        mask.insertEntry(cube[2]);
        mask.insertEntry(cube[3]);
        mask.insertEntry(cube[4]);
        mask.insertEntry(cube[5]);
        mask.insertEntry(cube[6]);
        mask.insertEntry(cube[7]);
    }
}


template <class TIn, class TOut>
const sofa::defaulttype::BaseMatrix* BarycentricMapping<TIn, TOut>::getJ()
{
    if (
#ifdef SOFA_DEV
        sleeping.getValue()==false &&
#endif
        mapper!=NULL )
    {
        const size_t outStateSize = this->toModel->getSize();
        const size_t inStateSize = this->fromModel->getSize();
        const sofa::defaulttype::BaseMatrix* matJ = mapper->getJ((int)outStateSize, (int)inStateSize);

        return matJ;
    }
    else
        return NULL;
}

template <class In, class Out>
const sofa::defaulttype::BaseMatrix* BarycentricMapperMeshTopology<In,Out>::getJ(int outSize, int inSize)
{

    if (matrixJ && !updateJ && matrixJ->rowBSize() == (MatrixTypeIndex)outSize && matrixJ->colBSize() == (MatrixTypeIndex)inSize)
        return matrixJ;
    if (outSize > 0 && map1d.size()+map2d.size()+map3d.size() == 0)
        return NULL; // error: maps not yet created ?
//	std::cout << "BarycentricMapperMeshTopology: creating " << outSize << "x" << inSize << " " << NOut << "x" << NIn << " J matrix" << std::endl;
    if (!matrixJ) matrixJ = new MatrixType;
    if (matrixJ->rowBSize() != (MatrixTypeIndex)outSize || matrixJ->colBSize() != (MatrixTypeIndex)inSize)
        matrixJ->resize(outSize*NOut, inSize*NIn);
    else
        matrixJ->clear();

    const sofa::core::topology::BaseMeshTopology::SeqLines& lines = this->fromTopology->getLines();
    const sofa::core::topology::BaseMeshTopology::SeqTriangles& triangles = this->fromTopology->getTriangles();
    const sofa::core::topology::BaseMeshTopology::SeqQuads& quads = this->fromTopology->getQuads();
    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = this->fromTopology->getTetrahedra();
#ifdef SOFA_NEW_HEXA
    const sofa::core::topology::BaseMeshTopology::SeqHexahedra& cubes = this->fromTopology->getHexahedra();
#else
    const sofa::core::topology::BaseMeshTopology::SeqCubes& cubes = this->fromTopology->getCubes();
#endif

    // 1D elements
    {
        for ( size_t i=0; i<map1d.size(); i++ )
        {
            const size_t out = i;
            const Real fx = ( Real ) map1d[i].baryCoords[0];
            size_t index = map1d[i].in_index;
            {
                const sofa::core::topology::BaseMeshTopology::Line& line = lines[index];
                this->addMatrixContrib(matrixJ, out, line[0],  ( 1-fx ));
                this->addMatrixContrib(matrixJ, out, line[1],  fx);
            }
        }
    }
    // 2D elements
    {
        const size_t i0 = map1d.size();
        const size_t c0 = triangles.size();
        for ( size_t i=0; i<map2d.size(); i++ )
        {
            const size_t out = i+i0;
            const Real fx = ( Real ) map2d[i].baryCoords[0];
            const Real fy = ( Real ) map2d[i].baryCoords[1];
            size_t index = map2d[i].in_index;
            if ( index<c0 )
            {
                const sofa::core::topology::BaseMeshTopology::Triangle& triangle = triangles[index];
                this->addMatrixContrib(matrixJ, out, triangle[0],  ( 1-fx-fy ));
                this->addMatrixContrib(matrixJ, out, triangle[1],  fx);
                this->addMatrixContrib(matrixJ, out, triangle[2],  fy);
            }
            else
            {
                const sofa::core::topology::BaseMeshTopology::Quad& quad = quads[index-c0];
                this->addMatrixContrib(matrixJ, out, quad[0],  ( ( 1-fx ) * ( 1-fy ) ));
                this->addMatrixContrib(matrixJ, out, quad[1],  ( ( fx ) * ( 1-fy ) ));
                this->addMatrixContrib(matrixJ, out, quad[3],  ( ( 1-fx ) * ( fy ) ));
                this->addMatrixContrib(matrixJ, out, quad[2],  ( ( fx ) * ( fy ) ));
            }
        }
    }
    // 3D elements
    {
        const size_t i0 = map1d.size() + map2d.size();
        const size_t c0 = tetrahedra.size();
        for ( size_t i=0; i<map3d.size(); i++ )
        {
            const size_t out = i+i0;
            const Real fx = ( Real ) map3d[i].baryCoords[0];
            const Real fy = ( Real ) map3d[i].baryCoords[1];
            const Real fz = ( Real ) map3d[i].baryCoords[2];
            size_t index = map3d[i].in_index;
            if ( index<c0 )
            {
                const sofa::core::topology::BaseMeshTopology::Tetra& tetra = tetrahedra[index];
                this->addMatrixContrib(matrixJ, out, tetra[0],  ( 1-fx-fy-fz ));
                this->addMatrixContrib(matrixJ, out, tetra[1],  fx);
                this->addMatrixContrib(matrixJ, out, tetra[2],  fy);
                this->addMatrixContrib(matrixJ, out, tetra[3],  fz);
            }
            else
            {
#ifdef SOFA_NEW_HEXA
                const sofa::core::topology::BaseMeshTopology::Hexa& cube = cubes[index-c0];
#else
                const sofa::core::topology::BaseMeshTopology::Cube& cube = cubes[index-c0];
#endif
                this->addMatrixContrib(matrixJ, out, cube[0],  ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) ));
                this->addMatrixContrib(matrixJ, out, cube[1],  ( ( fx ) * ( 1-fy ) * ( 1-fz ) ));
#ifdef SOFA_NEW_HEXA
                this->addMatrixContrib(matrixJ, out, cube[3],  ( ( 1-fx ) * ( fy ) * ( 1-fz ) ));
                this->addMatrixContrib(matrixJ, out, cube[2],  ( ( fx ) * ( fy ) * ( 1-fz ) ));
#else
                this->addMatrixContrib(matrixJ, out, cube[2],  ( ( 1-fx ) * ( fy ) * ( 1-fz ) ));
                this->addMatrixContrib(matrixJ, out, cube[3],  ( ( fx ) * ( fy ) * ( 1-fz ) ));
#endif
                this->addMatrixContrib(matrixJ, out, cube[4],  ( ( 1-fx ) * ( 1-fy ) * ( fz ) ));
                this->addMatrixContrib(matrixJ, out, cube[5],  ( ( fx ) * ( 1-fy ) * ( fz ) ));
#ifdef SOFA_NEW_HEXA
                this->addMatrixContrib(matrixJ, out, cube[7],  ( ( 1-fx ) * ( fy ) * ( fz ) ));
                this->addMatrixContrib(matrixJ, out, cube[6],  ( ( fx ) * ( fy ) * ( fz ) ));
#else
                this->addMatrixContrib(matrixJ, out, cube[6],  ( ( 1-fx ) * ( fy ) * ( fz ) ));
                this->addMatrixContrib(matrixJ, out, cube[7],  ( ( fx ) * ( fy ) * ( fz ) ));
#endif
            }
        }
    }
    matrixJ->compress();
//	std::cout << "J = " << *matrixJ << std::endl;
    updateJ = false;
    return matrixJ;
}


template <class In, class Out>
const sofa::defaulttype::BaseMatrix* BarycentricMapperRegularGridTopology<In,Out>::getJ(int outSize, int inSize)
{

    if (matrixJ && !updateJ)
        return matrixJ;

    if (!matrixJ) matrixJ = new MatrixType;
    if (matrixJ->rowBSize() != (MatrixTypeIndex)outSize || matrixJ->colBSize() != (MatrixTypeIndex)inSize)
        matrixJ->resize(outSize*NOut, inSize*NIn);
    else
        matrixJ->clear();

    for ( size_t i=0; i<map.size(); i++ )
    {
        const int out = i;
#ifdef SOFA_NEW_HEXA
        const topology::RegularGridTopology::Hexa cube = this->fromTopology->getHexaCopy ( this->map[i].in_index );
#else
        const topology::RegularGridTopology::Cube cube = this->fromTopology->getCubeCopy ( this->map[i].in_index );
#endif
        const Real fx = ( Real ) map[i].baryCoords[0];
        const Real fy = ( Real ) map[i].baryCoords[1];
        const Real fz = ( Real ) map[i].baryCoords[2];
        this->addMatrixContrib(matrixJ, out, cube[0], ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[1], ( ( fx ) * ( 1-fy ) * ( 1-fz ) ));
#ifdef SOFA_NEW_HEXA
        this->addMatrixContrib(matrixJ, out, cube[3], ( ( 1-fx ) * ( fy ) * ( 1-fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[2], ( ( fx ) * ( fy ) * ( 1-fz ) ));
#else
        this->addMatrixContrib(matrixJ, out, cube[2], ( ( 1-fx ) * ( fy ) * ( 1-fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[3], ( ( fx ) * ( fy ) * ( 1-fz ) ));
#endif
        this->addMatrixContrib(matrixJ, out, cube[4], ( ( 1-fx ) * ( 1-fy ) * ( fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[5], ( ( fx ) * ( 1-fy ) * ( fz ) ));
#ifdef SOFA_NEW_HEXA
        this->addMatrixContrib(matrixJ, out, cube[7], ( ( 1-fx ) * ( fy ) * ( fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[6], ( ( fx ) * ( fy ) * ( fz ) ));
#else
        this->addMatrixContrib(matrixJ, out, cube[6], ( ( 1-fx ) * ( fy ) * ( fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[7], ( ( fx ) * ( fy ) * ( fz ) ));
#endif
    }
    updateJ = false;
    return matrixJ;
}

template <class In, class Out>
const sofa::defaulttype::BaseMatrix* BarycentricMapperSparseGridTopology<In,Out>::getJ(int outSize, int inSize)
{

    if (matrixJ && !updateJ)
        return matrixJ;

    if (!matrixJ) matrixJ = new MatrixType;
    if (matrixJ->rowBSize() != (MatrixTypeIndex)outSize || matrixJ->colBSize() != (MatrixTypeIndex)inSize)
        matrixJ->resize(outSize*NOut, inSize*NIn);
    else
        matrixJ->clear();

    for ( size_t i=0; i<map.size(); i++ )
    {
        const int out = i;
#ifdef SOFA_NEW_HEXA
        const topology::SparseGridTopology::Hexa cube = this->fromTopology->getHexahedron ( this->map[i].in_index );
#else
        const topology::SparseGridTopology::Cube cube = this->fromTopology->getCube ( this->map[i].in_index );
#endif
        const Real fx = ( Real ) map[i].baryCoords[0];
        const Real fy = ( Real ) map[i].baryCoords[1];
        const Real fz = ( Real ) map[i].baryCoords[2];
        this->addMatrixContrib(matrixJ, out, cube[0], ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[1], ( ( fx ) * ( 1-fy ) * ( 1-fz ) ));
#ifdef SOFA_NEW_HEXA
        this->addMatrixContrib(matrixJ, out, cube[3], ( ( 1-fx ) * ( fy ) * ( 1-fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[2], ( ( fx ) * ( fy ) * ( 1-fz ) ));
#else
        this->addMatrixContrib(matrixJ, out, cube[2], ( ( 1-fx ) * ( fy ) * ( 1-fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[3], ( ( fx ) * ( fy ) * ( 1-fz ) ));
#endif
        this->addMatrixContrib(matrixJ, out, cube[4], ( ( 1-fx ) * ( 1-fy ) * ( fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[5], ( ( fx ) * ( 1-fy ) * ( fz ) ));
#ifdef SOFA_NEW_HEXA
        this->addMatrixContrib(matrixJ, out, cube[7], ( ( 1-fx ) * ( fy ) * ( fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[6], ( ( fx ) * ( fy ) * ( fz ) ));
#else
        this->addMatrixContrib(matrixJ, out, cube[6], ( ( 1-fx ) * ( fy ) * ( fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[7], ( ( fx ) * ( fy ) * ( fz ) ));
#endif
    }
    matrixJ->compress();
//	std::cout << "J = " << *matrixJ << std::endl;
    updateJ = false;
    return matrixJ;
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

template <class In, class Out>
const sofa::defaulttype::BaseMatrix* BarycentricMapperEdgeSetTopology<In,Out>::getJ(int outSize, int inSize)
{
    if (matrixJ && !updateJ)
        return matrixJ;

    if (!matrixJ) matrixJ = new MatrixType;
    if (matrixJ->rowBSize() != (MatrixTypeIndex)outSize || matrixJ->colBSize() != (MatrixTypeIndex)inSize)
        matrixJ->resize(outSize*NOut, inSize*NIn);
    else
        matrixJ->clear();

    const sofa::helper::vector<core::topology::BaseMeshTopology::Edge>& edges = this->fromTopology->getEdges();

    for( size_t outId=0 ; outId<this->maskTo->size() ; ++outId)
    {
        if( !this->maskTo->getEntry(outId) ) continue;

        const Real fx = map.getValue()[outId].baryCoords[0];
        int index = map.getValue()[outId].in_index;
        const core::topology::BaseMeshTopology::Edge& edge = edges[index];

        this->addMatrixContrib(matrixJ, outId, edge[0], ( 1-fx));
        this->addMatrixContrib(matrixJ, outId, edge[1], (   fx));
    }
    matrixJ->compress();
    updateJ = false;
    return matrixJ;
}

template <class In, class Out>
const sofa::defaulttype::BaseMatrix* BarycentricMapperTriangleSetTopology<In,Out>::getJ(int outSize, int inSize)
{
    if (matrixJ && !updateJ)
        return matrixJ;

    if (!matrixJ) matrixJ = new MatrixType;
    if (matrixJ->rowBSize() != (MatrixTypeIndex)outSize || matrixJ->colBSize() != (MatrixTypeIndex)inSize)
        matrixJ->resize(outSize*NOut, inSize*NIn);
    else
        matrixJ->clear();

    const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle>& triangles = this->fromTopology->getTriangles();

    for( size_t outId=0 ; outId<this->maskTo->size() ; ++outId)
    {
        if( !this->maskTo->getEntry(outId) ) continue;

        const Real fx = map.getValue()[outId].baryCoords[0];
        const Real fy = map.getValue()[outId].baryCoords[1];
        int index = map.getValue()[outId].in_index;
        const core::topology::BaseMeshTopology::Triangle& triangle = triangles[index];
        this->addMatrixContrib(matrixJ, outId, triangle[0], ( 1-fx-fy ));
        this->addMatrixContrib(matrixJ, outId, triangle[1],      ( fx ));
        this->addMatrixContrib(matrixJ, outId, triangle[2],      ( fy ));
    }


    matrixJ->compress();
    updateJ = false;
    return matrixJ;
}

template <class In, class Out>
const sofa::defaulttype::BaseMatrix* BarycentricMapperQuadSetTopology<In,Out>::getJ(int outSize, int inSize)
{
    if (matrixJ && !updateJ)
        return matrixJ;

    if (!matrixJ) matrixJ = new MatrixType;
    if (matrixJ->rowBSize() != (MatrixTypeIndex)outSize || matrixJ->colBSize() != (MatrixTypeIndex)inSize)
        matrixJ->resize(outSize*NOut, inSize*NIn);
    else
        matrixJ->clear();

    const sofa::helper::vector<core::topology::BaseMeshTopology::Quad>& quads = this->fromTopology->getQuads();


    for( size_t outId=0 ; outId<this->maskTo->size() ; ++outId)
    {
        if( !this->maskTo->getEntry(outId) ) continue;

        const Real fx = map.getValue()[outId].baryCoords[0];
        const Real fy = map.getValue()[outId].baryCoords[1];
        int index = map.getValue()[outId].in_index;
        const core::topology::BaseMeshTopology::Quad& quad = quads[index];

        this->addMatrixContrib(matrixJ, outId, quad[0], ( ( 1-fx ) * ( 1-fy ) ));
        this->addMatrixContrib(matrixJ, outId, quad[1], (   ( fx ) * ( 1-fy ) ));
        this->addMatrixContrib(matrixJ, outId, quad[2], (   ( fx ) *   ( fy ) ));
        this->addMatrixContrib(matrixJ, outId, quad[3], ( ( 1-fx ) *   ( fy ) ));
    }

    matrixJ->compress();
    updateJ = false;
    return matrixJ;
}

template <class In, class Out>
const sofa::defaulttype::BaseMatrix* BarycentricMapperTetrahedronSetTopology<In,Out>::getJ(int outSize, int inSize)
{

    if (matrixJ && !updateJ)
        return matrixJ;

    if (!matrixJ) matrixJ = new MatrixType;
    if (matrixJ->rowBSize() != (MatrixTypeIndex)outSize || matrixJ->colBSize() != (MatrixTypeIndex)inSize)
        matrixJ->resize(outSize*NOut, inSize*NIn);
    else
        matrixJ->clear();

    const sofa::helper::vector<core::topology::BaseMeshTopology::Tetrahedron>& tetrahedra = this->fromTopology->getTetrahedra();

    for( size_t outId=0 ; outId<this->maskTo->size() ; ++outId)
    {
        if( !this->maskTo->getEntry(outId) ) continue;

        const Real fx = map.getValue()[outId].baryCoords[0];
        const Real fy = map.getValue()[outId].baryCoords[1];
        const Real fz = map.getValue()[outId].baryCoords[2];
        int index = map.getValue()[outId].in_index;
        const core::topology::BaseMeshTopology::Tetrahedron& tetra = tetrahedra[index];

        this->addMatrixContrib(matrixJ, outId, tetra[0], ( 1-fx-fy-fz ));
        this->addMatrixContrib(matrixJ, outId, tetra[1],         ( fx ));
        this->addMatrixContrib(matrixJ, outId, tetra[2],         ( fy ));
        this->addMatrixContrib(matrixJ, outId, tetra[3],         ( fz ));
    }

    matrixJ->compress();
    updateJ = false;
    return matrixJ;
}

template <class In, class Out>
const sofa::defaulttype::BaseMatrix* BarycentricMapperHexahedronSetTopology<In,Out>::getJ(int outSize, int inSize)
{
    if (matrixJ && !updateJ)
        return matrixJ;

    if (!matrixJ) matrixJ = new MatrixType;
    if (matrixJ->rowBSize() != (MatrixTypeIndex)outSize || matrixJ->colBSize() != (MatrixTypeIndex)inSize)
        matrixJ->resize(outSize*NOut, inSize*NIn);
    else
        matrixJ->clear();

    const sofa::helper::vector<core::topology::BaseMeshTopology::Hexahedron>& cubes = this->fromTopology->getHexahedra();

    for( size_t outId=0 ; outId<this->maskTo->size() ; ++outId)
    {
        if( !this->maskTo->getEntry(outId) ) continue;

        const Real fx = map.getValue()[outId].baryCoords[0];
        const Real fy = map.getValue()[outId].baryCoords[1];
        const Real fz = map.getValue()[outId].baryCoords[2];
        int index = map.getValue()[outId].in_index;
        const core::topology::BaseMeshTopology::Hexahedron& cube = cubes[index];

        this->addMatrixContrib(matrixJ, outId, cube[0], ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) ));
        this->addMatrixContrib(matrixJ, outId, cube[1], (   ( fx ) * ( 1-fy ) * ( 1-fz ) ));
        this->addMatrixContrib(matrixJ, outId, cube[2], (   ( fx ) *   ( fy ) * ( 1-fz ) ));
        this->addMatrixContrib(matrixJ, outId, cube[3], ( ( 1-fx ) *   ( fy ) * ( 1-fz ) ));
        this->addMatrixContrib(matrixJ, outId, cube[4], ( ( 1-fx ) * ( 1-fy ) *   ( fz ) ));
        this->addMatrixContrib(matrixJ, outId, cube[5], (   ( fx ) * ( 1-fy ) *   ( fz ) ));
        this->addMatrixContrib(matrixJ, outId, cube[6], (   ( fx ) *   ( fy ) *   ( fz ) ));
        this->addMatrixContrib(matrixJ, outId, cube[7], ( ( 1-fx ) *   ( fy ) *   ( fz ) ));
    }

    matrixJ->compress();
    updateJ = false;
    return matrixJ;
}


///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////






template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if ( !vparams->displayFlags().getShowMappings() ) return;

    const OutVecCoord& out = this->toModel->read(core::ConstVecCoordId::position())->getValue();
    std::vector< sofa::defaulttype::Vector3 > points;
    for ( unsigned int i=0; i<out.size(); i++ )
    {
        points.push_back ( OutDataTypes::getCPos(out[i]) );
    }
//	glEnd();
    const InVecCoord& in = this->fromModel->read(core::ConstVecCoordId::position())->getValue();
    if ( mapper!=NULL ) mapper->draw(vparams,out, in );

    vparams->drawTool()->drawPoints ( points, 7, sofa::defaulttype::Vec<4,float> ( 1,1,0,1 ) );
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::draw  (const core::visual::VisualParams* vparams,const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    const sofa::core::topology::BaseMeshTopology::SeqLines& lines = this->fromTopology->getLines();
    const sofa::core::topology::BaseMeshTopology::SeqTriangles& triangles = this->fromTopology->getTriangles();
    const sofa::core::topology::BaseMeshTopology::SeqQuads& quads = this->fromTopology->getQuads();
    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = this->fromTopology->getTetrahedra();
#ifdef SOFA_NEW_HEXA
    const sofa::core::topology::BaseMeshTopology::SeqHexahedra& cubes = this->fromTopology->getHexahedra();
#else
    const sofa::core::topology::BaseMeshTopology::SeqCubes& cubes = this->fromTopology->getCubes();
#endif
    std::vector< sofa::defaulttype::Vector3 > points;
    // 1D elements
    {
        const int i0 = 0;
        for ( unsigned int i=0; i<map1d.size(); i++ )
        {
            const Real fx = map1d[i].baryCoords[0];
            int index = map1d[i].in_index;
            {
                const sofa::core::topology::BaseMeshTopology::Line& line = lines[index];
                Real f[2];
                f[0] = ( 1-fx );
                f[1] = fx;
                for ( int j=0; j<2; j++ )
                {
                    if ( f[j]<=-0.0001 || f[j]>=0.0001 )
                    {
                        //                         glColor3f((float)f[j],1,(float)f[j]);
                        points.push_back ( Out::getCPos(out[i+i0]) );
                        points.push_back ( in[line[j]] );
                    }
                }
            }
        }
    }
    // 2D elements
    {
        const int i0 = map1d.size();
        const int c0 = triangles.size();
        for ( unsigned int i=0; i<map2d.size(); i++ )
        {
            const Real fx = map2d[i].baryCoords[0];
            const Real fy = map2d[i].baryCoords[1];
            int index = map2d[i].in_index;
            if ( index<c0 )
            {
                const sofa::core::topology::BaseMeshTopology::Triangle& triangle = triangles[index];
                Real f[3];
                f[0] = ( 1-fx-fy );
                f[1] = fx;
                f[2] = fy;
                for ( int j=0; j<3; j++ )
                {
                    if ( f[j]<=-0.0001 || f[j]>=0.0001 )
                    {
                        //                         glColor3f((float)f[j],1,(float)f[j]);
                        points.push_back ( Out::getCPos(out[i+i0]) );
                        points.push_back ( in[triangle[j]] );
                    }
                }
            }
            else
            {
                const sofa::core::topology::BaseMeshTopology::Quad& quad = quads[index-c0];
                Real f[4];
                f[0] = ( ( 1-fx ) * ( 1-fy ) );
                f[1] = ( ( fx ) * ( 1-fy ) );
                f[3] = ( ( 1-fx ) * ( fy ) );
                f[2] = ( ( fx ) * ( fy ) );
                for ( int j=0; j<4; j++ )
                {
                    if ( f[j]<=-0.0001 || f[j]>=0.0001 )
                    {
                        //                         glColor3f((float)f[j],1,(float)f[j]);
                        points.push_back ( Out::getCPos(out[i+i0]) );
                        points.push_back ( in[quad[j]] );
                    }
                }
            }
        }
    }
    // 3D elements
    {
        const int i0 = map1d.size() +map2d.size();
        const int c0 = tetrahedra.size();
        for ( unsigned int i=0; i<map3d.size(); i++ )
        {
            const Real fx = map3d[i].baryCoords[0];
            const Real fy = map3d[i].baryCoords[1];
            const Real fz = map3d[i].baryCoords[2];
            int index = map3d[i].in_index;
            if ( index<c0 )
            {
                const sofa::core::topology::BaseMeshTopology::Tetra& tetra = tetrahedra[index];
                Real f[4];
                f[0] = ( 1-fx-fy-fz );
                f[1] = fx;
                f[2] = fy;
                f[3] = fz;
                for ( int j=0; j<4; j++ )
                {
                    if ( f[j]<=-0.0001 || f[j]>=0.0001 )
                    {
                        //                         glColor3f((float)f[j],1,(float)f[j]);
                        points.push_back ( Out::getCPos(out[i+i0]) );
                        points.push_back ( in[tetra[j]] );
                    }
                }
            }
            else
            {
#ifdef SOFA_NEW_HEXA
                const sofa::core::topology::BaseMeshTopology::Hexa& cube = cubes[index-c0];
#else
                const sofa::core::topology::BaseMeshTopology::Cube& cube = cubes[index-c0];
#endif
                Real f[8];
                f[0] = ( 1-fx ) * ( 1-fy ) * ( 1-fz );
                f[1] = ( fx ) * ( 1-fy ) * ( 1-fz );
#ifdef SOFA_NEW_HEXA
                f[3] = ( 1-fx ) * ( fy ) * ( 1-fz );
                f[2] = ( fx ) * ( fy ) * ( 1-fz );
#else
                f[2] = ( 1-fx ) * ( fy ) * ( 1-fz );
                f[3] = ( fx ) * ( fy ) * ( 1-fz );
#endif
                f[4] = ( 1-fx ) * ( 1-fy ) * ( fz );
                f[5] = ( fx ) * ( 1-fy ) * ( fz );
#ifdef SOFA_NEW_HEXA
                f[7] = ( 1-fx ) * ( fy ) * ( fz );
                f[6] = ( fx ) * ( fy ) * ( fz );
#else
                f[6] = ( 1-fx ) * ( fy ) * ( fz );
                f[7] = ( fx ) * ( fy ) * ( fz );
#endif
                for ( int j=0; j<8; j++ )
                {
                    if ( f[j]<=-0.0001 || f[j]>=0.0001 )
                    {
                        //                         glColor3f((float)f[j],1,1);
                        points.push_back ( Out::getCPos(out[i+i0]) );
                        points.push_back ( in[cube[j]] );
                    }
                }
            }
        }
    }
    vparams->drawTool()->drawLines ( points, 1, sofa::defaulttype::Vec<4,float> ( 0,1,0,1 ) );
}

template <class In, class Out>
void BarycentricMapperRegularGridTopology<In,Out>::draw  (const core::visual::VisualParams* vparams,const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    std::vector< sofa::defaulttype::Vector3 > points;

    for ( unsigned int i=0; i<map.size(); i++ )
    {
#ifdef SOFA_NEW_HEXA
        const topology::RegularGridTopology::Hexa cube = this->fromTopology->getHexaCopy ( this->map[i].in_index );
#else
        const topology::RegularGridTopology::Cube cube = this->fromTopology->getCubeCopy ( this->map[i].in_index );
#endif
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        Real f[8];
        f[0] = ( 1-fx ) * ( 1-fy ) * ( 1-fz );
        f[1] = ( fx ) * ( 1-fy ) * ( 1-fz );
#ifdef SOFA_NEW_HEXA
        f[3] = ( 1-fx ) * ( fy ) * ( 1-fz );
        f[2] = ( fx ) * ( fy ) * ( 1-fz );
#else
        f[2] = ( 1-fx ) * ( fy ) * ( 1-fz );
        f[3] = ( fx ) * ( fy ) * ( 1-fz );
#endif
        f[4] = ( 1-fx ) * ( 1-fy ) * ( fz );
        f[5] = ( fx ) * ( 1-fy ) * ( fz );
#ifdef SOFA_NEW_HEXA
        f[7] = ( 1-fx ) * ( fy ) * ( fz );
        f[6] = ( fx ) * ( fy ) * ( fz );
#else
        f[6] = ( 1-fx ) * ( fy ) * ( fz );
        f[7] = ( fx ) * ( fy ) * ( fz );
#endif
        for ( int j=0; j<8; j++ )
        {
            if ( f[j]<=-0.0001 || f[j]>=0.0001 )
            {
                //glColor3f((float)f[j],(float)f[j],1);
                points.push_back ( Out::getCPos(out[i]) );
                points.push_back ( in[cube[j]] );
            }
        }
    }
    vparams->drawTool()->drawLines ( points, 1, sofa::defaulttype::Vec<4,float> ( 0,0,1,1 ) );

}

template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::draw  (const core::visual::VisualParams* vparams,const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    std::vector< sofa::defaulttype::Vector3 > points;
    for ( unsigned int i=0; i<map.size(); i++ )
    {
#ifdef SOFA_NEW_HEXA
        const topology::SparseGridTopology::Hexa cube = this->fromTopology->getHexahedron ( this->map[i].in_index );
#else
        const topology::SparseGridTopology::Cube cube = this->fromTopology->getCube ( this->map[i].in_index );
#endif
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        Real f[8];
        f[0] = ( 1-fx ) * ( 1-fy ) * ( 1-fz );
        f[1] = ( fx ) * ( 1-fy ) * ( 1-fz );
#ifdef SOFA_NEW_HEXA
        f[3] = ( 1-fx ) * ( fy ) * ( 1-fz );
        f[2] = ( fx ) * ( fy ) * ( 1-fz );
#else
        f[2] = ( 1-fx ) * ( fy ) * ( 1-fz );
        f[3] = ( fx ) * ( fy ) * ( 1-fz );
#endif
        f[4] = ( 1-fx ) * ( 1-fy ) * ( fz );
        f[5] = ( fx ) * ( 1-fy ) * ( fz );
#ifdef SOFA_NEW_HEXA
        f[7] = ( 1-fx ) * ( fy ) * ( fz );
        f[6] = ( fx ) * ( fy ) * ( fz );
#else
        f[6] = ( 1-fx ) * ( fy ) * ( fz );
        f[7] = ( fx ) * ( fy ) * ( fz );
#endif
        for ( int j=0; j<8; j++ )
        {
            if ( f[j]<=-0.0001 || f[j]>=0.0001 )
            {
                //glColor3f((float)f[j],(float)f[j],1);
                points.push_back ( Out::getCPos(out[i]) );
                points.push_back ( in[cube[j]] );
            }
        }
    }
    vparams->drawTool()->drawLines ( points, 1, sofa::defaulttype::Vec<4,float> ( 0,0,1,1 ) );

}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::draw  (const core::visual::VisualParams* vparams,const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    const sofa::helper::vector<core::topology::BaseMeshTopology::Edge>& edges = this->fromTopology->getEdges();

    std::vector< sofa::defaulttype::Vector3 > points;
    {
        for ( unsigned int i=0; i<map.getValue().size(); i++ )
        {
            const Real fx = map.getValue()[i].baryCoords[0];
            int index = map.getValue()[i].in_index;
            const core::topology::BaseMeshTopology::Edge& edge = edges[index];
            {
                const Real f = Real ( 1.0 )-fx;
                if ( f<=-0.0001 || f>=0.0001 )
                {
                    //                     glColor3f((float)f,1,(float)f);
                    points.push_back ( Out::getCPos(out[i]) );
                    points.push_back ( in[edge[0]] );
                }
            }
            {
                const Real f = fx;
                if ( f<=-0.0001 || f>=0.0001 )
                {
                    //                     glColor3f((float)f,1,(float)f);
                    points.push_back ( Out::getCPos(out[i]) );
                    points.push_back ( in[edge[1]] );
                }
            }
        }
    }
    vparams->drawTool()->drawLines ( points, 1, sofa::defaulttype::Vec<4,float> ( 0,1,0,1 ) );
}

template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::draw  (const core::visual::VisualParams* vparams,const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle>& triangles = this->fromTopology->getTriangles();

    std::vector< sofa::defaulttype::Vector3 > points;
    {
        for ( unsigned int i=0; i<map.getValue().size(); i++ )
        {
            const Real fx = map.getValue()[i].baryCoords[0];
            const Real fy = map.getValue()[i].baryCoords[1];
            int index = map.getValue()[i].in_index;
            const core::topology::BaseMeshTopology::Triangle& triangle = triangles[index];
            Real f[3];
            f[0] = ( 1-fx-fy );
            f[1] = fx;
            f[2] = fy;
            for ( int j=0; j<3; j++ )
            {
                if ( f[j]<=-0.0001 || f[j]>=0.0001 )
                {
                    //                     glColor3f((float)f[j],1,(float)f[j]);
                    points.push_back ( Out::getCPos(out[i]) );
                    points.push_back ( in[triangle[j]] );
                }
            }
        }
    }
    vparams->drawTool()->drawLines ( points, 1, sofa::defaulttype::Vec<4,float> ( 0,1,0,1 ) );
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::draw  (const core::visual::VisualParams* vparams,const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    const sofa::helper::vector<core::topology::BaseMeshTopology::Quad>& quads = this->fromTopology->getQuads();
    std::vector< sofa::defaulttype::Vector3 > points;
    {
        for ( unsigned int i=0; i<map.getValue().size(); i++ )
        {
            const Real fx = map.getValue()[i].baryCoords[0];
            const Real fy = map.getValue()[i].baryCoords[1];
            int index = map.getValue()[i].in_index;
            const core::topology::BaseMeshTopology::Quad& quad = quads[index];
            Real f[4];
            f[0] = ( ( 1-fx ) * ( 1-fy ) );
            f[1] = ( ( fx ) * ( 1-fy ) );
            f[3] = ( ( 1-fx ) * ( fy ) );
            f[2] = ( ( fx ) * ( fy ) );
            for ( int j=0; j<4; j++ )
            {
                if ( f[j]<=-0.0001 || f[j]>=0.0001 )
                {
                    //                     glColor3f((float)f[j],1,(float)f[j]);
                    points.push_back ( Out::getCPos(out[i]) );
                    points.push_back ( in[quad[j]] );
                }
            }
        }
    }
    vparams->drawTool()->drawLines ( points, 1, sofa::defaulttype::Vec<4,float> ( 0,1,0,1 ) );
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::draw  (const core::visual::VisualParams* vparams,const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    const sofa::helper::vector<core::topology::BaseMeshTopology::Tetrahedron>& tetrahedra = this->fromTopology->getTetrahedra();

    std::vector< sofa::defaulttype::Vector3 > points;
    {
        for ( unsigned int i=0; i<map.getValue().size(); i++ )
        {
            const Real fx = map.getValue()[i].baryCoords[0];
            const Real fy = map.getValue()[i].baryCoords[1];
            const Real fz = map.getValue()[i].baryCoords[2];
            int index = map.getValue()[i].in_index;
            const core::topology::BaseMeshTopology::Tetrahedron& tetra = tetrahedra[index];
            Real f[4];
            f[0] = ( 1-fx-fy-fz );
            f[1] = fx;
            f[2] = fy;
            f[3] = fz;
            for ( int j=0; j<4; j++ )
            {
                if ( f[j]<=-0.0001 || f[j]>=0.0001 )
                {
                    //                     glColor3f((float)f[j],1,(float)f[j]);
                    points.push_back ( Out::getCPos(out[i]) );
                    points.push_back ( in[tetra[j]] );
                }
            }
        }
    }
    vparams->drawTool()->drawLines ( points, 1, sofa::defaulttype::Vec<4,float> ( 0,1,0,1 ) );
}

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::draw  (const core::visual::VisualParams* vparams,const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    const sofa::helper::vector<core::topology::BaseMeshTopology::Hexahedron>& cubes = this->fromTopology->getHexahedra();

    std::vector< sofa::defaulttype::Vector3 > points;
    {
        for ( unsigned int i=0; i<map.getValue().size(); i++ )
        {
            const Real fx = map.getValue()[i].baryCoords[0];
            const Real fy = map.getValue()[i].baryCoords[1];
            const Real fz = map.getValue()[i].baryCoords[2];
            int index = map.getValue()[i].in_index;
            const core::topology::BaseMeshTopology::Hexahedron& cube = cubes[index];
            Real f[8];
            f[0] = ( 1-fx ) * ( 1-fy ) * ( 1-fz );
            f[1] = ( fx ) * ( 1-fy ) * ( 1-fz );
            f[3] = ( 1-fx ) * ( fy ) * ( 1-fz );
            f[2] = ( fx ) * ( fy ) * ( 1-fz );
            f[4] = ( 1-fx ) * ( 1-fy ) * ( fz );
            f[5] = ( fx ) * ( 1-fy ) * ( fz );
            f[7] = ( 1-fx ) * ( fy ) * ( fz );
            f[6] = ( fx ) * ( fy ) * ( fz );
            for ( int j=0; j<8; j++ )
            {
                if ( f[j]<=-0.0001 || f[j]>=0.0001 )
                {
                    //                     glColor3f((float)f[j],1,1);
                    points.push_back ( Out::getCPos(out[i]) );
                    points.push_back ( in[cube[j]] );
                }
            }
        }
    }
    vparams->drawTool()->drawLines ( points, 1, sofa::defaulttype::Vec<4,float> ( 0,1,0,1 ) );
}

/************************************* PropagateConstraint ***********************************/


template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::applyJT(const core::ConstraintParams * /*cparams*/, Data< typename In::MatrixDeriv >& out, const Data< typename Out::MatrixDeriv >& in)
{
    if (
#ifdef SOFA_DEV
        sleeping.getValue()==false &&
#endif
        mapper!=NULL )
    {
        mapper->applyJT(*out.beginEdit(), in.getValue());
        out.endEdit();
    }
}


/// @todo Optimization
template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
{
    const sofa::core::topology::BaseMeshTopology::SeqLines& lines = this->fromTopology->getLines();
    const sofa::core::topology::BaseMeshTopology::SeqTriangles& triangles = this->fromTopology->getTriangles();
    const sofa::core::topology::BaseMeshTopology::SeqQuads& quads = this->fromTopology->getQuads();
    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = this->fromTopology->getTetrahedra();
#ifdef SOFA_NEW_HEXA
    const sofa::core::topology::BaseMeshTopology::SeqHexahedra& cubes = this->fromTopology->getHexahedra();
#else
    const sofa::core::topology::BaseMeshTopology::SeqCubes& cubes = this->fromTopology->getCubes();
#endif
    //const size_t iLine = lines.size();
    const size_t iTri = triangles.size();
    //const size_t iQuad = quads.size();
    const size_t iTetra= tetrahedra.size();
    //const size_t iCube = cubes.size();

    const size_t i1d = map1d.size();
    const size_t i2d = map2d.size();
    const size_t i3d = map3d.size();

    size_t indexIn;

    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();

        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            for ( ; colIt != colItEnd; ++colIt)
            {
                indexIn = colIt.index();
                InDeriv data = (InDeriv) Out::getDPos(colIt.val());

                // 1D elements
                if ( indexIn < i1d )
                {
                    const OutReal fx = ( OutReal ) map1d[indexIn].baryCoords[0];
                    size_t index = map1d[indexIn].in_index;
                    {
                        const sofa::core::topology::BaseMeshTopology::Line& line = lines[index];
                        o.addCol( line[0], data * ( 1-fx ) );
                        o.addCol( line[1], data * fx );
                    }
                }
                // 2D elements : triangle or quad
                else if ( indexIn < i2d )
                {
                    const OutReal fx = ( OutReal ) map2d[indexIn].baryCoords[0];
                    const OutReal fy = ( OutReal ) map2d[indexIn].baryCoords[1];
                    size_t index = map2d[indexIn].in_index;
                    if ( index < iTri ) // triangle
                    {
                        const sofa::core::topology::BaseMeshTopology::Triangle& triangle = triangles[index];
                        o.addCol( triangle[0], data * ( 1-fx-fy ) );
                        o.addCol( triangle[1], data * fx );
                        o.addCol( triangle[2], data * fy );
                    }
                    else // 2D element : Quad
                    {
                        const sofa::core::topology::BaseMeshTopology::Quad& quad = quads[index - iTri];
                        o.addCol( quad[0], data * ( ( 1-fx ) * ( 1-fy ) ) );
                        o.addCol( quad[1], data * ( ( fx ) * ( 1-fy ) ) );
                        o.addCol( quad[3], data * ( ( 1-fx ) * ( fy ) ) );
                        o.addCol( quad[2], data * ( ( fx ) * ( fy ) ) );
                    }
                }
                // 3D elements
                else if ( indexIn < i3d )
                {
                    const OutReal fx = ( OutReal ) map3d[indexIn].baryCoords[0];
                    const OutReal fy = ( OutReal ) map3d[indexIn].baryCoords[1];
                    const OutReal fz = ( OutReal ) map3d[indexIn].baryCoords[2];
                    size_t index = map3d[indexIn].in_index;
                    if ( index < iTetra ) // tetra
                    {
                        const sofa::core::topology::BaseMeshTopology::Tetra& tetra = tetrahedra[index];
                        o.addCol ( tetra[0], data * ( 1-fx-fy-fz ) );
                        o.addCol ( tetra[1], data * fx );
                        o.addCol ( tetra[2], data * fy );
                        o.addCol ( tetra[3], data * fz );
                    }
                    else // cube
                    {
#ifdef SOFA_NEW_HEXA
                        const sofa::core::topology::BaseMeshTopology::Hexa& cube = cubes[index-iTetra];
#else
                        const sofa::core::topology::BaseMeshTopology::Cube& cube = cubes[index-iTetra];
#endif
                        o.addCol ( cube[0],data * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) ) ) ;
                        o.addCol ( cube[1],data * ( ( fx ) * ( 1-fy ) * ( 1-fz ) ) ) ;
#ifdef SOFA_NEW_HEXA
                        o.addCol ( cube[3],data * ( ( 1-fx ) * ( fy ) * ( 1-fz ) ) ) ;
                        o.addCol ( cube[2],data * ( ( fx ) * ( fy ) * ( 1-fz ) ) ) ;
#else
                        o.addCol ( cube[2],data * ( ( 1-fx ) * ( fy ) * ( 1-fz ) ) ) ;
                        o.addCol ( cube[3],data * ( ( fx ) * ( fy ) * ( 1-fz ) ) ) ;
#endif
                        o.addCol ( cube[4],data * ( ( 1-fx ) * ( 1-fy ) * ( fz ) ) ) ;
                        o.addCol ( cube[5],data * ( ( fx ) * ( 1-fy ) * ( fz ) ) ) ;
#ifdef SOFA_NEW_HEXA
                        o.addCol ( cube[7],data * ( ( 1-fx ) * ( fy ) * ( fz ) ) ) ;
                        o.addCol ( cube[6],data * ( ( fx ) * ( fy ) * ( fz ) ) ) ;
#else
                        o.addCol ( cube[6],data * ( ( 1-fx ) * ( fy ) * ( fz ) ) ) ;
                        o.addCol ( cube[7],data * ( ( fx ) * ( fy ) * ( fz ) ) );
#endif
                    }
                }
            }
        }
    }
}

template <class In, class Out>
void BarycentricMapperRegularGridTopology<In,Out>::applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
{
    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();

        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            for ( ; colIt != colItEnd; ++colIt)
            {
                unsigned int indexIn = colIt.index();
                InDeriv data = (InDeriv) Out::getDPos(colIt.val());

#ifdef SOFA_NEW_HEXA
                const topology::RegularGridTopology::Hexa cube = this->fromTopology->getHexaCopy ( this->map[indexIn].in_index );
#else
                const topology::RegularGridTopology::Cube cube = this->fromTopology->getCubeCopy ( this->map[indexIn].in_index );
#endif
                const OutReal fx = (OutReal) map[indexIn].baryCoords[0];
                const OutReal fy = (OutReal) map[indexIn].baryCoords[1];
                const OutReal fz = (OutReal) map[indexIn].baryCoords[2];
                const OutReal oneMinusFx = 1-fx;
                const OutReal oneMinusFy = 1-fy;
                const OutReal oneMinusFz = 1-fz;

                o.addCol(cube[0], data * ((oneMinusFx) * (oneMinusFy) * (oneMinusFz)));
                o.addCol(cube[1], data * ((fx) * (oneMinusFy) * (oneMinusFz)));
#ifdef SOFA_NEW_HEXA
                o.addCol(cube[3], data * ((oneMinusFx) * (fy) * (oneMinusFz)));
                o.addCol(cube[2], data * ((fx) * (fy) * (oneMinusFz)));
#else
                o.addCol(cube[2], data * ((oneMinusFx) * (fy) * (oneMinusFz)));
                o.addCol(cube[3], data * ((fx) * (fy) * (oneMinusFz)));
#endif
                o.addCol(cube[4], data * ((oneMinusFx) * (oneMinusFy) * (fz)));
                o.addCol(cube[5], data * ((fx) * (oneMinusFy) * (fz)));
#ifdef SOFA_NEW_HEXA
                o.addCol(cube[7], data * ((oneMinusFx) * (fy) * (fz)));
                o.addCol(cube[6], data * ((fx) * (fy) * (fz)));
#else
                o.addCol(cube[6], data * ((oneMinusFx) * (fy) * (fz)));
                o.addCol(cube[7], data * ((fx) * (fy) * (fz)));
#endif
            }
        }
    }
}

template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
{
    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();

        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            for ( ; colIt != colItEnd; ++colIt)
            {
                unsigned indexIn = colIt.index();
                InDeriv data = (InDeriv) Out::getDPos(colIt.val());

#ifdef SOFA_NEW_HEXA
                const topology::SparseGridTopology::Hexa cube = this->fromTopology->getHexahedron ( this->map[indexIn].in_index );
#else
                const topology::SparseGridTopology::Cube cube = this->fromTopology->getCube ( this->map[indexIn].in_index );
#endif
                const OutReal fx = ( OutReal ) map[indexIn].baryCoords[0];
                const OutReal fy = ( OutReal ) map[indexIn].baryCoords[1];
                const OutReal fz = ( OutReal ) map[indexIn].baryCoords[2];
                const OutReal oneMinusFx = 1-fx;
                const OutReal oneMinusFy = 1-fy;
                const OutReal oneMinusFz = 1-fz;

                OutReal f = ( oneMinusFx * oneMinusFy * oneMinusFz );
                o.addCol ( cube[0],  ( data * f ) );

                f = ( ( fx ) * oneMinusFy * oneMinusFz );
                o.addCol ( cube[1],  ( data * f ) );

#ifdef SOFA_NEW_HEXA
                f = ( oneMinusFx * ( fy ) * oneMinusFz );
                o.addCol ( cube[3],  ( data * f ) );

                f = ( ( fx ) * ( fy ) * oneMinusFz );
                o.addCol ( cube[2],  ( data * f ) );

#else
                f = ( oneMinusFx * ( fy ) * oneMinusFz );
                o.addCol ( cube[2],  ( data * f ) );

                f = ( ( fx ) * ( fy ) * oneMinusFz );
                o.addCol ( cube[3],  ( data * f ) );

#endif
                f = ( oneMinusFx * oneMinusFy * ( fz ) );
                o.addCol ( cube[4],  ( data * f ) );

                f = ( ( fx ) * oneMinusFy * ( fz ) );
                o.addCol ( cube[5],  ( data * f ) );

#ifdef SOFA_NEW_HEXA
                f = ( oneMinusFx * ( fy ) * ( fz ) );
                o.addCol ( cube[7],  ( data * f ) );

                f = ( ( fx ) * ( fy ) * ( fz ) );
                o.addCol ( cube[6],  ( data * f ) );
#else
                f = ( oneMinusFx * ( fy ) * ( fz ) );
                o.addCol ( cube[6],  ( data * f ) );

                f = ( ( fx ) * ( fy ) * ( fz ) );
                o.addCol ( cube[7],  ( data * f ) );
#endif
            }
        }
    }
}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
{
    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Edge>& edges = this->fromTopology->getEdges();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();

        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());
            for ( ; colIt != colItEnd; ++colIt)
            {
                unsigned indexIn = colIt.index();
                InDeriv data = (InDeriv) Out::getDPos(colIt.val());

                const core::topology::BaseMeshTopology::Edge edge = edges[this->map.getValue()[indexIn].in_index];
                const OutReal fx = ( OutReal ) map.getValue()[indexIn].baryCoords[0];

                o.addCol ( edge[0], data * ( 1-fx ) );
                o.addCol ( edge[1], data * ( fx ) );
            }
        }
    }
}

template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
{
    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle>& triangles = this->fromTopology->getTriangles();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();

        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            for ( ; colIt != colItEnd; ++colIt)
            {
                unsigned indexIn = colIt.index();
                InDeriv data = (InDeriv) Out::getDPos(colIt.val());

                const core::topology::BaseMeshTopology::Triangle triangle = triangles[this->map.getValue()[indexIn].in_index];
                const OutReal fx = ( OutReal ) map.getValue()[indexIn].baryCoords[0];
                const OutReal fy = ( OutReal ) map.getValue()[indexIn].baryCoords[1];

                o.addCol (triangle[0],data * ( 1-fx-fy ) );
                o.addCol (triangle[1],data * ( fx ) );
                o.addCol (triangle[2],data * ( fy ) );
            }
        }
    }
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
{
    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Quad>& quads = this->fromTopology->getQuads();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();

        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            for ( ; colIt != colItEnd; ++colIt)
            {
                unsigned indexIn = colIt.index();
                InDeriv data = (InDeriv) Out::getDPos(colIt.val());

                const OutReal fx = ( OutReal ) map.getValue()[indexIn].baryCoords[0];
                const OutReal fy = ( OutReal ) map.getValue()[indexIn].baryCoords[1];
                const sofa::core::topology::BaseMeshTopology::Quad& quad = quads[map.getValue()[indexIn].in_index];

                o.addCol (quad[0], data * ( ( 1-fx ) * ( 1-fy ) ) );
                o.addCol (quad[1], data * ( ( fx ) * ( 1-fy ) ) );
                o.addCol (quad[3], data * ( ( 1-fx ) * ( fy ) ) );
                o.addCol (quad[2], data * ( ( fx ) * ( fy ) ) );
            }
        }
    }
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
{
    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Tetrahedron>& tetrahedra = this->fromTopology->getTetrahedra();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();

        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            for ( ; colIt != colItEnd; ++colIt)
            {
                unsigned indexIn = colIt.index();
                InDeriv data = (InDeriv) Out::getDPos(colIt.val());

                const OutReal fx = ( OutReal ) map.getValue()[indexIn].baryCoords[0];
                const OutReal fy = ( OutReal ) map.getValue()[indexIn].baryCoords[1];
                const OutReal fz = ( OutReal ) map.getValue()[indexIn].baryCoords[2];
                int index = map.getValue()[indexIn].in_index;
                const core::topology::BaseMeshTopology::Tetrahedron& tetra = tetrahedra[index];

                o.addCol (tetra[0], data * ( 1-fx-fy-fz ) );
                o.addCol (tetra[1], data * fx );
                o.addCol (tetra[2], data * fy );
                o.addCol (tetra[3], data * fz );
            }
        }
    }
}

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
{
    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();
    const sofa::helper::vector< core::topology::BaseMeshTopology::Hexahedron >& cubes = this->fromTopology->getHexahedra();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();

        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            for ( ; colIt != colItEnd; ++colIt)
            {
                unsigned indexIn = colIt.index();
                InDeriv data = (InDeriv) Out::getDPos(colIt.val());

                const OutReal fx = ( OutReal ) map.getValue()[indexIn].baryCoords[0];
                const OutReal fy = ( OutReal ) map.getValue()[indexIn].baryCoords[1];
                const OutReal fz = ( OutReal ) map.getValue()[indexIn].baryCoords[2];
                const OutReal oneMinusFx = 1-fx;
                const OutReal oneMinusFy = 1-fy;
                const OutReal oneMinusFz = 1-fz;

                int index = map.getValue()[indexIn].in_index;
                const core::topology::BaseMeshTopology::Hexahedron& cube = cubes[index];

                o.addCol (cube[0], data * ( oneMinusFx * oneMinusFy * oneMinusFz ) );
                o.addCol (cube[1], data * ( ( fx ) * oneMinusFy * oneMinusFz ) );
                o.addCol (cube[3], data * ( oneMinusFx * ( fy ) * oneMinusFz ) );
                o.addCol (cube[2], data * ( ( fx ) * ( fy ) * oneMinusFz ) );
                o.addCol (cube[4], data * ( oneMinusFx * oneMinusFy * ( fz ) ) );
                o.addCol (cube[5], data * ( ( fx ) * oneMinusFy * ( fz ) ) );
                o.addCol (cube[7], data * ( oneMinusFx * ( fy ) * ( fz ) ) );
                o.addCol (cube[6], data * ( ( fx ) * ( fy ) * ( fz ) ) );
            }
        }
    }
}

/************************************* Topological Changes ***********************************/

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::handleTopologyChange(core::topology::Topology* t)
{
    using sofa::core::behavior::MechanicalState;

    if (t != this->fromTopology) return;

    if ( this->fromTopology->beginChange() == this->fromTopology->endChange() )
        return;

    std::list<const core::topology::TopologyChange *>::const_iterator itBegin = this->fromTopology->beginChange();
    std::list<const core::topology::TopologyChange *>::const_iterator itEnd = this->fromTopology->endChange();

    for ( std::list<const core::topology::TopologyChange *>::const_iterator changeIt = itBegin;
            changeIt != itEnd; ++changeIt )
    {
        const core::topology::TopologyChangeType changeType = ( *changeIt )->getChangeType();
        switch ( changeType )
        {
            //TODO: implementation of BarycentricMapperHexahedronSetTopology<In,Out>::handleTopologyChange()
        case core::topology::ENDING_EVENT:       ///< To notify the end for the current sequence of topological change events
        {
            if(!_invalidIndex.empty())
            {
                helper::vector<MappingData>& mapData = *(map.beginEdit());

                for ( std::set<int>::const_iterator iter = _invalidIndex.begin();
                        iter != _invalidIndex.end(); ++iter )
                {
                    const int j = *iter;
                    if ( mapData[j].in_index == -1 ) // compute new mapping
                    {
                        sofa::defaulttype::Vector3 coefs;
                        typename In::Coord pos;
                        pos[0] = mapData[j].baryCoords[0];
                        pos[1] = mapData[j].baryCoords[1];
                        pos[2] = mapData[j].baryCoords[2];

                        // find nearest cell and barycentric coords
                        Real distance = 1e10;

                        int index = -1;
                        // When smoothing a mesh, the element has to be found using the rest position of the point. Then, its position is set using this element.
                        if( this->toTopology)
                        {
                            typedef MechanicalState<Out> MechanicalStateT;
                            MechanicalStateT* mState;
                            this->toTopology->getContext()->get( mState);
                            if( !mState)
                            {
                                msg_error() << "Can not find mechanical state." ;
                            }
                            else
                            {
                                const typename MechanicalStateT::VecCoord& xto0 = (mState->read(core::ConstVecCoordId::restPosition())->getValue());
                                index = _fromGeomAlgo->findNearestElementInRestPos ( Out::getCPos(xto0[j]), coefs, distance );
                                coefs = _fromGeomAlgo->computeHexahedronRestBarycentricCoeficients(index, pos);
                            }
                        }
                        else
                        {
                            index = _fromGeomAlgo->findNearestElementInRestPos ( pos, coefs, distance );
                        }

                        if ( index != -1 )
                        {
                            mapData[j].baryCoords[0] = ( Real ) coefs[0];
                            mapData[j].baryCoords[1] = ( Real ) coefs[1];
                            mapData[j].baryCoords[2] = ( Real ) coefs[2];
                            mapData[j].in_index = index;
                        }
                    }
                }

                map.endEdit();
                _invalidIndex.clear();
            }
        }
        break;
        case core::topology::POINTSINDICESSWAP:  ///< For PointsIndicesSwap.
            break;
        case core::topology::POINTSADDED:        ///< For PointsAdded.
            break;
        case core::topology::POINTSREMOVED:      ///< For PointsRemoved.
            break;
        case core::topology::POINTSRENUMBERING:  ///< For PointsRenumbering.
            break;
        case core::topology::TRIANGLESADDED:  ///< For Triangles Added.
            break;
        case core::topology::TRIANGLESREMOVED:  ///< For Triangles Removed.
            break;
        case core::topology::HEXAHEDRAADDED:     ///< For HexahedraAdded.
        {
        }
        break;
        case core::topology::HEXAHEDRAREMOVED:   ///< For HexahedraRemoved.
        {
            const unsigned int nbHexahedra = this->fromTopology->getNbHexahedra();

            const sofa::helper::vector<unsigned int> &hexahedra =
                    ( static_cast< const core::topology::HexahedraRemoved *> ( *changeIt ) )->getArray();
            //        sofa::helper::vector<unsigned int> hexahedra(tab);

            for ( unsigned int i=0; i<hexahedra.size(); ++i )
            {
                // remove all references to the removed cubes from the mapping data
                unsigned int cubeId = hexahedra[i];
                for ( unsigned int j=0; j<map.getValue().size(); ++j )
                {
                    if ( map.getValue()[j].in_index == ( int ) cubeId ) // invalidate mapping
                    {
                        sofa::defaulttype::Vector3 coefs;
                        coefs[0] = map.getValue()[j].baryCoords[0];
                        coefs[1] = map.getValue()[j].baryCoords[1];
                        coefs[2] = map.getValue()[j].baryCoords[2];

                        typename In::Coord restPos = _fromGeomAlgo->getRestPointPositionInHexahedron ( cubeId, coefs );

                        helper::vector<MappingData>& vectorData = *(map.beginEdit());
                        vectorData[j].in_index = -1;
                        vectorData[j].baryCoords[0] = restPos[0];
                        vectorData[j].baryCoords[1] = restPos[1];
                        vectorData[j].baryCoords[2] = restPos[2];
                        map.endEdit();

                        _invalidIndex.insert(j);
                    }
                }
            }

            // renumber
            unsigned int lastCubeId = nbHexahedra-1;
            for ( unsigned int i=0; i<hexahedra.size(); ++i, --lastCubeId )
            {
                unsigned int cubeId = hexahedra[i];
                for ( unsigned int j=0; j<map.getValue().size(); ++j )
                {
                    if ( map.getValue()[j].in_index == ( int ) lastCubeId )
                    {
                        helper::vector<MappingData>& vectorData = *(map.beginEdit());
                        vectorData[j].in_index = cubeId;
                        map.endEdit();
                    }
                }
            }
        }
        break;
        case core::topology::HEXAHEDRARENUMBERING: ///< For HexahedraRenumbering.
            break;
        default:
            break;
        }
    }
}

// handle topology changes depending on the topology
template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::handleTopologyChange ( core::topology::Topology* /*t*/ )
{
//    if (mapper)
//        mapper->handleTopologyChange(t);
    reinit(); // we now recompute the entire mapping when there is a topologychange
}

#ifdef BARYCENTRIC_MAPPER_TOPOCHANGE_REINIT
template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::handleTopologyChange(core::topology::Topology* t)
{
    using core::topology::TopologyChange;

    if (t != this->fromTopology) return;

    if ( this->fromTopology->beginChange() == this->fromTopology->endChange() )
        return;

    MechanicalState< In >* mStateFrom = NULL;
    MechanicalState< Out >* mStateTo = NULL;

    this->fromTopology->getContext()->get(mStateFrom);
    this->toTopology->getContext()->get(mStateTo);

    if ((mStateFrom == NULL) || (mStateTo == NULL))
        return;

    const typename MechanicalState< In >::VecCoord& in = *(mStateFrom->getX0());
    const typename MechanicalState< Out >::VecCoord& out = *(mStateTo->getX0());

	for (std::list< const TopologyChange *>::const_iterator it = this->fromTopology->beginChange(), itEnd = this->fromTopology->endChange(); it != itEnd; ++it)
	{
		const core::topology::TopologyChangeType& changeType = (*it)->getChangeType();

		switch ( changeType )
		{
        case core::topology::ENDING_EVENT :
        {
            const helper::vector< topology::Triangle >& triangles = this->fromTopology->getTriangles();
            helper::vector< Mat3x3d > bases;
            helper::vector< Vector3 > centers;

            // clear and reserve space for 2D mapping
            this->clear(out.size());
            bases.resize(triangles.size());
            centers.resize(triangles.size());

            for ( unsigned int t = 0; t < triangles.size(); t++ )
            {
                Mat3x3d m,mt;
                m[0] = in[triangles[t][1]]-in[triangles[t][0]];
                m[1] = in[triangles[t][2]]-in[triangles[t][0]];
                m[2] = cross ( m[0],m[1] );
                mt.transpose ( m );
                bases[t].invert ( mt );
                centers[t] = ( in[triangles[t][0]]+in[triangles[t][1]]+in[triangles[t][2]] ) /3;
            }

            for ( unsigned int i=0; i<out.size(); i++ )
            {
                Vec3d pos = Out::getCPos(out[i]);
                Vector3 coefs;
                int index = -1;
                double distance = 1e10;
                for ( unsigned int t = 0; t < triangles.size(); t++ )
                {
                    Vec3d v = bases[t] * ( pos - in[triangles[t][0]] );
                    double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( ( v[2]<0?-v[2]:v[2] )-0.01,v[0]+v[1]-1 ) );
                    if ( d>0 ) d = ( pos-centers[t] ).norm2();
                    if ( d<distance ) { coefs = v; distance = d; index = t; }
                }

                this->addPointInTriangle ( index, coefs.ptr() );
            }
            break;
        }
		default:
			break;
		}
	}
}

#endif // BARYCENTRIC_MAPPER_TOPOCHANGE_REINIT


template<class TIn, class TOut>
const helper::vector< defaulttype::BaseMatrix*>* BarycentricMapping<TIn, TOut>::getJs()
{
    typedef typename Mapper::MatrixType mat_type;
    const sofa::defaulttype::BaseMatrix* matJ = getJ();

    const mat_type* mat = dynamic_cast<const mat_type*>(matJ);
    assert( mat );

    eigen.copyFrom( *mat );   // woot

    js.resize( 1 );
    js[0] = &eigen;
    return &js;
}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::updateForceMask()
{
    if( mapper )
        mapper->updateForceMask();
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
