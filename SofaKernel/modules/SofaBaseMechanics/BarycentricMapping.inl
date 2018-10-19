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

namespace sofa
{

namespace component
{

namespace mapping
{

using sofa::defaulttype::Vector3;
using sofa::defaulttype::Matrix3;
using sofa::defaulttype::Mat3x3d;
using sofa::defaulttype::Vec3d;
// 10/18 E.Coevoet: what's the difference between edge/line, tetra/tetrahedron, hexa/hexahedron?
typedef typename sofa::core::topology::BaseMeshTopology::Line Line;
typedef typename sofa::core::topology::BaseMeshTopology::Edge Edge;
typedef typename sofa::core::topology::BaseMeshTopology::Triangle Triangle;
typedef typename sofa::core::topology::BaseMeshTopology::Quad Quad;
typedef typename sofa::core::topology::BaseMeshTopology::Tetra Tetra;
typedef typename sofa::core::topology::BaseMeshTopology::Tetrahedron Tetrahedron;
typedef typename sofa::core::topology::BaseMeshTopology::Hexa Hexa;
typedef typename sofa::core::topology::BaseMeshTopology::Hexahedron Hexahedron;
typedef typename sofa::core::topology::BaseMeshTopology::SeqLines SeqLines;
typedef typename sofa::core::topology::BaseMeshTopology::SeqEdges SeqEdges;
typedef typename sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
typedef typename sofa::core::topology::BaseMeshTopology::SeqQuads SeqQuads;
typedef typename sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
typedef typename sofa::core::topology::BaseMeshTopology::SeqHexahedra SeqHexahedra;

template <class TIn, class TOut>
BarycentricMapping<TIn, TOut>::BarycentricMapping()
    : Inherit()
    , m_mapper(initLink("mapper","Internal mapper created depending on the type of topology"))
    , useRestPosition(core::objectmodel::Base::initData(&useRestPosition, false, "useRestPosition", "Use the rest position of the input and output models to initialize the mapping"))
#ifdef SOFA_DEV
    , sleeping(core::objectmodel::Base::initData(&sleeping, false, "sleeping", "is the mapping sleeping (not computed)"))
#endif
{
}

template <class TIn, class TOut>
BarycentricMapping<TIn, TOut>::BarycentricMapping(core::State<In>* from, core::State<Out>* to, typename Mapper::SPtr mapper)
    : Inherit ( from, to )
    , m_mapper(initLink("mapper","Internal mapper created depending on the type of topology"), mapper)
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
    , m_mapper (initLink("mapper","Internal mapper created depending on the type of topology"))
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
void BarycentricMapperRegularGridTopology<In,Out>::clearMapAndReserve ( int size )
{
    updateJ = true;
    m_map.clear();
    if ( size>0 ) m_map.reserve ( size );
}

template <class In, class Out>
int BarycentricMapperRegularGridTopology<In,Out>::addPointInCube ( const int cubeIndex, const SReal* baryCoords )
{
    m_map.resize ( m_map.size() +1 );
    CubeData& data = *m_map.rbegin();
    data.in_index = cubeIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    data.baryCoords[2] = ( Real ) baryCoords[2];
    return (int)m_map.size()-1;
}

template <class In, class Out>
void BarycentricMapperRegularGridTopology<In,Out>::init ( const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    SOFA_UNUSED(in);

    updateJ = true;

    clearMapAndReserve ( (int)out.size() );
    if ( m_fromTopology->isVolume() )
    {
        for ( unsigned int i=0; i<out.size(); i++ )
        {
            Vector3 coefs;
            int cube = m_fromTopology->findCube ( Vector3 ( Out::getCPos(out[i]) ), coefs[0], coefs[1], coefs[2] );
            if ( cube==-1 )
                cube = m_fromTopology->findNearestCube ( Vector3 ( Out::getCPos(out[i]) ), coefs[0], coefs[1], coefs[2] );

            this->addPointInCube ( cube, coefs.ptr() );
        }
    }
}



template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::clearMapAndReserve ( int size )
{
    updateJ = true;
    m_map.clear();
    if ( size>0 ) m_map.reserve ( size );
}

template <class In, class Out>
int BarycentricMapperSparseGridTopology<In,Out>::addPointInCube ( const int cubeIndex, const SReal* baryCoords )
{
    m_map.resize ( m_map.size() +1 );
    CubeData& data = *m_map.rbegin();
    data.in_index = cubeIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    data.baryCoords[2] = ( Real ) baryCoords[2];
    return (int)m_map.size()-1;
}

template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::init ( const typename Out::VecCoord& out, const typename In::VecCoord& /*in*/ )
{
    if ( this->m_map.size() != 0 ) return;
    updateJ = true;
    clearMapAndReserve ( (int)out.size() );

    if ( m_fromTopology->isVolume() )
    {
        for ( unsigned int i=0; i<out.size(); i++ )
        {
            Vector3 coefs;
            int cube = m_fromTopology->findCube ( Vector3 ( Out::getCPos(out[i]) ), coefs[0], coefs[1], coefs[2] );
            if ( cube==-1 )
            {
                cube = m_fromTopology->findNearestCube ( Vector3 ( Out::getCPos(out[i]) ), coefs[0], coefs[1], coefs[2] );
            }
            Vector3 baryCoords = coefs;
            this->addPointInCube ( cube, baryCoords.ptr() );
        }
    }
}



template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::clearMap1dAndReserve ( int size )
{
    m_updateJ = true;
    m_map1d.clear();
    if ( size>0 ) m_map1d.reserve ( size );
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::clearMap2dAndReserve ( int size )
{
    m_updateJ = true;
    m_map2d.clear();
    if ( size>0 ) m_map2d.reserve ( size );
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::clearMap3dAndReserve ( int size )
{
    m_updateJ = true;
    m_map3d.clear();
    if ( size>0 ) m_map3d.reserve ( size );
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::clearMapAndReserve ( int size )
{
    m_updateJ = true;
    clearMap1dAndReserve(size);
    clearMap2dAndReserve(size);
    clearMap3dAndReserve(size);
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::addPointInLine ( const int lineIndex, const SReal* baryCoords )
{
    m_map1d.resize ( m_map1d.size() +1 );
    MappingData1D& data = *m_map1d.rbegin();
    data.in_index = lineIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    return (int)m_map1d.size()-1;
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::addPointInTriangle ( const int triangleIndex, const SReal* baryCoords )
{
    m_map2d.resize ( m_map2d.size() +1 );
    MappingData2D& data = *m_map2d.rbegin();
    data.in_index = triangleIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    return (int)m_map2d.size()-1;
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::addPointInQuad ( const int quadIndex, const SReal* baryCoords )
{
    m_map2d.resize ( m_map2d.size() +1 );
    MappingData2D& data = *m_map2d.rbegin();
    data.in_index = quadIndex + this->m_fromTopology->getNbTriangles();
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    return (int)m_map2d.size()-1;
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::addPointInTetra ( const int tetraIndex, const SReal* baryCoords )
{
    m_map3d.resize ( m_map3d.size() +1 );
    MappingData3D& data = *m_map3d.rbegin();
    data.in_index = tetraIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    data.baryCoords[2] = ( Real ) baryCoords[2];
    return (int)m_map3d.size()-1;
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::addPointInCube ( const int cubeIndex, const SReal* baryCoords )
{
    m_map3d.resize ( m_map3d.size() +1 );
    MappingData3D& data = *m_map3d.rbegin();
    data.in_index = cubeIndex + this->m_fromTopology->getNbTetrahedra();
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    data.baryCoords[2] = ( Real ) baryCoords[2];
    return (int)m_map3d.size()-1;
}

template <class In, class Out>
int BarycentricMapperMeshTopology<In,Out>::createPointInLine ( const typename Out::Coord& p, int lineIndex, const typename In::VecCoord* points )
{
    SReal baryCoords[1];
    const Line& elem = this->m_fromTopology->getLine ( lineIndex );
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
    const Triangle& elem = this->m_fromTopology->getTriangle ( triangleIndex );

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
    const Quad& elem = this->m_fromTopology->getQuad ( quadIndex );
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
    m_updateJ = true;

    const SeqTetrahedra& tetras = this->m_fromTopology->getTetrahedra();
    const SeqHexahedra& hexas = this->m_fromTopology->getHexahedra();

    const SeqTriangles& triangles = this->m_fromTopology->getTriangles();
    const SeqQuads& quads = this->m_fromTopology->getQuads();
    helper::vector<Matrix3> bases;
    helper::vector<Vector3> centers;
    if ( tetras.empty() && hexas.empty() )
    {
        if ( triangles.empty() && quads.empty() )
        {
            const SeqEdges& edges = this->m_fromTopology->getEdges();
            if ( edges.empty() ) return;

            clearMap1dAndReserve ( (int)out.size() );

            helper::vector< SReal >   lengthEdges;
            helper::vector< Vector3 > unitaryVectors;

            unsigned int e;
            for ( e=0; e<edges.size(); e++ )
            {
                lengthEdges.push_back ( ( in[edges[e][1]]-in[edges[e][0]] ).norm() );

                Vector3 V12 = ( in[edges[e][1]]-in[edges[e][0]] ); V12.normalize();
                unitaryVectors.push_back ( V12 );
            }

            for ( unsigned int i=0; i<out.size(); i++ )
            {
                SReal coef=0;
                for ( e=0; e<edges.size(); e++ )
                {
                    SReal lengthEdge = lengthEdges[e];
                    Vector3 V12 =unitaryVectors[e];

                    coef = ( V12 ) * Vector3 ( Out::getCPos(out[i])-in[edges[e][0]] ) /lengthEdge;
                    if ( coef >= 0 && coef <= 1 ) {addPointInLine ( e,&coef );  break; }
                }
                //If no good coefficient has been found, we add to the last element
                if ( e == edges.size() ) addPointInLine ( (int)edges.size()-1,&coef );
            }
        }
        else
        {
            clearMap2dAndReserve ( (int)out.size() );
            size_t nbTriangles = triangles.size();
            bases.resize ( triangles.size() +quads.size() );
            centers.resize ( triangles.size() +quads.size() );
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
            for ( unsigned int q = 0; q < quads.size(); q++ )
            {
                Mat3x3d m,mt;
                m[0] = in[quads[q][1]]-in[quads[q][0]];
                m[1] = in[quads[q][3]]-in[quads[q][0]];
                m[2] = cross ( m[0],m[1] );
                mt.transpose ( m );
                bases[nbTriangles+q].invert ( mt );
                centers[nbTriangles+q] = ( in[quads[q][0]]+in[quads[q][1]]+in[quads[q][2]]+in[quads[q][3]] ) *0.25;
            }
            for ( unsigned int i=0; i<out.size(); i++ )
            {
                Vector3 outPos = Out::getCPos(out[i]);
                Vector3 coefs;
                int index = -1;
                double distance = 1e10;
                for ( unsigned int t = 0; t < triangles.size(); t++ )
                {
                    Vec3d v = bases[t] * ( outPos - in[triangles[t][0]] );
                    double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( ( v[2]<0?-v[2]:v[2] )-0.01,v[0]+v[1]-1 ) );
                    if ( d>0 ) d = ( outPos-centers[t] ).norm2();
                    if ( d<distance ) { coefs = v; distance = d; index = t; }
                }
                for ( unsigned int q = 0; q < quads.size(); q++ )
                {
                    Vec3d v = bases[nbTriangles+q] * ( outPos - in[quads[q][0]] );
                    double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( v[1]-1,v[0]-1 ),std::max ( v[2]-0.01,-v[2]-0.01 ) ) );
                    if ( d>0 ) d = ( outPos-centers[nbTriangles+q] ).norm2();
                    if ( d<distance ) { coefs = v; distance = d; index = nbTriangles+q; }
                }
                if ( index < (int)nbTriangles )
                    addPointInTriangle ( index, coefs.ptr() );
                else
                    addPointInQuad ( index-nbTriangles, coefs.ptr() );
            }
        }
    }
    else
    {
        clearMap3dAndReserve ( out.size() );
        int nbTetras = tetras.size();
        bases.resize ( tetras.size() + hexas.size() );
        centers.resize ( tetras.size() + hexas.size() );
        for ( unsigned int t = 0; t < tetras.size(); t++ )
        {
            Mat3x3d m,mt;
            m[0] = in[tetras[t][1]]-in[tetras[t][0]];
            m[1] = in[tetras[t][2]]-in[tetras[t][0]];
            m[2] = in[tetras[t][3]]-in[tetras[t][0]];
            mt.transpose ( m );
            bases[t].invert ( mt );
            centers[t] = ( in[tetras[t][0]]+in[tetras[t][1]]+in[tetras[t][2]]+in[tetras[t][3]] ) *0.25;
        }
        for ( unsigned int h = 0; h < hexas.size(); h++ )
        {
            Mat3x3d m,mt;
            m[0] = in[hexas[h][1]]-in[hexas[h][0]];
            m[1] = in[hexas[h][3]]-in[hexas[h][0]];
            m[2] = in[hexas[h][4]]-in[hexas[h][0]];
            mt.transpose ( m );
            bases[nbTetras+h].invert ( mt );
            centers[nbTetras+h] = ( in[hexas[h][0]]+in[hexas[h][1]]+in[hexas[h][2]]+in[hexas[h][3]]+in[hexas[h][4]]+in[hexas[h][5]]+in[hexas[h][6]]+in[hexas[h][7]] ) *0.125;
        }
        for ( unsigned int i=0; i<out.size(); i++ )
        {
            Vector3 pos = Out::getCPos(out[i]);
            Vector3 coefs;
            int index = -1;
            double distance = 1e10;
            for ( unsigned int t = 0; t < tetras.size(); t++ )
            {
                Vector3 v = bases[t] * ( pos - in[tetras[t][0]] );
                double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( -v[2],v[0]+v[1]+v[2]-1 ) );
                if ( d>0 ) d = ( pos-centers[t] ).norm2();
                if ( d<distance ) { coefs = v; distance = d; index = t; }
            }
            for ( unsigned int h = 0; h < hexas.size(); h++ )
            {
                Vector3 v = bases[nbTetras+h] * ( pos - in[hexas[h][0]] );
                double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( -v[2],v[0]-1 ),std::max ( v[1]-1,v[2]-1 ) ) );
                if ( d>0 ) d = ( pos-centers[nbTetras+h] ).norm2();
                if ( d<distance ) { coefs = v; distance = d; index = nbTetras+h; }
            }
            if ( index < nbTetras )
                addPointInTetra ( index, coefs.ptr() );
            else
                addPointInCube ( index-nbTetras, coefs.ptr() );
        }
    }
}



template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::clearMapAndReserve(int size)
{
    helper::vector<MappingDataType>& vectorData = *(d_map.beginEdit());
    vectorData.clear();
    if ( size>0 ) vectorData.reserve ( size );
    d_map.endEdit();
}

template <class In, class Out, class MappingDataType, class Element>
int BarycentricMapperEdgeSetTopology<In,Out,MappingDataType,Element>::addPointInLine ( const int edgeIndex, const SReal* baryCoords )
{
    helper::vector<MappingData>& vectorData = *(d_map.beginEdit());
    vectorData.resize ( d_map.getValue().size() +1 );
    d_map.endEdit();
    MappingData& data = *vectorData.rbegin();
    data.in_index = edgeIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    return (int)d_map.getValue().size()-1;
}

template <class In, class Out, class MappingDataType, class Element>
int BarycentricMapperEdgeSetTopology<In,Out,MappingDataType,Element>::createPointInLine ( const typename Out::Coord& p, int edgeIndex, const typename In::VecCoord* points )
{
    SReal baryCoords[1];
    const Edge& elem = this->m_fromTopology->getEdge ( edgeIndex );
    const typename In::Coord p0 = ( *points ) [elem[0]];
    const typename In::Coord pA = ( *points ) [elem[1]] - p0;
    typename In::Coord pos = Out::getCPos(p) - p0;
    baryCoords[0] = dot ( pA,pos ) /dot ( pA,pA );
    return this->addPointInLine ( edgeIndex, baryCoords );
}

template <class In, class Out, class MappingDataType, class Element>
int BarycentricMapperTriangleSetTopology<In,Out,MappingDataType,Element>::addPointInTriangle ( const int triangleIndex, const SReal* baryCoords )
{
    helper::vector<MappingData>& vectorData = *(d_map.beginEdit());
    vectorData.resize ( d_map.getValue().size() +1 );
    MappingData& data = *vectorData.rbegin();
    d_map.endEdit();
    data.in_index = triangleIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    return (int)d_map.getValue().size()-1;
}

template <class In, class Out, class MappingDataType, class Element>
int BarycentricMapperTriangleSetTopology<In,Out,MappingDataType,Element>::createPointInTriangle ( const typename Out::Coord& p, int triangleIndex, const typename In::VecCoord* points )
{
    SReal baryCoords[2];
    const Triangle& elem = this->m_fromTopology->getTriangle ( triangleIndex );
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

template <class In, class Out, class MappingDataType, class Element>
int BarycentricMapperQuadSetTopology<In,Out,MappingDataType,Element>::addPointInQuad ( const int quadIndex, const SReal* baryCoords )
{
    helper::vector<MappingData>& vectorData = *(d_map.beginEdit());
    vectorData.resize ( d_map.getValue().size() +1 );
    MappingData& data = *vectorData.rbegin();
    d_map.endEdit();
    data.in_index = quadIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    return (int)d_map.getValue().size()-1;
}

template <class In, class Out, class MappingDataType, class Element>
int BarycentricMapperQuadSetTopology<In,Out,MappingDataType,Element>::createPointInQuad ( const typename Out::Coord& p, int quadIndex, const typename In::VecCoord* points )
{
    SReal baryCoords[2];
    const Quad& elem = this->m_fromTopology->getQuad ( quadIndex );
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

template <class In, class Out, class MappingDataType, class Element>
int BarycentricMapperTetrahedronSetTopology<In,Out,MappingDataType,Element>::addPointInTetra ( const int tetraIndex, const SReal* baryCoords )
{
    helper::vector<MappingData>& vectorData = *(d_map.beginEdit());
    vectorData.resize ( d_map.getValue().size() +1 );
    MappingData& data = *vectorData.rbegin();
    d_map.endEdit();
    data.in_index = tetraIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    data.baryCoords[2] = ( Real ) baryCoords[2];
    return (int)d_map.getValue().size()-1;
}

template <class In, class Out, class MappingDataType, class Element>
int BarycentricMapperHexahedronSetTopology<In,Out,MappingDataType,Element>::addPointInCube ( const int cubeIndex, const SReal* baryCoords )
{
    helper::vector<MappingData>& vectorData = *(d_map.beginEdit());
    vectorData.resize ( d_map.getValue().size() +1 );
    MappingData& data = *vectorData.rbegin();
    d_map.endEdit();
    data.in_index = cubeIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    data.baryCoords[2] = ( Real ) baryCoords[2];
    return (int)d_map.getValue().size()-1;
}

template <class In, class Out, class MappingDataType, class Element>
int BarycentricMapperHexahedronSetTopology<In,Out,MappingDataType,Element>::setPointInCube ( const int pointIndex, const int cubeIndex, const SReal* baryCoords )
{
    if ( pointIndex >= ( int ) d_map.getValue().size() )
        return -1;

    helper::vector<MappingData>& vectorData = *(d_map.beginEdit());
    MappingData& data = vectorData[pointIndex];
    data.in_index = cubeIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    data.baryCoords[2] = ( Real ) baryCoords[2];
    d_map.endEdit();

    if(cubeIndex == -1)
        m_invalidIndex.insert(pointIndex);
    else
        m_invalidIndex.erase(pointIndex);

    return pointIndex;
}

template <class In, class Out, class MappingDataType, class Element>

void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::initHashing( const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    computeHashingCellSize(in);
    computeBB(out,in);
    computeHashTable(in);
}

template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::computeHashingCellSize(const typename In::VecCoord& in )
{
    // The grid cell size is set to the average edge length of all elements
    const SeqEdges& edges = m_fromTopology->getEdges();
    Real averageLength=0.;

    if(edges.size()>0)
    {
        for(unsigned int i=0; i<edges.size(); i++)
        {
            Edge edge = edges[i];
            averageLength += (in[edge[0]]-in[edge[1]]).norm();
        }
        averageLength/=(Real)edges.size();

    }
    else
    {
        const helper::vector<Element>& elements = getElements();

        for(unsigned int i=0; i<elements.size(); i++)
        {
            Element element = elements[i];
            averageLength += (in[element[0]]-in[element[1]]).norm();
        }
        averageLength/=(Real)elements.size();
    }

    m_gridCellSize = averageLength;
    m_convFactor = 1./(Real)m_gridCellSize;
}

template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::computeBB( const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    Vector3 BBmin1 = {1e10,1e10,1e10}, BBmin2 = {1e10,1e10,1e10};
    Vector3 BBmax1 = {-1e10,-1e10,-1e10}, BBmax2 = {-1e10,-1e10,-1e10};

    for(unsigned int i=0; i<in.size(); i++)
    {
        for(int k=0; k<3; k++)
        {
            if(in[i][k]<BBmin1[k]) BBmin1[k]=in[i][k];
            if(in[i][k]>BBmax1[k]) BBmax1[k]=in[i][k];
        }
    }

    for(unsigned int i=0; i<out.size(); i++)
    {
        for(int k=0; k<3; k++)
        {
            if(out[i][k]<BBmin2[k]) BBmin2[k]=out[i][k];
            if(out[i][k]>BBmax2[k]) BBmax2[k]=out[i][k];
        }
    }

    m_computeDistances = false;
    for(int k=0; k<3; k++)
    {
        if ((BBmin1[k]<=BBmax2[k] && BBmin1[k]>=BBmin2[k]) || (BBmax1[k]<=BBmax2[k] && BBmax1[k]>=BBmin2[k]))
            m_computeDistances = true;

        if ((BBmin2[k]<=BBmax1[k] && BBmin2[k]>=BBmin1[k]) || (BBmax2[k]<=BBmax1[k] && BBmax2[k]>=BBmin1[k]))
            m_computeDistances = true;
    }
}

template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::computeHashTable( const typename In::VecCoord& in )
{
    const helper::vector<Element>& elements = getElements();
    m_hashTableSize = elements.size()*2; // Next prime number would be better
    m_hashTable.clear();
    m_hashTable.resize(m_hashTableSize);
    for (unsigned int i=0; i<m_hashTableSize; i++)
        m_hashTable[i].clear();

    for(unsigned int i=0; i<elements.size(); i++)
    {
        Element element = elements[i];
        Vector3 min=in[element[0]], max=in[element[0]];

        for(unsigned int j=0; j<element.size(); j++)
        {
            for(int k=0; k<3; k++)
            {
                if(in[element[j]][k]<min[k]) min[k]=in[element[j]][k];
                if(in[element[j]][k]>max[k]) max[k]=in[element[j]][k];
            }
        }

        Vec3i i_min=getGridIndices(min);
        Vec3i i_max=getGridIndices(max);

        for(int j=i_min[0]; j<=i_max[0]; j++)
            for(int k=i_min[1]; k<=i_max[1]; k++)
                for(int l=i_min[2]; l<=i_max[2]; l++)
                {
                    unsigned int h = getHashIndexFromIndices(j,k,l);
                    addToHashTable(h, i);
                }
    }
}

template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::init ( const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    initHashing(out,in);
    const helper::vector<Element>& elements = getElements();
    helper::vector<Mat3x3d> bases;
    helper::vector<Vector3> centers;

    this->clearMapAndReserve ( (int)out.size() );
    bases.resize ( elements.size() );
    centers.resize ( elements.size() );

    bool wrongMapping = false;
    if(m_computeDistances)
    {
        // Compute bases and centers of each element
        for ( unsigned int e = 0; e < elements.size(); e++ )
        {
            Element element = elements[e];

            Mat3x3d base;
            computeBase(base,in,element);
            bases[e] = base;

            Vector3 center;
            computeCenter(center,in,element);
            centers[e] = center;
        }

        // Compute distances to get nearest element and corresponding bary coef
        for ( unsigned int i=0; i<out.size(); i++ )
        {
            Vec3d outPos = Out::getCPos(out[i]);
            Vector3 baryCoords;
            int elementIndex = -1;
            double distance = 1e10;

            unsigned int h = getHashIndexFromCoord(outPos);
            for ( unsigned int j=0; j<m_hashTable[h].size(); j++)
            {
                int e = m_hashTable[h][j];
                Vec3d bary = bases[e] * ( outPos - in[elements[e][0]] );
                double dist;
                computeDistance(dist, bary);
                if ( dist>0 )
                    dist = ( outPos-centers[e] ).norm2();
                if ( dist<distance )
                {
                    baryCoords = bary;
                    distance = dist;
                    elementIndex = e;
                }
            }

            if(elementIndex==-1)
            {
                baryCoords = Vector3{0.,0.,0.};
                wrongMapping = true;
                addPointInElement(elements.size(), baryCoords.ptr());
            }
            else
                addPointInElement(elementIndex, baryCoords.ptr());
        }

        if(wrongMapping)
            msg_warning() << "Some points seem to be away from the model their should be mapped on. The mapping may act wrong.";
    }
    else
    {
        msg_warning() << "The two models you are trying to map are far from each other. The mapping will be wrong.";

        //Bounding box of each object do not intersect: avoid expensive computations and map each point to last element
        Vector3 baryCoords{0.,0.,0.};
        for ( unsigned int i=0; i<out.size(); i++ )
            addPointInElement(elements.size(), baryCoords.ptr());
    }
}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::createMapperFromTopology ( BaseMeshTopology * topology )
{
    using sofa::core::behavior::BaseMechanicalState;

    m_mapper = NULL;

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
            m_mapper = sofa::core::objectmodel::New<HexahedronSetMapper>(t1, toTopoCont);
        }
        else
        {
            topology::TetrahedronSetTopologyContainer* t2 = dynamic_cast<topology::TetrahedronSetTopologyContainer*>(fromTopoCont);
            if (t2 != NULL)
            {
                typedef BarycentricMapperTetrahedronSetTopology<InDataTypes, OutDataTypes> TetrahedronSetMapper;
                m_mapper = sofa::core::objectmodel::New<TetrahedronSetMapper>(t2, toTopoCont);
            }
            else
            {
                topology::QuadSetTopologyContainer* t3 = dynamic_cast<topology::QuadSetTopologyContainer*>(fromTopoCont);
                if (t3 != NULL)
                {
                    typedef BarycentricMapperQuadSetTopology<InDataTypes, OutDataTypes> QuadSetMapper;
                    m_mapper = sofa::core::objectmodel::New<QuadSetMapper>(t3, toTopoCont);
                }
                else
                {
                    topology::TriangleSetTopologyContainer* t4 = dynamic_cast<topology::TriangleSetTopologyContainer*>(fromTopoCont);
                    if (t4 != NULL)
                    {
                        typedef BarycentricMapperTriangleSetTopology<InDataTypes, OutDataTypes> TriangleSetMapper;
                        m_mapper = sofa::core::objectmodel::New<TriangleSetMapper>(t4, toTopoCont);
                    }
                    else
                    {
                        topology::EdgeSetTopologyContainer* t5 = dynamic_cast<topology::EdgeSetTopologyContainer*>(fromTopoCont);
                        if ( t5 != NULL )
                        {
                            typedef BarycentricMapperEdgeSetTopology<InDataTypes, OutDataTypes> EdgeSetMapper;
                            m_mapper = sofa::core::objectmodel::New<EdgeSetMapper>(t5, toTopoCont);
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

            m_mapper = sofa::core::objectmodel::New<RegularGridMapper>(rgt, toTopoCont);
        }
        else
        {
            using sofa::component::topology::SparseGridTopology;

            SparseGridTopology* sgt = dynamic_cast< SparseGridTopology* >(topology);
            if (sgt != NULL && sgt->isVolume())
            {
                typedef BarycentricMapperSparseGridTopology< InDataTypes, OutDataTypes > SparseGridMapper;
                m_mapper = sofa::core::objectmodel::New<SparseGridMapper>(sgt, toTopoCont);
            }
            else // generic MeshTopology
            {
                using sofa::core::topology::BaseMeshTopology;

                typedef BarycentricMapperMeshTopology< InDataTypes, OutDataTypes > MeshMapper;
                BaseMeshTopology* bmt = dynamic_cast< BaseMeshTopology* >(topology);
                m_mapper = sofa::core::objectmodel::New<MeshMapper>(bmt, toTopoCont);
            }
        }
    }
    if (m_mapper)
    {
        m_mapper->setName("mapper");
        this->addSlave(m_mapper.get());
        m_mapper->maskFrom = this->maskFrom;
        m_mapper->maskTo = this->maskTo;
    }
}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::init()
{
    topology_from = this->fromModel->getContext()->getMeshTopology();
    topology_to = this->toModel->getContext()->getMeshTopology();

    Inherit::init();

    if ( m_mapper == NULL ) // try to create a mapper according to the topology of the In model
    {
        if ( topology_from!=NULL )
        {
            createMapperFromTopology ( topology_from );
        }
    }

    if ( m_mapper != NULL )
    {
        if (useRestPosition.getValue())
            m_mapper->init ( ((const core::State<Out> *)this->toModel)->read(core::ConstVecCoordId::restPosition())->getValue(), ((const core::State<In> *)this->fromModel)->read(core::ConstVecCoordId::restPosition())->getValue() );
        else
            m_mapper->init (((const core::State<Out> *)this->toModel)->read(core::ConstVecCoordId::position())->getValue(), ((const core::State<In> *)this->fromModel)->read(core::ConstVecCoordId::position())->getValue() );
    }
    else
    {
        msg_error() << "Barycentric mapping does not understand topology.";
    }

}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::reinit()
{
    if ( m_mapper != NULL )
    {
        m_mapper->clearMapAndReserve();
        m_mapper->init (((const core::State<Out> *)this->toModel)->read(core::ConstVecCoordId::position())->getValue(), ((const core::State<In> *)this->fromModel)->read(core::ConstVecCoordId::position())->getValue() );
    }
}



/************************************* Apply and Resize ***********************************/



template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::apply(const core::MechanicalParams * mparams, Data< typename Out::VecCoord >& out, const Data< typename In::VecCoord >& in)
{
    SOFA_UNUSED(mparams);

    if (
#ifdef SOFA_DEV
        sleeping.getValue()==false &&
#endif
        m_mapper != NULL)
    {
        m_mapper->resize( this->toModel );
        m_mapper->apply(*out.beginWriteOnly(), in.getValue());
        out.endEdit();
    }
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::resize( core::State<Out>* toModel )
{
    toModel->resize(m_map1d.size() +m_map2d.size() +m_map3d.size());
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize( m_map1d.size() +m_map2d.size() +m_map3d.size() );

    const SeqLines& lines = this->m_fromTopology->getLines();
    const SeqTriangles& triangles = this->m_fromTopology->getTriangles();
    const SeqQuads& quads = this->m_fromTopology->getQuads();
    const SeqTetrahedra& tetrahedra = this->m_fromTopology->getTetrahedra();
    const SeqHexahedra& cubes = this->m_fromTopology->getHexahedra();

    // 1D elements
    {
        for ( unsigned int i=0; i<m_map1d.size(); i++ )
        {
            const Real fx = m_map1d[i].baryCoords[0];
            int index = m_map1d[i].in_index;
            {
                const Line& line = lines[index];
                Out::setCPos(out[i] , in[line[0]] * ( 1-fx )
                        + in[line[1]] * fx );
            }
        }
    }
    // 2D elements
    {
        const int i0 = m_map1d.size();
        const int c0 = triangles.size();
        for ( unsigned int i=0; i<m_map2d.size(); i++ )
        {
            const Real fx = m_map2d[i].baryCoords[0];
            const Real fy = m_map2d[i].baryCoords[1];
            int index = m_map2d[i].in_index;
            if ( index<c0 )
            {
                const Triangle& triangle = triangles[index];
                Out::setCPos(out[i+i0] , in[triangle[0]] * ( 1-fx-fy )
                        + in[triangle[1]] * fx
                        + in[triangle[2]] * fy );
            }
            else
            {
                if (quads.size())
                {
                    const Quad& quad = quads[index-c0];
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
        const int i0 = m_map1d.size() + m_map2d.size();
        const int c0 = tetrahedra.size();
        for ( unsigned int i=0; i<m_map3d.size(); i++ )
        {
            const Real fx = m_map3d[i].baryCoords[0];
            const Real fy = m_map3d[i].baryCoords[1];
            const Real fz = m_map3d[i].baryCoords[2];
            int index = m_map3d[i].in_index;
            if ( index<c0 )
            {
                const Tetra& tetra = tetrahedra[index];
                Out::setCPos(out[i+i0] , in[tetra[0]] * ( 1-fx-fy-fz )
                        + in[tetra[1]] * fx
                        + in[tetra[2]] * fy
                        + in[tetra[3]] * fz );
            }
            else
            {
                const Hexa& cube = cubes[index-c0];

                Out::setCPos(out[i+i0] , in[cube[0]] * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) )
                        + in[cube[1]] * ( ( fx ) * ( 1-fy ) * ( 1-fz ) )
                        + in[cube[3]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                        + in[cube[2]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
                        + in[cube[4]] * ( ( 1-fx ) * ( 1-fy ) * ( fz ) )
                        + in[cube[5]] * ( ( fx ) * ( 1-fy ) * ( fz ) )
                        + in[cube[7]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                        + in[cube[6]] * ( ( fx ) * ( fy ) * ( fz ) ) );
            }
        }
    }
}

template <class In, class Out>
void BarycentricMapperRegularGridTopology<In,Out>::resize( core::State<Out>* toModel )
{
    toModel->resize(m_map.size());
}

template <class In, class Out>
void BarycentricMapperRegularGridTopology<In,Out>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize( m_map.size() );

    for ( unsigned int i=0; i<m_map.size(); i++ )
    {
        const topology::RegularGridTopology::Hexa cube = this->m_fromTopology->getHexaCopy ( this->m_map[i].in_index );

        const Real fx = m_map[i].baryCoords[0];
        const Real fy = m_map[i].baryCoords[1];
        const Real fz = m_map[i].baryCoords[2];
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

template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::resize( core::State<Out>* toModel )
{
    toModel->resize(m_map.size());
}

template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize( m_map.size() );

    typedef helper::vector< CubeData > CubeDataVector;
    typedef typename CubeDataVector::const_iterator CubeDataVectorIt;

    CubeDataVectorIt it = m_map.begin();
    CubeDataVectorIt itEnd = m_map.end();

    unsigned int i = 0;

    while (it != itEnd)
    {
        const topology::SparseGridTopology::Hexa cube = this->m_fromTopology->getHexahedron( it->in_index );

        const Real fx = it->baryCoords[0];
        const Real fy = it->baryCoords[1];
        const Real fz = it->baryCoords[2];

        Out::setCPos(out[i] , in[cube[0]] * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) )
                + in[cube[1]] * ( ( fx ) * ( 1-fy ) * ( 1-fz ) )
                + in[cube[3]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                + in[cube[2]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
                + in[cube[4]] * ( ( 1-fx ) * ( 1-fy ) * ( fz ) )
                + in[cube[5]] * ( ( fx ) * ( 1-fy ) * ( fz ) )
                + in[cube[7]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                + in[cube[6]] * ( ( fx ) * ( fy ) * ( fz ) ) );

        ++it;
        ++i;
    }
}

template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize( d_map.getValue().size() );

    const helper::vector<Element>& elements = getElements();
    for ( unsigned int i=0; i<d_map.getValue().size(); i++ )
    {
        int index = d_map.getValue()[i].in_index;
        const Element& element = elements[index];

        helper::vector<Real> baryCoef = getBaryCoef(d_map.getValue()[i].baryCoords);
        InDeriv inPos{0.,0.,0.};
        for (unsigned int j=0; j<element.size(); j++)
            inPos += in[element[j]] * baryCoef[j];

        Out::setCPos(out[i] , inPos);
    }
}

template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::resize( core::State<Out>* toModel )
{
    toModel->resize(d_map.getValue().size());
}

//-- test mapping partiel
template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperHexahedronSetTopology<In,Out,MappingDataType, Element>::applyOnePoint( const unsigned int& hexaPointId,typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    const helper::vector<Hexahedron>& cubes = this->m_fromTopology->getHexahedra();
    const Real fx = d_map.getValue()[hexaPointId].baryCoords[0];
    const Real fy = d_map.getValue()[hexaPointId].baryCoords[1];
    const Real fz = d_map.getValue()[hexaPointId].baryCoords[2];
    int index = d_map.getValue()[hexaPointId].in_index;
    const Hexahedron& cube = cubes[index];
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


/************************************* ApplyJ ***********************************/


template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::applyJ (const core::MechanicalParams * mparams, Data< typename Out::VecDeriv >& _out, const Data< typename In::VecDeriv >& in)
{
    SOFA_UNUSED(mparams);

#ifdef SOFA_DEV
    if ( sleeping.getValue()==false)
    {
#endif
        typename Out::VecDeriv* out = _out.beginEdit();
        if (m_mapper != NULL)
        {
            m_mapper->applyJ(*out, in.getValue());
        }
        _out.endEdit();
#ifdef SOFA_DEV
    }
#endif
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize( m_map1d.size() +m_map2d.size() +m_map3d.size() );

    const SeqLines& lines = this->m_fromTopology->getLines();
    const SeqTriangles& triangles = this->m_fromTopology->getTriangles();
    const SeqQuads& quads = this->m_fromTopology->getQuads();
    const SeqTetrahedra& tetrahedra = this->m_fromTopology->getTetrahedra();
    const SeqHexahedra& cubes = this->m_fromTopology->getHexahedra();

    const size_t sizeMap1d=m_map1d.size();
    const size_t sizeMap2d=m_map2d.size();
    const size_t sizeMap3d=m_map3d.size();

    const size_t idxStart1=sizeMap1d;
    const size_t idxStart2=sizeMap1d+sizeMap2d;
    const size_t idxStart3=sizeMap1d+sizeMap2d+sizeMap3d;

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( this->maskTo->isActivated() && !this->maskTo->getEntry(i) ) continue;

        // 1D elements
        if (i < idxStart1)
        {
            const Real fx = m_map1d[i].baryCoords[0];
            int index = m_map1d[i].in_index;
            {
                const Line& line = lines[index];
                Out::setDPos(out[i] , in[line[0]] * ( 1-fx )
                        + in[line[1]] * fx );
            }
        }
        // 2D elements
        else if (i < idxStart2)
        {
            const size_t i0 = idxStart1;
            const size_t c0 = triangles.size();

            const Real fx = m_map2d[i-i0].baryCoords[0];
            const Real fy = m_map2d[i-i0].baryCoords[1];
            size_t index = m_map2d[i-i0].in_index;

            if ( index<c0 )
            {
                const Triangle& triangle = triangles[index];
                Out::setDPos(out[i] , in[triangle[0]] * ( 1-fx-fy )
                        + in[triangle[1]] * fx
                        + in[triangle[2]] * fy );
            }
            else
            {
                const Quad& quad = quads[index-c0];
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
            const Real fx = m_map3d[i-i0].baryCoords[0];
            const Real fy = m_map3d[i-i0].baryCoords[1];
            const Real fz = m_map3d[i-i0].baryCoords[2];
            size_t index = m_map3d[i-i0].in_index;
            if ( index<c0 )
            {
                const Tetra& tetra = tetrahedra[index];
                Out::setDPos(out[i] , in[tetra[0]] * ( 1-fx-fy-fz )
                        + in[tetra[1]] * fx
                        + in[tetra[2]] * fy
                        + in[tetra[3]] * fz );
            }
            else
            {
                const Hexa& cube = cubes[index-c0];

                Out::setDPos(out[i] , in[cube[0]] * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) )
                        + in[cube[1]] * ( ( fx ) * ( 1-fy ) * ( 1-fz ) )
                        + in[cube[3]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                        + in[cube[2]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
                        + in[cube[4]] * ( ( 1-fx ) * ( 1-fy ) * ( fz ) )
                        + in[cube[5]] * ( ( fx ) * ( 1-fy ) * ( fz ) )
                        + in[cube[7]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                        + in[cube[6]] * ( ( fx ) * ( fy ) * ( fz ) ) );
            }
        }
    }

}

template <class In, class Out>
void BarycentricMapperRegularGridTopology<In,Out>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize( m_map.size() );

    for( size_t index=0 ; index<this->maskTo->size() ; ++index)
    {
        if( this->maskTo->isActivated() && !this->maskTo->getEntry(index) ) continue;

        const topology::RegularGridTopology::Hexa cube = this->m_fromTopology->getHexaCopy ( this->m_map[index].in_index );

        const Real fx = m_map[index].baryCoords[0];
        const Real fy = m_map[index].baryCoords[1];
        const Real fz = m_map[index].baryCoords[2];
        Out::setDPos(out[index] , in[cube[0]] * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) )
                + in[cube[1]] * ( ( fx ) * ( 1-fy ) * ( 1-fz ) )
                + in[cube[3]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                + in[cube[2]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
                + in[cube[4]] * ( ( 1-fx ) * ( 1-fy ) * ( fz ) )
                + in[cube[5]] * ( ( fx ) * ( 1-fy ) * ( fz ) )
                + in[cube[7]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                + in[cube[6]] * ( ( fx ) * ( fy ) * ( fz ) ) );
    }
}

template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize( m_map.size() );

    for( size_t index=0 ; index<this->maskTo->size() ; ++index)
    {
        if( this->maskTo->isActivated() && !this->maskTo->getEntry(index) ) continue;

        const topology::SparseGridTopology::Hexa cube = this->m_fromTopology->getHexahedron ( this->m_map[index].in_index );

        const Real fx = m_map[index].baryCoords[0];
        const Real fy = m_map[index].baryCoords[1];
        const Real fz = m_map[index].baryCoords[2];
        Out::setDPos(out[index] , in[cube[0]] * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) )
                + in[cube[1]] * ( ( fx ) * ( 1-fy ) * ( 1-fz ) )
                + in[cube[3]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
                + in[cube[2]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
                + in[cube[4]] * ( ( 1-fx ) * ( 1-fy ) * ( fz ) )
                + in[cube[5]] * ( ( fx ) * ( 1-fy ) * ( fz ) )
                + in[cube[7]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
                + in[cube[6]] * ( ( fx ) * ( fy ) * ( fz ) ) );
    }

}

template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize( d_map.getValue().size() );

    const helper::vector<Element>& elements = getElements();

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( this->maskTo->isActivated() && !this->maskTo->getEntry(i) ) continue;

        int index = d_map.getValue()[i].in_index;
        const Element& element = elements[index];

        helper::vector<Real> baryCoef = getBaryCoef(d_map.getValue()[i].baryCoords);
        InDeriv inPos{0.,0.,0.};
        for (unsigned int j=0; j<element.size(); j++)
            inPos += in[element[j]] * baryCoef[j];

        Out::setDPos(out[i] , inPos);
    }
}


/************************************* ApplyJT ***********************************/


template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::applyJT (const core::MechanicalParams * mparams, Data< typename In::VecDeriv >& out, const Data< typename Out::VecDeriv >& in)
{
    SOFA_UNUSED(mparams);

    if (
#ifdef SOFA_DEV
        sleeping.getValue()==false &&
#endif
        m_mapper != NULL)
    {
        m_mapper->applyJT(*out.beginEdit(), in.getValue());
        out.endEdit();
    }
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const SeqLines& lines = this->m_fromTopology->getLines();
    const SeqTriangles& triangles = this->m_fromTopology->getTriangles();
    const SeqQuads& quads = this->m_fromTopology->getQuads();
    const SeqTetrahedra& tetrahedra = this->m_fromTopology->getTetrahedra();
    const SeqHexahedra& cubes = this->m_fromTopology->getHexahedra();

    const size_t i1d = m_map1d.size();
    const size_t i2d = m_map2d.size();
    const size_t i3d = m_map3d.size();

    ForceMask& mask = *this->maskFrom;

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( !this->maskTo->getEntry(i) ) continue;

        // 1D elements
        if (i < i1d)
        {
            const typename Out::DPos v = Out::getDPos(in[i]);
            const OutReal fx = ( OutReal ) m_map1d[i].baryCoords[0];
            size_t index = m_map1d[i].in_index;
            {
                const Line& line = lines[index];
                out[line[0]] += v * ( 1-fx );
                out[line[1]] += v * fx;
                mask.insertEntry(line[0]);
                mask.insertEntry(line[1]);
            }
        }
        // 2D elements
        else if (i < i1d+i2d)
        {
            const size_t i0 = m_map1d.size();
            const size_t c0 = triangles.size();
            const typename Out::DPos v = Out::getDPos(in[i]);
            const OutReal fx = ( OutReal ) m_map2d[i-i0].baryCoords[0];
            const OutReal fy = ( OutReal ) m_map2d[i-i0].baryCoords[1];
            size_t index = m_map2d[i-i0].in_index;
            if ( index<c0 )
            {
                const Triangle& triangle = triangles[index];
                out[triangle[0]] += v * ( 1-fx-fy );
                out[triangle[1]] += v * fx;
                out[triangle[2]] += v * fy;
                mask.insertEntry(triangle[0]);
                mask.insertEntry(triangle[1]);
                mask.insertEntry(triangle[2]);
            }
            else
            {
                const Quad& quad = quads[index-c0];
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
            const size_t i0 = m_map1d.size() + m_map2d.size();
            const size_t c0 = tetrahedra.size();
            const typename Out::DPos v = Out::getDPos(in[i]);
            const OutReal fx = ( OutReal ) m_map3d[i-i0].baryCoords[0];
            const OutReal fy = ( OutReal ) m_map3d[i-i0].baryCoords[1];
            const OutReal fz = ( OutReal ) m_map3d[i-i0].baryCoords[2];
            size_t index = m_map3d[i-i0].in_index;
            if ( index<c0 )
            {
                const Tetra& tetra = tetrahedra[index];
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

                const Hexa& cube = cubes[index-c0];

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
        const topology::RegularGridTopology::Hexa cube = this->m_fromTopology->getHexaCopy ( this->m_map[index].in_index );

        const OutReal fx = ( OutReal ) m_map[index].baryCoords[0];
        const OutReal fy = ( OutReal ) m_map[index].baryCoords[1];
        const OutReal fz = ( OutReal ) m_map[index].baryCoords[2];
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

template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    ForceMask& mask = *this->maskFrom;

    for( size_t index=0 ; index<this->maskTo->size() ; ++index)
    {
        if( !this->maskTo->getEntry(index) ) continue;

        const typename Out::DPos v = Out::getDPos(in[index]);

        const topology::SparseGridTopology::Hexa cube = this->m_fromTopology->getHexahedron ( this->m_map[index].in_index );

        const OutReal fx = ( OutReal ) m_map[index].baryCoords[0];
        const OutReal fy = ( OutReal ) m_map[index].baryCoords[1];
        const OutReal fz = ( OutReal ) m_map[index].baryCoords[2];
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

template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const helper::vector<Element>& elements = getElements();

    ForceMask& mask = *this->maskFrom;
    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( !this->maskTo->getEntry(i) ) continue;

        int index = d_map.getValue()[i].in_index;
        const Element& element = elements[index];

        const typename Out::DPos inPos = Out::getDPos(in[i]);
        helper::vector<Real> baryCoef = getBaryCoef(d_map.getValue()[i].baryCoords);
        for (unsigned int j=0; j<element.size(); j++)
        {
            out[element[j]] += inPos * baryCoef[j];
            mask.insertEntry(element[j]);
        }
    }
}


/************************************* GetJ ***********************************/

template <class TIn, class TOut>
const sofa::defaulttype::BaseMatrix* BarycentricMapping<TIn, TOut>::getJ()
{
    if (
#ifdef SOFA_DEV
        sleeping.getValue()==false &&
#endif
        m_mapper!=NULL )
    {
        const size_t outStateSize = this->toModel->getSize();
        const size_t inStateSize = this->fromModel->getSize();
        const sofa::defaulttype::BaseMatrix* matJ = m_mapper->getJ((int)outStateSize, (int)inStateSize);

        return matJ;
    }
    else
        return NULL;
}

template <class In, class Out>
const sofa::defaulttype::BaseMatrix* BarycentricMapperMeshTopology<In,Out>::getJ(int outSize, int inSize)
{

    if (m_matrixJ && !m_updateJ && m_matrixJ->rowBSize() == (MatrixTypeIndex)outSize && m_matrixJ->colBSize() == (MatrixTypeIndex)inSize)
        return m_matrixJ;
    if (outSize > 0 && m_map1d.size()+m_map2d.size()+m_map3d.size() == 0)
        return NULL; // error: maps not yet created ?
    if (!m_matrixJ) m_matrixJ = new MatrixType;
    if (m_matrixJ->rowBSize() != (MatrixTypeIndex)outSize || m_matrixJ->colBSize() != (MatrixTypeIndex)inSize)
        m_matrixJ->resize(outSize*NOut, inSize*NIn);
    else
        m_matrixJ->clear();

    const SeqLines& lines = this->m_fromTopology->getLines();
    const SeqTriangles& triangles = this->m_fromTopology->getTriangles();
    const SeqQuads& quads = this->m_fromTopology->getQuads();
    const SeqTetrahedra& tetrahedra = this->m_fromTopology->getTetrahedra();
    const SeqHexahedra& cubes = this->m_fromTopology->getHexahedra();


    // 1D elements
    {
        for ( size_t i=0; i<m_map1d.size(); i++ )
        {
            const size_t out = i;
            const Real fx = ( Real ) m_map1d[i].baryCoords[0];
            size_t index = m_map1d[i].in_index;
            {
                const Line& line = lines[index];
                this->addMatrixContrib(m_matrixJ, out, line[0],  ( 1-fx ));
                this->addMatrixContrib(m_matrixJ, out, line[1],  fx);
            }
        }
    }
    // 2D elements
    {
        const size_t i0 = m_map1d.size();
        const size_t c0 = triangles.size();
        for ( size_t i=0; i<m_map2d.size(); i++ )
        {
            const size_t out = i+i0;
            const Real fx = ( Real ) m_map2d[i].baryCoords[0];
            const Real fy = ( Real ) m_map2d[i].baryCoords[1];
            size_t index = m_map2d[i].in_index;
            if ( index<c0 )
            {
                const Triangle& triangle = triangles[index];
                this->addMatrixContrib(m_matrixJ, out, triangle[0],  ( 1-fx-fy ));
                this->addMatrixContrib(m_matrixJ, out, triangle[1],  fx);
                this->addMatrixContrib(m_matrixJ, out, triangle[2],  fy);
            }
            else
            {
                const Quad& quad = quads[index-c0];
                this->addMatrixContrib(m_matrixJ, out, quad[0],  ( ( 1-fx ) * ( 1-fy ) ));
                this->addMatrixContrib(m_matrixJ, out, quad[1],  ( ( fx ) * ( 1-fy ) ));
                this->addMatrixContrib(m_matrixJ, out, quad[3],  ( ( 1-fx ) * ( fy ) ));
                this->addMatrixContrib(m_matrixJ, out, quad[2],  ( ( fx ) * ( fy ) ));
            }
        }
    }
    // 3D elements
    {
        const size_t i0 = m_map1d.size() + m_map2d.size();
        const size_t c0 = tetrahedra.size();
        for ( size_t i=0; i<m_map3d.size(); i++ )
        {
            const size_t out = i+i0;
            const Real fx = ( Real ) m_map3d[i].baryCoords[0];
            const Real fy = ( Real ) m_map3d[i].baryCoords[1];
            const Real fz = ( Real ) m_map3d[i].baryCoords[2];
            size_t index = m_map3d[i].in_index;
            if ( index<c0 )
            {
                const Tetra& tetra = tetrahedra[index];
                this->addMatrixContrib(m_matrixJ, out, tetra[0],  ( 1-fx-fy-fz ));
                this->addMatrixContrib(m_matrixJ, out, tetra[1],  fx);
                this->addMatrixContrib(m_matrixJ, out, tetra[2],  fy);
                this->addMatrixContrib(m_matrixJ, out, tetra[3],  fz);
            }
            else
            {
                const Hexa& cube = cubes[index-c0];

                this->addMatrixContrib(m_matrixJ, out, cube[0],  ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) ));
                this->addMatrixContrib(m_matrixJ, out, cube[1],  ( ( fx ) * ( 1-fy ) * ( 1-fz ) ));

                this->addMatrixContrib(m_matrixJ, out, cube[3],  ( ( 1-fx ) * ( fy ) * ( 1-fz ) ));
                this->addMatrixContrib(m_matrixJ, out, cube[2],  ( ( fx ) * ( fy ) * ( 1-fz ) ));

                this->addMatrixContrib(m_matrixJ, out, cube[4],  ( ( 1-fx ) * ( 1-fy ) * ( fz ) ));
                this->addMatrixContrib(m_matrixJ, out, cube[5],  ( ( fx ) * ( 1-fy ) * ( fz ) ));

                this->addMatrixContrib(m_matrixJ, out, cube[7],  ( ( 1-fx ) * ( fy ) * ( fz ) ));
                this->addMatrixContrib(m_matrixJ, out, cube[6],  ( ( fx ) * ( fy ) * ( fz ) ));
            }
        }
    }
    m_matrixJ->compress();
    m_updateJ = false;
    return m_matrixJ;
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

    for ( size_t i=0; i<m_map.size(); i++ )
    {
        const int out = i;

        const topology::RegularGridTopology::Hexa cube = this->m_fromTopology->getHexaCopy ( this->m_map[i].in_index );

        const Real fx = ( Real ) m_map[i].baryCoords[0];
        const Real fy = ( Real ) m_map[i].baryCoords[1];
        const Real fz = ( Real ) m_map[i].baryCoords[2];
        this->addMatrixContrib(matrixJ, out, cube[0], ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[1], ( ( fx ) * ( 1-fy ) * ( 1-fz ) ));

        this->addMatrixContrib(matrixJ, out, cube[3], ( ( 1-fx ) * ( fy ) * ( 1-fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[2], ( ( fx ) * ( fy ) * ( 1-fz ) ));

        this->addMatrixContrib(matrixJ, out, cube[4], ( ( 1-fx ) * ( 1-fy ) * ( fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[5], ( ( fx ) * ( 1-fy ) * ( fz ) ));

        this->addMatrixContrib(matrixJ, out, cube[7], ( ( 1-fx ) * ( fy ) * ( fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[6], ( ( fx ) * ( fy ) * ( fz ) ));
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

    for ( size_t i=0; i<m_map.size(); i++ )
    {
        const int out = i;

        const topology::SparseGridTopology::Hexa cube = this->m_fromTopology->getHexahedron ( this->m_map[i].in_index );

        const Real fx = ( Real ) m_map[i].baryCoords[0];
        const Real fy = ( Real ) m_map[i].baryCoords[1];
        const Real fz = ( Real ) m_map[i].baryCoords[2];
        this->addMatrixContrib(matrixJ, out, cube[0], ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[1], ( ( fx ) * ( 1-fy ) * ( 1-fz ) ));

        this->addMatrixContrib(matrixJ, out, cube[3], ( ( 1-fx ) * ( fy ) * ( 1-fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[2], ( ( fx ) * ( fy ) * ( 1-fz ) ));

        this->addMatrixContrib(matrixJ, out, cube[4], ( ( 1-fx ) * ( 1-fy ) * ( fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[5], ( ( fx ) * ( 1-fy ) * ( fz ) ));

        this->addMatrixContrib(matrixJ, out, cube[7], ( ( 1-fx ) * ( fy ) * ( fz ) ));
        this->addMatrixContrib(matrixJ, out, cube[6], ( ( fx ) * ( fy ) * ( fz ) ));
    }
    matrixJ->compress();
    updateJ = false;
    return matrixJ;
}

template <class In, class Out, class MappingDataType, class Element>
const sofa::defaulttype::BaseMatrix* BarycentricMapperTopologyContainer<In,Out,MappingDataType, Element>::getJ(int outSize, int inSize)
{
    if (m_matrixJ && !m_updateJ)
        return m_matrixJ;

    if (!m_matrixJ) m_matrixJ = new MatrixType;
    if (m_matrixJ->rowBSize() != (MatrixTypeIndex)outSize || m_matrixJ->colBSize() != (MatrixTypeIndex)inSize)
        m_matrixJ->resize(outSize*NOut, inSize*NIn);
    else
        m_matrixJ->clear();

    return m_matrixJ;

    const helper::vector<Element>& elements = getElements();

    for( size_t outId=0 ; outId<this->maskTo->size() ; ++outId)
    {
        if( !this->maskTo->getEntry(outId) ) continue;

        const Element& element = elements[d_map.getValue()[outId].in_index];

        helper::vector<Real> baryCoef = getBaryCoef(d_map.getValue()[outId].baryCoords);
        for (unsigned int j=0; j<element.size(); j++)
            this->addMatrixContrib(m_matrixJ, outId, element[j], baryCoef[j]);
    }

    m_matrixJ->compress();
    m_updateJ = false;
    return m_matrixJ;
}


/************************************* Draw ***********************************/


template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if ( !vparams->displayFlags().getShowMappings() ) return;

    // Draw model (out) points
    const OutVecCoord& out = this->toModel->read(core::ConstVecCoordId::position())->getValue();
    std::vector< Vector3 > points;
    for ( unsigned int i=0; i<out.size(); i++ )
    {
        points.push_back ( OutDataTypes::getCPos(out[i]) );
    }
    vparams->drawTool()->drawPoints ( points, 7, sofa::defaulttype::Vec<4,float> ( 1,1,0,1 ) );

    // Draw mapping line between models
    const InVecCoord& in = this->fromModel->read(core::ConstVecCoordId::position())->getValue();
    if ( m_mapper!=NULL )
        m_mapper->draw(vparams,out,in);

}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::draw  (const core::visual::VisualParams* vparams,
                                                   const typename Out::VecCoord& out,
                                                   const typename In::VecCoord& in )
{
    const SeqLines& lines = this->m_fromTopology->getLines();
    const SeqTriangles& triangles = this->m_fromTopology->getTriangles();
    const SeqQuads& quads = this->m_fromTopology->getQuads();
    const SeqTetrahedra& tetrahedra = this->m_fromTopology->getTetrahedra();
    const SeqHexahedra& cubes = this->m_fromTopology->getHexahedra();

    std::vector< Vector3 > points;
    // 1D elements
    {
        const int i0 = 0;
        for ( unsigned int i=0; i<m_map1d.size(); i++ )
        {
            const Real fx = m_map1d[i].baryCoords[0];
            int index = m_map1d[i].in_index;
            {
                const Line& line = lines[index];
                Real f[2];
                f[0] = ( 1-fx );
                f[1] = fx;
                for ( int j=0; j<2; j++ )
                {
                    if ( f[j]<=-0.0001 || f[j]>=0.0001 )
                    {
                        points.push_back ( Out::getCPos(out[i+i0]) );
                        points.push_back ( in[line[j]] );
                    }
                }
            }
        }
    }
    // 2D elements
    {
        const int i0 = m_map1d.size();
        const int c0 = triangles.size();
        for ( unsigned int i=0; i<m_map2d.size(); i++ )
        {
            const Real fx = m_map2d[i].baryCoords[0];
            const Real fy = m_map2d[i].baryCoords[1];
            int index = m_map2d[i].in_index;
            if ( index<c0 )
            {
                const Triangle& triangle = triangles[index];
                Real f[3];
                f[0] = ( 1-fx-fy );
                f[1] = fx;
                f[2] = fy;
                for ( int j=0; j<3; j++ )
                {
                    if ( f[j]<=-0.0001 || f[j]>=0.0001 )
                    {
                        points.push_back ( Out::getCPos(out[i+i0]) );
                        points.push_back ( in[triangle[j]] );
                    }
                }
            }
            else
            {
                const Quad& quad = quads[index-c0];
                Real f[4];
                f[0] = ( ( 1-fx ) * ( 1-fy ) );
                f[1] = ( ( fx ) * ( 1-fy ) );
                f[3] = ( ( 1-fx ) * ( fy ) );
                f[2] = ( ( fx ) * ( fy ) );
                for ( int j=0; j<4; j++ )
                {
                    if ( f[j]<=-0.0001 || f[j]>=0.0001 )
                    {
                        points.push_back ( Out::getCPos(out[i+i0]) );
                        points.push_back ( in[quad[j]] );
                    }
                }
            }
        }
    }
    // 3D elements
    {
        const int i0 = m_map1d.size() +m_map2d.size();
        const int c0 = tetrahedra.size();
        for ( unsigned int i=0; i<m_map3d.size(); i++ )
        {
            const Real fx = m_map3d[i].baryCoords[0];
            const Real fy = m_map3d[i].baryCoords[1];
            const Real fz = m_map3d[i].baryCoords[2];
            int index = m_map3d[i].in_index;
            if ( index<c0 )
            {
                const Tetra& tetra = tetrahedra[index];
                Real f[4];
                f[0] = ( 1-fx-fy-fz );
                f[1] = fx;
                f[2] = fy;
                f[3] = fz;
                for ( int j=0; j<4; j++ )
                {
                    if ( f[j]<=-0.0001 || f[j]>=0.0001 )
                    {
                        points.push_back ( Out::getCPos(out[i+i0]) );
                        points.push_back ( in[tetra[j]] );
                    }
                }
            }
            else
            {
                const Hexa& cube = cubes[index-c0];

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
void BarycentricMapperRegularGridTopology<In,Out>::draw  (const core::visual::VisualParams* vparams,
                                                          const typename Out::VecCoord& out,
                                                          const typename In::VecCoord& in )
{
    std::vector< Vector3 > points;

    for ( unsigned int i=0; i<m_map.size(); i++ )
    {

        const topology::RegularGridTopology::Hexa cube = this->m_fromTopology->getHexaCopy ( this->m_map[i].in_index );

        const Real fx = m_map[i].baryCoords[0];
        const Real fy = m_map[i].baryCoords[1];
        const Real fz = m_map[i].baryCoords[2];
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
                points.push_back ( Out::getCPos(out[i]) );
                points.push_back ( in[cube[j]] );
            }
        }
    }
    vparams->drawTool()->drawLines ( points, 1, sofa::defaulttype::Vec<4,float> ( 0,0,1,1 ) );

}

template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::draw  (const core::visual::VisualParams* vparams,
                                                         const typename Out::VecCoord& out,
                                                         const typename In::VecCoord& in )
{
    std::vector< Vector3 > points;
    for ( unsigned int i=0; i<m_map.size(); i++ )
    {

        const topology::SparseGridTopology::Hexa cube = this->m_fromTopology->getHexahedron ( this->m_map[i].in_index );

        const Real fx = m_map[i].baryCoords[0];
        const Real fy = m_map[i].baryCoords[1];
        const Real fz = m_map[i].baryCoords[2];
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
                points.push_back ( Out::getCPos(out[i]) );
                points.push_back ( in[cube[j]] );
            }
        }
    }
    vparams->drawTool()->drawLines ( points, 1, sofa::defaulttype::Vec<4,float> ( 0,0,1,1 ) );
}


template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::draw  (const core::visual::VisualParams* vparams,
                                                                                const typename Out::VecCoord& out,
                                                                                const typename In::VecCoord& in )
{
    // Draw line between mapped node (out) and nodes of nearest element (in)
    const helper::vector<Element>& elements = getElements();

    std::vector< Vector3 > points;
    {
        for ( unsigned int i=0; i<d_map.getValue().size(); i++ )
        {
            int index = d_map.getValue()[i].in_index;
            const Element& element = elements[index];
            helper::vector<Real> baryCoef = getBaryCoef(d_map.getValue()[i].baryCoords);
            for ( unsigned int j=0; j<element.size(); j++ )
            {
                if ( baryCoef[j]<=-0.0001 || baryCoef[j]>=0.0001 )
                {
                    points.push_back ( Out::getCPos(out[i]) );
                    points.push_back ( in[element[j]] );
                }
            }
        }
    }
    vparams->drawTool()->drawLines ( points, 1, sofa::defaulttype::Vec<4,float> ( 0,1,0,1 ) );
}


/************************************* PropagateConstraint ***********************************/


template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::applyJT(const core::ConstraintParams * cparams, Data< typename In::MatrixDeriv >& out, const Data< typename Out::MatrixDeriv >& in)
{
    SOFA_UNUSED(cparams);

    if (
#ifdef SOFA_DEV
        sleeping.getValue()==false &&
#endif
        m_mapper!=NULL )
    {
        m_mapper->applyJT(*out.beginEdit(), in.getValue());
        out.endEdit();
    }
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
{
    const SeqLines& lines = this->m_fromTopology->getLines();
    const SeqTriangles& triangles = this->m_fromTopology->getTriangles();
    const SeqQuads& quads = this->m_fromTopology->getQuads();
    const SeqTetrahedra& tetrahedra = this->m_fromTopology->getTetrahedra();
    const SeqHexahedra& hexas = this->m_fromTopology->getHexahedra();

    const size_t iTri = triangles.size();
    const size_t iTetra= tetrahedra.size();

    const size_t i1d = m_map1d.size();
    const size_t i2d = m_map2d.size();
    const size_t i3d = m_map3d.size();

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
                    const OutReal fx = ( OutReal ) m_map1d[indexIn].baryCoords[0];
                    size_t index = m_map1d[indexIn].in_index;
                    {
                        const Line& line = lines[index];
                        o.addCol( line[0], data * ( 1-fx ) );
                        o.addCol( line[1], data * fx );
                    }
                }
                // 2D elements : triangle or quad
                else if ( indexIn < i2d )
                {
                    const OutReal fx = ( OutReal ) m_map2d[indexIn].baryCoords[0];
                    const OutReal fy = ( OutReal ) m_map2d[indexIn].baryCoords[1];
                    size_t index = m_map2d[indexIn].in_index;
                    if ( index < iTri ) // triangle
                    {
                        const Triangle& triangle = triangles[index];
                        o.addCol( triangle[0], data * ( 1-fx-fy ) );
                        o.addCol( triangle[1], data * fx );
                        o.addCol( triangle[2], data * fy );
                    }
                    else // quad
                    {
                        const Quad& quad = quads[index - iTri];
                        o.addCol( quad[0], data * ( ( 1-fx ) * ( 1-fy ) ) );
                        o.addCol( quad[1], data * ( ( fx ) * ( 1-fy ) ) );
                        o.addCol( quad[3], data * ( ( 1-fx ) * ( fy ) ) );
                        o.addCol( quad[2], data * ( ( fx ) * ( fy ) ) );
                    }
                }
                // 3D elements : tetra or hexa
                else if ( indexIn < i3d )
                {
                    const OutReal fx = ( OutReal ) m_map3d[indexIn].baryCoords[0];
                    const OutReal fy = ( OutReal ) m_map3d[indexIn].baryCoords[1];
                    const OutReal fz = ( OutReal ) m_map3d[indexIn].baryCoords[2];
                    size_t index = m_map3d[indexIn].in_index;
                    if ( index < iTetra ) // tetra
                    {
                        const Tetra& tetra = tetrahedra[index];
                        o.addCol ( tetra[0], data * ( 1-fx-fy-fz ) );
                        o.addCol ( tetra[1], data * fx );
                        o.addCol ( tetra[2], data * fy );
                        o.addCol ( tetra[3], data * fz );
                    }
                    else // hexa
                    {
                        const Hexa& hexa = hexas[index-iTetra];

                        o.addCol ( hexa[0],data * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) ) ) ;
                        o.addCol ( hexa[1],data * ( ( fx ) * ( 1-fy ) * ( 1-fz ) ) ) ;

                        o.addCol ( hexa[3],data * ( ( 1-fx ) * ( fy ) * ( 1-fz ) ) ) ;
                        o.addCol ( hexa[2],data * ( ( fx ) * ( fy ) * ( 1-fz ) ) ) ;

                        o.addCol ( hexa[4],data * ( ( 1-fx ) * ( 1-fy ) * ( fz ) ) ) ;
                        o.addCol ( hexa[5],data * ( ( fx ) * ( 1-fy ) * ( fz ) ) ) ;

                        o.addCol ( hexa[7],data * ( ( 1-fx ) * ( fy ) * ( fz ) ) ) ;
                        o.addCol ( hexa[6],data * ( ( fx ) * ( fy ) * ( fz ) ) ) ;
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


                const topology::RegularGridTopology::Hexa cube = this->m_fromTopology->getHexaCopy ( this->m_map[indexIn].in_index );

                const OutReal fx = (OutReal) m_map[indexIn].baryCoords[0];
                const OutReal fy = (OutReal) m_map[indexIn].baryCoords[1];
                const OutReal fz = (OutReal) m_map[indexIn].baryCoords[2];
                const OutReal oneMinusFx = 1-fx;
                const OutReal oneMinusFy = 1-fy;
                const OutReal oneMinusFz = 1-fz;

                o.addCol(cube[0], data * ((oneMinusFx) * (oneMinusFy) * (oneMinusFz)));
                o.addCol(cube[1], data * ((fx) * (oneMinusFy) * (oneMinusFz)));

                o.addCol(cube[3], data * ((oneMinusFx) * (fy) * (oneMinusFz)));
                o.addCol(cube[2], data * ((fx) * (fy) * (oneMinusFz)));

                o.addCol(cube[4], data * ((oneMinusFx) * (oneMinusFy) * (fz)));
                o.addCol(cube[5], data * ((fx) * (oneMinusFy) * (fz)));

                o.addCol(cube[7], data * ((oneMinusFx) * (fy) * (fz)));
                o.addCol(cube[6], data * ((fx) * (fy) * (fz)));
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


                const topology::SparseGridTopology::Hexa cube = this->m_fromTopology->getHexahedron ( this->m_map[indexIn].in_index );

                const OutReal fx = ( OutReal ) m_map[indexIn].baryCoords[0];
                const OutReal fy = ( OutReal ) m_map[indexIn].baryCoords[1];
                const OutReal fz = ( OutReal ) m_map[indexIn].baryCoords[2];
                const OutReal oneMinusFx = 1-fx;
                const OutReal oneMinusFy = 1-fy;
                const OutReal oneMinusFz = 1-fz;

                OutReal f = ( oneMinusFx * oneMinusFy * oneMinusFz );
                o.addCol ( cube[0],  ( data * f ) );

                f = ( ( fx ) * oneMinusFy * oneMinusFz );
                o.addCol ( cube[1],  ( data * f ) );


                f = ( oneMinusFx * ( fy ) * oneMinusFz );
                o.addCol ( cube[3],  ( data * f ) );

                f = ( ( fx ) * ( fy ) * oneMinusFz );
                o.addCol ( cube[2],  ( data * f ) );


                f = ( oneMinusFx * oneMinusFy * ( fz ) );
                o.addCol ( cube[4],  ( data * f ) );

                f = ( ( fx ) * oneMinusFy * ( fz ) );
                o.addCol ( cube[5],  ( data * f ) );


                f = ( oneMinusFx * ( fy ) * ( fz ) );
                o.addCol ( cube[7],  ( data * f ) );

                f = ( ( fx ) * ( fy ) * ( fz ) );
                o.addCol ( cube[6],  ( data * f ) );
            }
        }
    }
}

template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
{
    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();
    const helper::vector< Element >& elements = getElements();

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

                const Element& element = elements[d_map.getValue()[indexIn].in_index];

                helper::vector<Real> baryCoef = getBaryCoef(d_map.getValue()[indexIn].baryCoords);
                for (unsigned int j=0; j<element.size(); j++)
                    o.addCol(element[j], data*baryCoef[j]);
            }
        }
    }
}


/************************************* Topological Changes ***********************************/


template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperHexahedronSetTopology<In,Out,MappingDataType,Element>::handleTopologyChange(core::topology::Topology* t)
{
    using sofa::core::behavior::MechanicalState;

    if (t != this->m_fromTopology) return;

    if ( this->m_fromTopology->beginChange() == this->m_fromTopology->endChange() )
        return;

    std::list<const core::topology::TopologyChange *>::const_iterator itBegin = this->m_fromTopology->beginChange();
    std::list<const core::topology::TopologyChange *>::const_iterator itEnd = this->m_fromTopology->endChange();

    for ( std::list<const core::topology::TopologyChange *>::const_iterator changeIt = itBegin;
            changeIt != itEnd; ++changeIt )
    {
        const core::topology::TopologyChangeType changeType = ( *changeIt )->getChangeType();
        switch ( changeType )
        {
            //TODO: implementation of BarycentricMapperHexahedronSetTopology<In,Out>::handleTopologyChange()
        case core::topology::ENDING_EVENT:       ///< To notify the end for the current sequence of topological change events
        {
            if(!m_invalidIndex.empty())
            {
                helper::vector<MappingData>& mapData = *(d_map.beginEdit());

                for ( std::set<int>::const_iterator iter = m_invalidIndex.begin();
                        iter != m_invalidIndex.end(); ++iter )
                {
                    const int j = *iter;
                    if ( mapData[j].in_index == -1 ) // compute new mapping
                    {
                        Vector3 coefs;
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
                                index = m_fromGeomAlgo->findNearestElementInRestPos ( Out::getCPos(xto0[j]), coefs, distance );
                                coefs = m_fromGeomAlgo->computeHexahedronRestBarycentricCoeficients(index, pos);
                            }
                        }
                        else
                        {
                            index = m_fromGeomAlgo->findNearestElementInRestPos ( pos, coefs, distance );
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

                d_map.endEdit();
                m_invalidIndex.clear();
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
            const unsigned int nbHexahedra = this->m_fromTopology->getNbHexahedra();

            const helper::vector<unsigned int> &hexahedra =
                    ( static_cast< const core::topology::HexahedraRemoved *> ( *changeIt ) )->getArray();
            //        helper::vector<unsigned int> hexahedra(tab);

            for ( unsigned int i=0; i<hexahedra.size(); ++i )
            {
                // remove all references to the removed cubes from the mapping data
                unsigned int cubeId = hexahedra[i];
                for ( unsigned int j=0; j<d_map.getValue().size(); ++j )
                {
                    if ( d_map.getValue()[j].in_index == ( int ) cubeId ) // invalidate mapping
                    {
                        Vector3 coefs;
                        coefs[0] = d_map.getValue()[j].baryCoords[0];
                        coefs[1] = d_map.getValue()[j].baryCoords[1];
                        coefs[2] = d_map.getValue()[j].baryCoords[2];

                        typename In::Coord restPos = m_fromGeomAlgo->getRestPointPositionInHexahedron ( cubeId, coefs );

                        helper::vector<MappingData>& vectorData = *(d_map.beginEdit());
                        vectorData[j].in_index = -1;
                        vectorData[j].baryCoords[0] = restPos[0];
                        vectorData[j].baryCoords[1] = restPos[1];
                        vectorData[j].baryCoords[2] = restPos[2];
                        d_map.endEdit();

                        m_invalidIndex.insert(j);
                    }
                }
            }

            // renumber
            unsigned int lastCubeId = nbHexahedra-1;
            for ( unsigned int i=0; i<hexahedra.size(); ++i, --lastCubeId )
            {
                unsigned int cubeId = hexahedra[i];
                for ( unsigned int j=0; j<d_map.getValue().size(); ++j )
                {
                    if ( d_map.getValue()[j].in_index == ( int ) lastCubeId )
                    {
                        helper::vector<MappingData>& vectorData = *(d_map.beginEdit());
                        vectorData[j].in_index = cubeId;
                        d_map.endEdit();
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

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::handleTopologyChange ( core::topology::Topology* t )
{
    SOFA_UNUSED(t);
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
    if( m_mapper )
        m_mapper->updateForceMask();
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
