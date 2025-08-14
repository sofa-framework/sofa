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
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperMeshTopology.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/State.h>

namespace sofa::component::mapping::linear
{

using sofa::type::Vec3;
using sofa::core::visual::VisualParams;
using sofa::type::Vec;
using sofa::type::Vec3;
using sofa::type::Matrix3;
typedef typename sofa::core::topology::BaseMeshTopology::Edge Edge;
typedef typename sofa::core::topology::BaseMeshTopology::Triangle Triangle;
typedef typename sofa::core::topology::BaseMeshTopology::Quad Quad;
typedef typename sofa::core::topology::BaseMeshTopology::Tetrahedron Tetrahedron;
typedef typename sofa::core::topology::BaseMeshTopology::Tetra Tetra;
typedef typename sofa::core::topology::BaseMeshTopology::Hexahedron Hexahedron;
typedef typename sofa::core::topology::BaseMeshTopology::Hexa Hexa;
typedef typename sofa::core::topology::BaseMeshTopology::SeqLines SeqLines;
typedef typename sofa::core::topology::BaseMeshTopology::SeqEdges SeqEdges;
typedef typename sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
typedef typename sofa::core::topology::BaseMeshTopology::SeqQuads SeqQuads;
typedef typename sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
typedef typename sofa::core::topology::BaseMeshTopology::SeqHexahedra SeqHexahedra;

template <class In, class Out>
BarycentricMapperMeshTopology<In,Out>::BarycentricMapperMeshTopology(core::topology::BaseMeshTopology* fromTopology,
    core::topology::BaseMeshTopology* toTopology)
    : TopologyBarycentricMapper<In,Out>(fromTopology, toTopology),
      m_matrixJ(nullptr), m_updateJ(true)
{
}

template <class In, class Out>
BarycentricMapperMeshTopology<In,Out>::~BarycentricMapperMeshTopology()
{
    if (m_matrixJ)
        delete m_matrixJ;
}

template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::init ( const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    m_updateJ = true;

    const SeqTetrahedra& tetras = this->m_fromTopology->getTetrahedra();
    const SeqHexahedra& hexas = this->m_fromTopology->getHexahedra();

    const SeqTriangles& triangles = this->m_fromTopology->getTriangles();
    const SeqQuads& quads = this->m_fromTopology->getQuads();
    type::vector<Matrix3> bases;
    type::vector<Vec3> centers;
    if ( tetras.empty() && hexas.empty() )
    {
        if ( triangles.empty() && quads.empty() )
        {
            const SeqEdges& edges = this->m_fromTopology->getEdges();
            if ( edges.empty() ) return;

            clearMap1dAndReserve ( out.size() );

            type::vector< SReal >   lengthEdges;
            type::vector< Vec3 > unitaryVectors;

            Index e;
            for ( e=0; e<edges.size(); e++ )
            {
                lengthEdges.push_back ( ( in[edges[e][1]]-in[edges[e][0]] ).norm() );

                Vec3 V12 = ( in[edges[e][1]]-in[edges[e][0]] ); V12.normalize();
                unitaryVectors.push_back ( V12 );
            }

            for ( std::size_t i=0; i<out.size(); i++ )
            {
                SReal coef=0;
                for ( e=0; e<edges.size(); e++ )
                {
                    SReal lengthEdge = lengthEdges[e];
                    Vec3 V12 =unitaryVectors[e];

                    coef = ( V12 ) * Vec3 ( Out::getCPos(out[i])-in[edges[e][0]] ) /lengthEdge;
                    if ( coef >= 0 && coef <= 1 ) {addPointInLine ( e,&coef );  break; }
                }
                //If no good coefficient has been found, we add to the last element
                if ( e == edges.size() ) addPointInLine ( Index(edges.size()-1),&coef );
            }
        }
        else
        {
            clearMap2dAndReserve ( (out.size()) );
            Size nbTriangles = Size(triangles.size());
            bases.resize ( triangles.size() +quads.size() );
            centers.resize ( triangles.size() +quads.size() );
            for ( std::size_t t = 0; t < triangles.size(); t++ )
            {
                Mat3x3 m,mt;
                m[0] = in[triangles[t][1]]-in[triangles[t][0]];
                m[1] = in[triangles[t][2]]-in[triangles[t][0]];
                m[2] = cross ( m[0],m[1] );
                mt.transpose ( m );
                const bool canInvert = bases[t].invert ( mt );
                assert(canInvert);
                SOFA_UNUSED(canInvert);
                centers[t] = ( in[triangles[t][0]]+in[triangles[t][1]]+in[triangles[t][2]] ) /3;
            }
            for ( std::size_t q = 0; q < quads.size(); q++ )
            {
                Mat3x3 m,mt;
                m[0] = in[quads[q][1]]-in[quads[q][0]];
                m[1] = in[quads[q][3]]-in[quads[q][0]];
                m[2] = cross ( m[0],m[1] );
                mt.transpose ( m );
                const bool canInvert = bases[nbTriangles+q].invert ( mt );
                assert(canInvert);
                SOFA_UNUSED(canInvert);
                centers[nbTriangles+q] = ( in[quads[q][0]]+in[quads[q][1]]+in[quads[q][2]]+in[quads[q][3]] ) *0.25;
            }
            for ( std::size_t i=0; i<out.size(); i++ )
            {
                const auto outPos = Out::getCPos(out[i]);
                sofa::type::Vec3 coefs;
                Index index = sofa::InvalidID;
                SReal distance = 1e10;
                for ( Index t = 0; t < triangles.size(); t++ )
                {
                    const auto v = bases[t] * ( outPos - in[triangles[t][0]] );
                    SReal d = std::max ( std::max (SReal(-v[0]), SReal(-v[1]) ),std::max ( SReal( ( v[2]<0?-v[2]:v[2] )-0.01), SReal(v[0]+v[1]-1 )));
                    if ( d>0 ) d = ( outPos-centers[t] ).norm2();
                    if ( d<distance ) { coefs = v; distance = d; index = (t); }
                }
                for ( Index q = 0; q < quads.size(); q++ )
                {
                    const auto v = bases[nbTriangles+q] * ( outPos - in[quads[q][0]] );
                    SReal d = std::max ( std::max (SReal(-v[0]), SReal(-v[1])),std::max ( std::max (SReal(v[1]-1), SReal(v[0]-1)),std::max (SReal(v[2]-0.01), SReal(-v[2]-0.01) ) ) );
                    if ( d>0 ) d = ( outPos-centers[nbTriangles+q] ).norm2();
                    if ( d<distance ) { coefs = v; distance = d; index = nbTriangles+q; }
                }
                if ( index < (nbTriangles) )
                    addPointInTriangle ( index, coefs.ptr() );
                else
                    addPointInQuad ( index-nbTriangles, coefs.ptr() );
            }
        }
    }
    else
    {
        clearMap3dAndReserve ( out.size() );
        Size nbTetras = Size(tetras.size());
        bases.resize ( tetras.size() + hexas.size() );
        centers.resize ( tetras.size() + hexas.size() );
        for ( std::size_t t = 0; t < tetras.size(); t++ )
        {
            Mat3x3 m,mt;
            m[0] = in[tetras[t][1]]-in[tetras[t][0]];
            m[1] = in[tetras[t][2]]-in[tetras[t][0]];
            m[2] = in[tetras[t][3]]-in[tetras[t][0]];
            mt.transpose ( m );
            const bool canInvert = bases[t].invert ( mt );
            assert(canInvert);
            SOFA_UNUSED(canInvert);
            centers[t] = ( in[tetras[t][0]]+in[tetras[t][1]]+in[tetras[t][2]]+in[tetras[t][3]] ) *0.25;
        }
        for ( std::size_t h = 0; h < hexas.size(); h++ )
        {
            Mat3x3 m,mt;
            m[0] = in[hexas[h][1]]-in[hexas[h][0]];
            m[1] = in[hexas[h][3]]-in[hexas[h][0]];
            m[2] = in[hexas[h][4]]-in[hexas[h][0]];
            mt.transpose ( m );
            const bool canInvert = bases[nbTetras+h].invert ( mt );
            assert(canInvert);
            SOFA_UNUSED(canInvert);
            centers[nbTetras+h] = ( in[hexas[h][0]]+in[hexas[h][1]]+in[hexas[h][2]]+in[hexas[h][3]]+in[hexas[h][4]]+in[hexas[h][5]]+in[hexas[h][6]]+in[hexas[h][7]] ) *0.125;
        }
        for ( std::size_t i=0; i<out.size(); i++ )
        {
            auto pos = Out::getCPos(out[i]);
            sofa::type::Vec3 coefs;
            Index index = sofa::InvalidID;
            double distance = 1e10;
            for (Index t = 0; t < tetras.size(); t++ )
            {
                const auto v = bases[t] * ( pos - in[tetras[t][0]] );
                SReal d = std::max ( std::max ( SReal(-v[0]), SReal(-v[1]) ),std::max (SReal(-v[2]), SReal(v[0]+v[1]+v[2]-1) ) );
                if ( d>0 ) d = ( pos-centers[t] ).norm2();
                if ( d<distance ) { coefs = v; distance = d; index = (t); }
            }
            for (Index h = 0; h < hexas.size(); h++ )
            {
                const auto v = bases[nbTetras+h] * ( pos - in[hexas[h][0]] );
                SReal d = std::max ( std::max (SReal(-v[0]), SReal(-v[1]) ),std::max ( std::max (SReal(-v[2]), SReal(v[0]-1) ),std::max (SReal(v[1]-1), SReal(v[2]-1) ) ) );
                if ( d>0 ) d = ( pos-centers[nbTetras+h] ).norm2();
                if ( d<distance ) { coefs = v; distance = d; index = (nbTetras+h); }
            }
            if ( index < (nbTetras) )
                addPointInTetra ( index, coefs.ptr() );
            else
                addPointInCube ( index-nbTetras, coefs.ptr() );
        }
    }
}


template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::clearMap1dAndReserve ( std::size_t size )
{
    m_updateJ = true;
    m_map1d.clear();
    if ( size>0 ) m_map1d.reserve ( size );
}


template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::clearMap2dAndReserve ( std::size_t size )
{
    m_updateJ = true;
    m_map2d.clear();
    if ( size>0 ) m_map2d.reserve ( size );
}


template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::clearMap3dAndReserve ( std::size_t size )
{
    m_updateJ = true;
    m_map3d.clear();
    if ( size>0 ) m_map3d.reserve ( size );
}


template <class In, class Out>
void BarycentricMapperMeshTopology<In,Out>::clear ( std::size_t size )
{
    m_updateJ = true;
    clearMap1dAndReserve(size);
    clearMap2dAndReserve(size);
    clearMap3dAndReserve(size);
}


template <class In, class Out>
typename BarycentricMapperMeshTopology<In, Out>::Index 
BarycentricMapperMeshTopology<In,Out>::addPointInLine ( const Index lineIndex, const SReal* baryCoords )
{
    m_map1d.resize ( m_map1d.size() +1 );
    MappingData1D& data = *m_map1d.rbegin();
    data.in_index = lineIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    return Index(m_map1d.size()-1);
}


template <class In, class Out>
typename BarycentricMapperMeshTopology<In, Out>::Index 
BarycentricMapperMeshTopology<In,Out>::addPointInTriangle ( const Index triangleIndex, const SReal* baryCoords )
{
    m_map2d.resize ( m_map2d.size() +1 );
    MappingData2D& data = *m_map2d.rbegin();
    data.in_index = triangleIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    return Index(m_map2d.size()-1);
}


template <class In, class Out>
typename BarycentricMapperMeshTopology<In, Out>::Index
BarycentricMapperMeshTopology<In,Out>::addPointInQuad ( const Index quadIndex, const SReal* baryCoords )
{
    m_map2d.resize ( m_map2d.size() +1 );
    MappingData2D& data = *m_map2d.rbegin();
    data.in_index = quadIndex + this->m_fromTopology->getNbTriangles();
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    return Index(m_map2d.size()-1);
}


template <class In, class Out>
typename BarycentricMapperMeshTopology<In, Out>::Index 
BarycentricMapperMeshTopology<In,Out>::addPointInTetra ( const Index tetraIndex, const SReal* baryCoords )
{
    m_map3d.resize ( m_map3d.size() +1 );
    MappingData3D& data = *m_map3d.rbegin();
    data.in_index = tetraIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    data.baryCoords[2] = ( Real ) baryCoords[2];
    return Index(m_map3d.size()-1);
}


template <class In, class Out>
typename BarycentricMapperMeshTopology<In, Out>::Index 
BarycentricMapperMeshTopology<In,Out>::addPointInCube ( const Index cubeIndex, const SReal* baryCoords )
{
    m_map3d.resize ( m_map3d.size() +1 );
    MappingData3D& data = *m_map3d.rbegin();
    data.in_index = cubeIndex + this->m_fromTopology->getNbTetrahedra();
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    data.baryCoords[2] = ( Real ) baryCoords[2];
    return Index(m_map3d.size()-1);
}


template <class In, class Out>
typename BarycentricMapperMeshTopology<In, Out>::Index 
BarycentricMapperMeshTopology<In,Out>::createPointInLine ( const typename Out::Coord& p, Index lineIndex, const typename In::VecCoord* points )
{
    SReal baryCoords[1];
    const Edge& elem = this->m_fromTopology->getLine ( lineIndex );
    const typename In::Coord p0 = ( *points ) [elem[0]];
    const typename In::Coord pA = ( *points ) [elem[1]] - p0;
    typename In::Coord pos = Out::getCPos(p) - p0;

    const SReal L2 = pA.norm2(); 
    if (L2 < std::numeric_limits<SReal>::epsilon()) // in case of null length edge, avoid division by 0
        baryCoords[0] = 0.0;
    else
        baryCoords[0] = ((pos * pA) / L2);

    return this->addPointInLine ( lineIndex, baryCoords );
}


template <class In, class Out>
typename BarycentricMapperMeshTopology<In, Out>::Index 
BarycentricMapperMeshTopology<In,Out>::createPointInTriangle ( const typename Out::Coord& p, Index triangleIndex, const typename In::VecCoord* points )
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
    sofa::type::Mat<2,2,typename In::Real> A;
    sofa::type::Vec<2,typename In::Real> b;
    A(0,0) = AB*AB;
    A(1,1) = AC*AC;
    A(0,1) = A(1,0) = AB*AC;
    b[0] = AQ*AB;
    b[1] = AQ*AC;
    const typename In::Real det = sofa::type::determinant(A);

    baryCoords[0] = (b[0]*A(1,1) - b[1]*A(0,1))/det;
    baryCoords[1]  = (b[1]*A(0,0) - b[0]*A(1,0))/det;

    if (baryCoords[0] < 0 || baryCoords[1] < 0 || baryCoords[0] + baryCoords[1] > 1)
    {
        // nearest point is on an edge or corner
        // barycentric coordinate on AB
        const SReal pAB = b[0] / A(0,0); // AQ*AB / AB*AB
        // barycentric coordinate on AC
        const SReal pAC = b[1] / A(1,1); // AQ*AC / AB*AB
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
            const SReal pBC = (b[1] - b[0] + A(0,0) - A(0,1)) / (A(0,0) + A(1,1) - 2*A(0,1)); // BQ*BC / BC*BC
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
typename BarycentricMapperMeshTopology<In, Out>::Index 
BarycentricMapperMeshTopology<In,Out>::createPointInQuad ( const typename Out::Coord& p, Index quadIndex, const typename In::VecCoord* points )
{
    SReal baryCoords[2];
    const Quad& elem = this->m_fromTopology->getQuad ( quadIndex );
    const typename In::Coord p0 = ( *points ) [elem[0]];
    const typename In::Coord pA = ( *points ) [elem[1]] - p0;
    const typename In::Coord pB = ( *points ) [elem[3]] - p0;
    typename In::Coord pos = Out::getCPos(p) - p0;
    sofa::type::Mat<3,3,typename In::Real> m,mt,base;
    m[0] = pA;
    m[1] = pB;
    m[2] = cross ( pA, pB );
    mt.transpose ( m );
    const bool canInvert = base.invert ( mt );
    assert(canInvert);
    SOFA_UNUSED(canInvert);
    const typename In::Coord base0 = base[0];
    const typename In::Coord base1 = base[1];
    baryCoords[0] = base0 * pos;
    baryCoords[1] = base1 * pos;
    return this->addPointInQuad ( quadIndex, baryCoords );
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
                    const Index index = m_map1d[indexIn].in_index;
                    {
                        const Edge& line = lines[index];
                        o.addCol( line[0], data * ( 1-fx ) );
                        o.addCol( line[1], data * fx );
                    }
                }
                // 2D elements : triangle or quad
                else if ( indexIn < i2d )
                {
                    const OutReal fx = ( OutReal ) m_map2d[indexIn].baryCoords[0];
                    const OutReal fy = ( OutReal ) m_map2d[indexIn].baryCoords[1];
                    const Index index = m_map2d[indexIn].in_index;
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
                    const size_t index = m_map3d[indexIn].in_index;
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
void BarycentricMapperMeshTopology<In,Out>::draw  (const core::visual::VisualParams* vparams,
                                                   const typename Out::VecCoord& out,
                                                   const typename In::VecCoord& in )
{
    const SeqLines& lines = this->m_fromTopology->getLines();
    const SeqTriangles& triangles = this->m_fromTopology->getTriangles();
    const SeqQuads& quads = this->m_fromTopology->getQuads();
    const SeqTetrahedra& tetrahedra = this->m_fromTopology->getTetrahedra();
    const SeqHexahedra& cubes = this->m_fromTopology->getHexahedra();

    std::vector< Vec3 > points;
    // 1D elements
    {
        const Index i0 = 0;
        for ( std::size_t i=0; i<m_map1d.size(); i++ )
        {
            const Real fx = m_map1d[i].baryCoords[0];
            const Index index = m_map1d[i].in_index;
            {
                const Edge& line = lines[index];
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
        const auto i0 = m_map1d.size();
        const auto c0 = triangles.size();
        for ( std::size_t i=0; i<m_map2d.size(); i++ )
        {
            const Real fx = m_map2d[i].baryCoords[0];
            const Real fy = m_map2d[i].baryCoords[1];
            const Index index = m_map2d[i].in_index;
            if ( index<c0 )
            {
                const Triangle& triangle = triangles[index];
                const Real f[3] = {( 1-fx-fy ), fx, fy};
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
        const auto i0 = m_map1d.size() +m_map2d.size();
        const auto c0 = tetrahedra.size();
        for ( std::size_t i=0; i<m_map3d.size(); i++ )
        {
            const Real fx = m_map3d[i].baryCoords[0];
            const Real fy = m_map3d[i].baryCoords[1];
            const Real fz = m_map3d[i].baryCoords[2];
            const Index index = m_map3d[i].in_index;
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
    vparams->drawTool()->drawLines ( points, 1, sofa::type::RGBAColor::green());
}

template <class In, class Out>
const sofa::linearalgebra::BaseMatrix* BarycentricMapperMeshTopology<In,Out>::getJ(int outSize, int inSize)
{

    if (m_matrixJ && !m_updateJ && m_matrixJ->rowBSize() == (MatrixTypeIndex)outSize && m_matrixJ->colBSize() == (MatrixTypeIndex)inSize)
        return m_matrixJ;
    if (outSize > 0 && m_map1d.size()+m_map2d.size()+m_map3d.size() == 0)
        return nullptr; // error: maps not yet created ?
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

    // Index of the current row. The matrix structure is composed of 3 parts: 1d elements first, then the 2d elements,
    // and finally the 3d elements. This variable is incremented at each element, and never reinitialized, so the
    // contributions go into the appropriate submatrix.
    int rowId {};

    // 1D elements
    {
        for ( const auto& map1d : m_map1d)
        {
            const auto fx = map1d.baryCoords[0];
            const auto index = map1d.in_index;
            const Edge& line = lines[index];
            this->addMatrixContrib(m_matrixJ, rowId, line[0],  1 - fx);
            this->addMatrixContrib(m_matrixJ, rowId, line[1],  fx);
            ++rowId;
        }
    }
    // 2D elements
    {
        const size_t c0 = triangles.size();
        for ( const auto& map2d : m_map2d)
        {
            const Real fx = map2d.baryCoords[0];
            const Real fy = map2d.baryCoords[1];
            const size_t index = map2d.in_index;
            if ( index < c0 )
            {
                const Triangle& triangle = triangles[index];
                this->addMatrixContrib(m_matrixJ, rowId, triangle[0],  ( 1-fx-fy ));
                this->addMatrixContrib(m_matrixJ, rowId, triangle[1],  fx);
                this->addMatrixContrib(m_matrixJ, rowId, triangle[2],  fy);
            }
            else
            {
                const Quad& quad = quads[index-c0];
                this->addMatrixContrib(m_matrixJ, rowId, quad[0],  ( ( 1-fx ) * ( 1-fy ) ));
                this->addMatrixContrib(m_matrixJ, rowId, quad[1],  ( ( fx ) * ( 1-fy ) ));
                this->addMatrixContrib(m_matrixJ, rowId, quad[3],  ( ( 1-fx ) * ( fy ) ));
                this->addMatrixContrib(m_matrixJ, rowId, quad[2],  ( ( fx ) * ( fy ) ));
            }
            ++rowId;
        }
    }
    // 3D elements
    {
        const size_t c0 = tetrahedra.size();
        for ( const auto& map3d : m_map3d)
        {
            const Real fx = map3d.baryCoords[0];
            const Real fy = map3d.baryCoords[1];
            const Real fz = map3d.baryCoords[2];
            const size_t index = map3d.in_index;
            if ( index<c0 )
            {
                const Tetra& tetra = tetrahedra[index];
                this->addMatrixContrib(m_matrixJ, rowId, tetra[0],  ( 1-fx-fy-fz ));
                this->addMatrixContrib(m_matrixJ, rowId, tetra[1],  fx);
                this->addMatrixContrib(m_matrixJ, rowId, tetra[2],  fy);
                this->addMatrixContrib(m_matrixJ, rowId, tetra[3],  fz);
            }
            else
            {
                const Hexa& cube = cubes[index-c0];

                this->addMatrixContrib(m_matrixJ, rowId, cube[0],  ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) ));
                this->addMatrixContrib(m_matrixJ, rowId, cube[1],  ( ( fx ) * ( 1-fy ) * ( 1-fz ) ));

                this->addMatrixContrib(m_matrixJ, rowId, cube[3],  ( ( 1-fx ) * ( fy ) * ( 1-fz ) ));
                this->addMatrixContrib(m_matrixJ, rowId, cube[2],  ( ( fx ) * ( fy ) * ( 1-fz ) ));

                this->addMatrixContrib(m_matrixJ, rowId, cube[4],  ( ( 1-fx ) * ( 1-fy ) * ( fz ) ));
                this->addMatrixContrib(m_matrixJ, rowId, cube[5],  ( ( fx ) * ( 1-fy ) * ( fz ) ));

                this->addMatrixContrib(m_matrixJ, rowId, cube[7],  ( ( 1-fx ) * ( fy ) * ( fz ) ));
                this->addMatrixContrib(m_matrixJ, rowId, cube[6],  ( ( fx ) * ( fy ) * ( fz ) ));
            }
            ++rowId;
        }
    }
    assert(static_cast<std::size_t>(rowId) == m_map1d.size() + m_map2d.size() + m_map3d.size());

    m_matrixJ->compress();
    m_updateJ = false;
    return m_matrixJ;
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

    for( size_t i=0 ; i<in.size() ; ++i)
    {
        // 1D elements
        if (i < i1d)
        {
            const typename Out::DPos v = Out::getDPos(in[i]);
            const OutReal fx = ( OutReal ) m_map1d[i].baryCoords[0];
            const size_t index = m_map1d[i].in_index;
            {
                const Edge& line = lines[index];
                out[line[0]] += v * ( 1-fx );
                out[line[1]] += v * fx;
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
            const size_t index = m_map2d[i-i0].in_index;
            if ( index<c0 )
            {
                const Triangle& triangle = triangles[index];
                out[triangle[0]] += v * ( 1-fx-fy );
                out[triangle[1]] += v * fx;
                out[triangle[2]] += v * fy;
            }
            else
            {
                const Quad& quad = quads[index-c0];
                out[quad[0]] += v * ( ( 1-fx ) * ( 1-fy ) );
                out[quad[1]] += v * ( ( fx ) * ( 1-fy ) );
                out[quad[3]] += v * ( ( 1-fx ) * ( fy ) );
                out[quad[2]] += v * ( ( fx ) * ( fy ) );
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
            const size_t index = m_map3d[i-i0].in_index;
            if ( index<c0 )
            {
                const Tetra& tetra = tetrahedra[index];
                out[tetra[0]] += v * ( 1-fx-fy-fz );
                out[tetra[1]] += v * fx;
                out[tetra[2]] += v * fy;
                out[tetra[3]] += v * fz;
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

            }
        }
    }
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

    for( size_t i=0 ; i<out.size() ; ++i)
    {
        // 1D elements
        if (i < idxStart1)
        {
            const Real fx = m_map1d[i].baryCoords[0];
            const Index index = m_map1d[i].in_index;
            {
                const Edge& line = lines[index];
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
            const size_t index = m_map2d[i-i0].in_index;

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
            const size_t index = m_map3d[i-i0].in_index;
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
void BarycentricMapperMeshTopology<In,Out>::resize( core::State<Out>* toModel )
{
    toModel->resize(sofa::Size(m_map1d.size() +m_map2d.size() +m_map3d.size()));
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
        for ( std::size_t i=0; i<m_map1d.size(); i++ )
        {
            const Real fx = m_map1d[i].baryCoords[0];
            const Index index = m_map1d[i].in_index;
            {
                const Edge& line = lines[index];
                Out::setCPos(out[i] , in[line[0]] * ( 1-fx )
                        + in[line[1]] * fx );
            }
        }
    }
    // 2D elements
    {
        const std::size_t i0 = m_map1d.size();
        const std::size_t c0 = triangles.size();
        for ( std::size_t i=0; i<m_map2d.size(); i++ )
        {
            const Real fx = m_map2d[i].baryCoords[0];
            const Real fy = m_map2d[i].baryCoords[1];
            const Index index = m_map2d[i].in_index;
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
        const std::size_t i0 = m_map1d.size() + m_map2d.size();
        const std::size_t c0 = tetrahedra.size();
        for ( std::size_t i=0; i<m_map3d.size(); i++ )
        {
            const Real fx = m_map3d[i].baryCoords[0];
            const Real fy = m_map3d[i].baryCoords[1];
            const Real fz = m_map3d[i].baryCoords[2];
            const Index index = m_map3d[i].in_index;
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
std::istream& operator >> ( std::istream& in, BarycentricMapperMeshTopology<In, Out> &b )
{
    std::size_t size_vec;
    in >> size_vec;
    b.m_map1d.clear();
    typename BarycentricMapperMeshTopology<In, Out>::MappingData1D value1d;
    for (std::size_t i=0; i<size_vec; i++)
    {
        in >> value1d;
        b.m_map1d.push_back(value1d);
    }

    in >> size_vec;
    b.m_map2d.clear();
    typename BarycentricMapperMeshTopology<In, Out>::MappingData2D value2d;
    for (std::size_t i=0; i<size_vec; i++)
    {
        in >> value2d;
        b.m_map2d.push_back(value2d);
    }

    in >> size_vec;
    b.m_map3d.clear();
    typename BarycentricMapperMeshTopology<In, Out>::MappingData3D value3d;
    for (std::size_t i=0; i<size_vec; i++)
    {
        in >> value3d;
        b.m_map3d.push_back(value3d);
    }
    return in;
}

template <class In, class Out>
std::ostream& operator << ( std::ostream& out, const BarycentricMapperMeshTopology<In, Out> & b )
{

    out << b.m_map1d.size();
    out << " " ;
    out << b.m_map1d;
    out << " " ;
    out << b.m_map2d.size();
    out << " " ;
    out << b.m_map2d;
    out << " " ;
    out << b.m_map3d.size();
    out << " " ;
    out << b.m_map3d;

    return out;
}

} // namespace sofa::component::mapping::linear
