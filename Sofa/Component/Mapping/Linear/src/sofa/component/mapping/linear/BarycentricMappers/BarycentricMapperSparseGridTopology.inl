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
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperSparseGridTopology.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/State.h>

namespace sofa::component::mapping::linear
{

using sofa::type::Vec3;
using sofa::core::visual::VisualParams;
using sofa::type::Vec;

template<class In, class Out>
BarycentricMapperSparseGridTopology<In, Out>::BarycentricMapperSparseGridTopology(topology::container::grid::SparseGridTopology* fromTopology,
    core::topology::BaseMeshTopology* _toTopology)
    : TopologyBarycentricMapper<In,Out>(fromTopology, _toTopology),
      m_fromTopology(fromTopology),
      m_matrixJ(nullptr), m_updateJ(true)
{
}

template<class In, class Out>
BarycentricMapperSparseGridTopology<In, Out>::~BarycentricMapperSparseGridTopology()
{
    if (m_matrixJ)
        delete m_matrixJ;
}

template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::clear ( std::size_t size )
{
    m_updateJ = true;
    m_map.clear();
    if ( size>0 ) m_map.reserve ( size );
}


template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::resize( core::State<Out>* toModel )
{
    toModel->resize(Size(m_map.size()));
}


template <class In, class Out>
typename BarycentricMapperSparseGridTopology<In, Out>::Index
BarycentricMapperSparseGridTopology<In,Out>::addPointInCube ( const Index cubeIndex, const SReal* baryCoords )
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
    m_updateJ = true;
    clear ( (int)out.size() );

    if ( m_fromTopology->isVolume() )
    {
        for ( unsigned int i=0; i<out.size(); i++ )
        {
            sofa::type::Vec3 coefs;
            Index cube = m_fromTopology->findCube ( Out::getCPos(out[i]), coefs[0], coefs[1], coefs[2] );
            if ( cube==sofa::InvalidID )
            {
                cube = m_fromTopology->findNearestCube ( Out::getCPos(out[i]), coefs[0], coefs[1], coefs[2] );
            }

            this->addPointInCube ( cube, coefs.ptr() );
        }
    }
}


template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::draw  (const VisualParams* vparams,
                                                         const typename Out::VecCoord& out,
                                                         const typename In::VecCoord& in )
{
    std::vector< Vec3 > points;
    for ( unsigned int i=0; i<m_map.size(); i++ )
    {

        const topology::container::grid::SparseGridTopology::Hexa cube = this->m_fromTopology->getHexahedron ( this->m_map[i].in_index );

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
    vparams->drawTool()->drawLines ( points, 1, sofa::type::RGBAColor::blue());
}


template <class In, class Out>
const sofa::linearalgebra::BaseMatrix* BarycentricMapperSparseGridTopology<In,Out>::getJ(int outSize, int inSize)
{
    if (m_matrixJ && !m_updateJ)
        return m_matrixJ;

    if (!m_matrixJ) m_matrixJ = new MatrixType;
    if (m_matrixJ->rowBSize() != (MatrixTypeIndex)outSize || m_matrixJ->colBSize() != (MatrixTypeIndex)inSize)
        m_matrixJ->resize(outSize*NOut, inSize*NIn);
    else
        m_matrixJ->clear();

    for ( size_t i=0; i<m_map.size(); i++ )
    {
        const int out = int(i);

        const topology::container::grid::SparseGridTopology::Hexa cube = this->m_fromTopology->getHexahedron ( this->m_map[i].in_index );

        const Real fx = ( Real ) m_map[i].baryCoords[0];
        const Real fy = ( Real ) m_map[i].baryCoords[1];
        const Real fz = ( Real ) m_map[i].baryCoords[2];
        this->addMatrixContrib(m_matrixJ, out, cube[0], ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) ));
        this->addMatrixContrib(m_matrixJ, out, cube[1], ( ( fx ) * ( 1-fy ) * ( 1-fz ) ));

        this->addMatrixContrib(m_matrixJ, out, cube[3], ( ( 1-fx ) * ( fy ) * ( 1-fz ) ));
        this->addMatrixContrib(m_matrixJ, out, cube[2], ( ( fx ) * ( fy ) * ( 1-fz ) ));

        this->addMatrixContrib(m_matrixJ, out, cube[4], ( ( 1-fx ) * ( 1-fy ) * ( fz ) ));
        this->addMatrixContrib(m_matrixJ, out, cube[5], ( ( fx ) * ( 1-fy ) * ( fz ) ));

        this->addMatrixContrib(m_matrixJ, out, cube[7], ( ( 1-fx ) * ( fy ) * ( fz ) ));
        this->addMatrixContrib(m_matrixJ, out, cube[6], ( ( fx ) * ( fy ) * ( fz ) ));
    }
    m_matrixJ->compress();
    m_updateJ = false;
    return m_matrixJ;
}


template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const auto& hexahedra = this->m_fromTopology->getHexahedra();

    for( size_t index=0 ; index<in.size() ; ++index)
    {
        const typename Out::DPos v = Out::getDPos(in[index]);

        assert(this->m_map[index].in_index < hexahedra.size());
        const topology::container::grid::SparseGridTopology::Hexa& cube = hexahedra[this->m_map[index].in_index];

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
    }
}


template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
{
    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    const auto& hexahedra = this->m_fromTopology->getHexahedra();

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

                assert(this->m_map[indexIn].in_index < hexahedra.size());
                const topology::container::grid::SparseGridTopology::Hexa& cube = hexahedra[this->m_map[indexIn].in_index];

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


template <class In, class Out>
void BarycentricMapperSparseGridTopology<In,Out>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize( m_map.size() );

    const auto& hexahedra = this->m_fromTopology->getHexahedra();

    for( size_t index=0 ; index<out.size() ; ++index)
    {
        assert(this->m_map[index].in_index < hexahedra.size());
        const topology::container::grid::SparseGridTopology::Hexa& cube = hexahedra[this->m_map[index].in_index];

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
void BarycentricMapperSparseGridTopology<In,Out>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize( m_map.size() );

    typedef type::vector< CubeData > CubeDataVector;
    typedef typename CubeDataVector::const_iterator CubeDataVectorIt;

    CubeDataVectorIt it = m_map.begin();
    CubeDataVectorIt itEnd = m_map.end();

    unsigned int i = 0;

    const auto& hexahedra = this->m_fromTopology->getHexahedra();

    while (it != itEnd)
    {
        assert(it->in_index < hexahedra.size());
        const topology::container::grid::SparseGridTopology::Hexa& cube = hexahedra[it->in_index];

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



template<class In, class Out>
std::istream& operator >> ( std::istream& in, BarycentricMapperSparseGridTopology<In, Out> &b )
{
    in >> b.m_map;
    return in;
}

template<class In, class Out>
std::ostream& operator << ( std::ostream& out, const BarycentricMapperSparseGridTopology<In, Out> & b )
{
    out << b.m_map;
    return out;
}

} // namespace sofa::component::mapping::linear
