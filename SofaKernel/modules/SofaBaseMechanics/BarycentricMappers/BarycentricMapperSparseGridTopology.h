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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERSPARSEGRIDTOPOLOGY_H
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERSPARSEGRIDTOPOLOGY_H

#include <SofaBaseMechanics/BarycentricMappers/TopologyBarycentricMapper.h>
#include <SofaBaseTopology/SparseGridTopology.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using sofa::defaulttype::BaseMatrix;
using core::visual::VisualParams;
using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::ExtVec3Types;

/// Class allowing barycentric mapping computation on a SparseGridTopology
template<class In, class Out>
class BarycentricMapperSparseGridTopology : public TopologyBarycentricMapper<In,Out>
{
public:

    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapperSparseGridTopology,In,Out),SOFA_TEMPLATE2(TopologyBarycentricMapper,In,Out));
    typedef typename Inherit1::Real Real;
    typedef typename Inherit1::OutReal OutReal;
    typedef typename Inherit1::OutDeriv  OutDeriv;
    typedef typename Inherit1::InDeriv  InDeriv;
    typedef typename Inherit1::CubeData CubeData;
    typedef typename Inherit1::MBloc MBloc;
    typedef typename Inherit1::MatrixType MatrixType;
    typedef typename MatrixType::Index MatrixTypeIndex;
    typedef typename Inherit1::ForceMask ForceMask;
    enum { NIn = Inherit1::NIn };
    enum { NOut = Inherit1::NOut };

public:
    virtual ~BarycentricMapperSparseGridTopology() override ;

    virtual void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) override;

    virtual void clear(int reserve=0) override;
    virtual int addPointInCube(const int cubeIndex, const SReal* baryCoords) override;

    virtual void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) override;
    virtual void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) override;
    virtual void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) override;
    virtual void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) override;
    virtual const BaseMatrix* getJ(int outSize, int inSize) override;

    virtual void draw(const VisualParams*,const typename Out::VecCoord& out, const typename In::VecCoord& in) override;
    virtual void resize( core::State<Out>* toModel ) override;

    template<class _In, class _Out>
    inline friend std::istream& operator >> ( std::istream& in, BarycentricMapperSparseGridTopology<_In, _Out> &b );
    template<class _In, class _Out>
    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperSparseGridTopology<_In, _Out> & b );

protected:
    BarycentricMapperSparseGridTopology(topology::SparseGridTopology* fromTopology,
                                        topology::PointSetTopologyContainer* _toTopology);

    void addMatrixContrib(MatrixType* m, int row, int col, Real value);

    sofa::helper::vector<CubeData> m_map;
    topology::SparseGridTopology* m_fromTopology {nullptr};
    MatrixType* m_matrixJ {nullptr};
    bool m_updateJ {false};
};

#if !defined(SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERSPARSEGRIDTOPOLOGY_CPP)
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< Vec3dTypes, ExtVec3Types >;


#endif

}}}


#endif
