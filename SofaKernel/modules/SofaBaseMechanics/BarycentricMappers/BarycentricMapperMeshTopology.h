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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERMESHTOPOLOGY_H
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERMESHTOPOLOGY_H

#include <SofaBaseMechanics/BarycentricMappers/TopologyBarycentricMapper.h>

namespace sofa
{

namespace component
{

namespace mapping
{

/// Groupe the using as early as possible to make very obvious what are the
/// external dependencies of the following code.
using core::visual::VisualParams;
using sofa::defaulttype::BaseMatrix;
using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;

/// Class allowing barycentric mapping computation on a MeshTopology
template<class In, class Out>
class BarycentricMapperMeshTopology : public TopologyBarycentricMapper<In,Out>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapperMeshTopology,In,Out),
               SOFA_TEMPLATE2(TopologyBarycentricMapper,In,Out));

    typedef typename Inherit1::Real Real;
    typedef typename Inherit1::OutReal OutReal;
    typedef typename Inherit1::OutDeriv  OutDeriv;

    typedef typename Inherit1::InDeriv  InDeriv;
    typedef typename Inherit1::MappingData1D MappingData1D;
    typedef typename Inherit1::MappingData2D MappingData2D;
    typedef typename Inherit1::MappingData3D MappingData3D;

    enum { NIn = Inherit1::NIn };
    enum { NOut = Inherit1::NOut };
    typedef typename Inherit1::MBloc MBloc;
    typedef typename Inherit1::MatrixType MatrixType;
    typedef typename MatrixType::Index MatrixTypeIndex;

    typedef typename Inherit1::ForceMask ForceMask;

    using index_type = sofa::defaulttype::index_type;

public:
    void clear(std::size_t reserve=0) override;
    void resize( core::State<Out>* toModel ) override;
    index_type addPointInLine(const index_type lineIndex, const SReal* baryCoords) override;
    index_type createPointInLine(const typename Out::Coord& p, index_type lineIndex, const typename In::VecCoord* points) override;
    index_type addPointInTriangle(const index_type triangleIndex, const SReal* baryCoords) override;
    index_type createPointInTriangle(const typename Out::Coord& p, index_type triangleIndex, const typename In::VecCoord* points) override;
    index_type addPointInQuad(const index_type quadIndex, const SReal* baryCoords) override;
    index_type createPointInQuad(const typename Out::Coord& p, index_type quadIndex, const typename In::VecCoord* points) override;
    index_type addPointInTetra(const index_type tetraIndex, const SReal* baryCoords) override;
    index_type addPointInCube(const index_type cubeIndex, const SReal* baryCoords) override;

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) override;
    void draw(const core::visual::VisualParams*,const typename Out::VecCoord& out, const typename In::VecCoord& in) override;

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) override;
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) override;
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) override;
    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) override;
    const BaseMatrix* getJ(int outSize, int inSize) override;

    sofa::helper::vector< MappingData3D > const* getMap3d() const { return &m_map3d; }

    template<class I, class O>
    friend std::istream& operator >> ( std::istream& in, BarycentricMapperMeshTopology<I, O> &b );
    template<class I, class O>
    friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperMeshTopology<I, O> & b );

    ~BarycentricMapperMeshTopology() override ;

protected:
    BarycentricMapperMeshTopology(core::topology::BaseMeshTopology* fromTopology,
                                  topology::PointSetTopologyContainer* toTopology) ;

    void addMatrixContrib(MatrixType* m, int row, int col, Real value);

    sofa::helper::vector< MappingData1D >  m_map1d;
    sofa::helper::vector< MappingData2D >  m_map2d;
    sofa::helper::vector< MappingData3D >  m_map3d;

    MatrixType* m_matrixJ {nullptr};
    bool        m_updateJ {false};
private:
    void clearMap1dAndReserve(std::size_t size=0);
    void clearMap2dAndReserve(std::size_t size=0);
    void clearMap3dAndReserve(std::size_t size=0);
};

#if !defined(SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERMESHTOPOLOGY_CPP)
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< Vec3dTypes, Vec3dTypes >;


#endif

}}}


#endif
