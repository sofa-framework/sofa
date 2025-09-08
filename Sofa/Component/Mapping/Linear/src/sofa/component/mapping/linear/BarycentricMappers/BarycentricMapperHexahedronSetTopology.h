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
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperTopologyContainer.h>

namespace sofa::component::mapping::linear
{

/// Class allowing barycentric mapping computation on a HexahedronSetTopology
template<class In, class Out>
class BarycentricMapperHexahedronSetTopology : public BarycentricMapperTopologyContainer<In,Out,typename BarycentricMapper<In, Out>::MappingData3D, sofa::core::topology::BaseMeshTopology::Hexahedron>
{
    typedef typename BarycentricMapper<In, Out>::MappingData3D MappingData;
    using Hexahedron = sofa::core::topology::BaseMeshTopology::Hexahedron;

    using Mat3x3 = sofa::type::Mat3x3;
    using Vec3 = sofa::type::Vec3;
    using Index = sofa::Index;

public:
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapperHexahedronSetTopology,In,Out),
               SOFA_TEMPLATE4(BarycentricMapperTopologyContainer,In,Out,MappingData,Hexahedron));

    typedef typename Inherit1::Real Real;

    ~BarycentricMapperHexahedronSetTopology() override = default;
    virtual type::vector<Hexahedron> getElements() override;
    virtual std::array<Real, Hexahedron::NumberOfNodes>getBarycentricCoefficients(const std::array<Real, MappingData::NumberOfCoordinates>& barycentricCoordinates) override;
    void computeBase(Mat3x3& base, const typename In::VecCoord& in, const Hexahedron& element) override;
    void computeCenter(Vec3& center, const typename In::VecCoord& in, const Hexahedron& element) override;
    void computeDistance(SReal& d, const Vec3& v) override;
    void addPointInElement(const Index elementIndex, const SReal* baryCoords) override;

    Index addPointInCube(const Index index, const SReal* baryCoords) override;
    Index setPointInCube(const Index pointIndex, const Index cubeIndex, const SReal* baryCoords) override;
    void applyOnePoint( const Index& hexaId, typename Out::VecCoord& out, const typename In::VecCoord& in) override;
    void handleTopologyChange(core::topology::Topology* t) override;

protected:
    BarycentricMapperHexahedronSetTopology();
    BarycentricMapperHexahedronSetTopology(sofa::core::topology::TopologyContainer* fromTopology,
        core::topology::BaseMeshTopology* toTopology);

    void setTopology(sofa::core::topology::TopologyContainer* topology);

    std::set<Index> m_invalidIndex;

    using Inherit1::d_map;
    using Inherit1::m_matrixJ;
    using Inherit1::m_updateJ;
    using Inherit1::m_fromTopology;
};

#if !defined(SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERHEXAHEDRONSETTOPOLOGY_CPP)
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API BarycentricMapperHexahedronSetTopology< sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types >;


#endif

} // namespace sofa::component::mapping::linear
