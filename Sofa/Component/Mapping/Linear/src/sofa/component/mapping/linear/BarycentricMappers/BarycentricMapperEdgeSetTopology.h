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

/////// Class allowing barycentric mapping computation on a BaseMeshTopology with edges
template<class In, class Out>
class BarycentricMapperEdgeSetTopology : public BarycentricMapperTopologyContainer<In,Out,typename BarycentricMapper<In,Out>::MappingData1D, sofa::core::topology::BaseMeshTopology::Edge>
{
    typedef typename BarycentricMapper<In,Out>::MappingData1D MappingData;
    using Edge = sofa::core::topology::BaseMeshTopology::Edge;
    
    using Index = sofa::Index;
    using Mat3x3 = sofa::type::Mat3x3;
    using Vec3 = sofa::type::Vec3;

public:
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapperEdgeSetTopology,In,Out),SOFA_TEMPLATE4(BarycentricMapperTopologyContainer,In,Out,MappingData,Edge));
    typedef typename Inherit1::Real Real;

public:

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) override
    {
        SOFA_UNUSED(out);
        SOFA_UNUSED(in);
        msg_warning() << "BarycentricMapping not implemented for Topologies with edges.";
    }
    Index addPointInLine(const Index edgeIndex, const SReal* baryCoords) override;
    Index createPointInLine(const typename Out::Coord& p, Index edgeIndex, const typename In::VecCoord* points) override;


protected:
    BarycentricMapperEdgeSetTopology(sofa::core::topology::TopologyContainer* fromTopology,
        sofa::core::topology::BaseMeshTopology* toTopology);

    ~BarycentricMapperEdgeSetTopology() override = default;

    virtual type::vector<Edge> getElements() override;
    virtual std::array<Real, Edge::NumberOfNodes>getBarycentricCoefficients(const std::array<Real, MappingData::NumberOfCoordinates>& barycentricCoordinates) override;
    void computeBase(Mat3x3& base, const typename In::VecCoord& in, const Edge& element) override;
    void computeCenter(Vec3& center, const typename In::VecCoord& in, const Edge& element) override;
    void computeDistance(SReal& d, const Vec3& v) override;
    void addPointInElement(const Index elementIndex, const SReal* baryCoords) override;

    using Inherit1::d_map;
    using Inherit1::m_fromTopology;
    using Inherit1::m_matrixJ;
    using Inherit1::m_updateJ;
};

#if !defined(SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPEREDGESETTOPOLOGY_CPP)
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API BarycentricMapperEdgeSetTopology< sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types>;

#endif

} // namespace sofa::component::mapping::linear
