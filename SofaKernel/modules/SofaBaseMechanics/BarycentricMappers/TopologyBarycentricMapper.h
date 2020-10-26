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
#ifndef SOFA_COMPONENT_MAPPING_TOPOLOGYBARYCENTRICMAPPER_H
#define SOFA_COMPONENT_MAPPING_TOPOLOGYBARYCENTRICMAPPER_H
#include <SofaBaseMechanics/BarycentricMappers/BarycentricMapper.h>
#include <SofaBaseTopology/PointSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace mapping
{

namespace _topologybarycentricmapper_
{

using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;

/// Template class for barycentric mapping topology-specific mappers.
template<class In, class Out>
class TopologyBarycentricMapper : public BarycentricMapper<In,Out>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(TopologyBarycentricMapper,In,Out),
               SOFA_TEMPLATE2(BarycentricMapper,In,Out));

    typedef typename Inherit1::Real Real;
    typedef typename core::behavior::BaseMechanicalState::ForceMask ForceMask;

    using Index = sofa::Index;

    ForceMask *maskFrom;
    ForceMask *maskTo;

    virtual Index addPointInLine(const Index lineIndex, const SReal* baryCoords);
    virtual Index setPointInLine(const Index pointIndex, const Index lineIndex, const SReal* baryCoords);
    virtual Index createPointInLine(const typename Out::Coord& p, Index lineIndex, const typename In::VecCoord* points);

    virtual Index addPointInTriangle(const Index triangleIndex, const SReal* baryCoords);
    virtual Index setPointInTriangle(const Index pointIndex, const Index triangleIndex, const SReal* baryCoords);
    virtual Index createPointInTriangle(const typename Out::Coord& p, Index triangleIndex, const typename In::VecCoord* points);

    virtual Index addPointInQuad(const Index quadIndex, const SReal* baryCoords);
    virtual Index setPointInQuad(const Index pointIndex, const Index quadIndex, const SReal* baryCoords);
    virtual Index createPointInQuad(const typename Out::Coord& p, Index quadIndex, const typename In::VecCoord* points);

    virtual Index addPointInTetra(const Index tetraIndex, const SReal* baryCoords);
    virtual Index setPointInTetra(const Index pointIndex, const Index tetraIndex, const SReal* baryCoords);
    virtual Index createPointInTetra(const typename Out::Coord& p, Index tetraIndex, const typename In::VecCoord* points);

    virtual Index addPointInCube(const Index cubeIndex, const SReal* baryCoords);
    virtual Index setPointInCube(const Index pointIndex, const Index cubeIndex, const SReal* baryCoords);
    virtual Index createPointInCube(const typename Out::Coord& p, Index cubeIndex, const typename In::VecCoord* points);

    virtual void setToTopology( topology::PointSetTopologyContainer* toTopology) {this->m_toTopology = toTopology;}
    const topology::PointSetTopologyContainer *getToTopology() const {return m_toTopology;}

    virtual void updateForceMask(){/*mask is already filled in the mapper's applyJT*/}
    virtual void resize( core::State<Out>* toModel ) = 0;

    void processTopologicalChanges(const typename Out::VecCoord& out, const typename In::VecCoord& in, core::topology::Topology* t) {
        SOFA_UNUSED(t);
        this->clear();
        this->init(out,in);
    }

protected:
    TopologyBarycentricMapper(core::topology::BaseMeshTopology* fromTopology,
                              topology::PointSetTopologyContainer* toTopology = nullptr)
        : m_fromTopology(fromTopology)
        , m_toTopology(toTopology)
    {}

    ~TopologyBarycentricMapper() override {}

    core::topology::BaseMeshTopology*    m_fromTopology;
    topology::PointSetTopologyContainer* m_toTopology;
};

#if !defined(SOFA_COMPONENT_MAPPING_TOPOLOGYBARYCENTRICMAPPER_CPP)
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3dTypes, Vec3dTypes >;


#endif

}

using _topologybarycentricmapper_::TopologyBarycentricMapper;

}}}


#endif
