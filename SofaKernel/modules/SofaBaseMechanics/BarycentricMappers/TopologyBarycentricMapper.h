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
using sofa::defaulttype::ExtVec3fTypes;

/// Template class for barycentric mapping topology-specific mappers.
template<class In, class Out>
class TopologyBarycentricMapper : public BarycentricMapper<In,Out>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(TopologyBarycentricMapper,In,Out),
               SOFA_TEMPLATE2(BarycentricMapper,In,Out));

    typedef typename Inherit1::Real Real;
    typedef typename core::behavior::BaseMechanicalState::ForceMask ForceMask;

    ForceMask *maskFrom;
    ForceMask *maskTo;

    virtual int addPointInLine(const int lineIndex, const SReal* baryCoords);
    virtual int setPointInLine(const int pointIndex, const int lineIndex, const SReal* baryCoords);
    virtual int createPointInLine(const typename Out::Coord& p, int lineIndex, const typename In::VecCoord* points);

    virtual int addPointInTriangle(const int triangleIndex, const SReal* baryCoords);
    virtual int setPointInTriangle(const int pointIndex, const int triangleIndex, const SReal* baryCoords);
    virtual int createPointInTriangle(const typename Out::Coord& p, int triangleIndex, const typename In::VecCoord* points);

    virtual int addPointInQuad(const int quadIndex, const SReal* baryCoords);
    virtual int setPointInQuad(const int pointIndex, const int quadIndex, const SReal* baryCoords);
    virtual int createPointInQuad(const typename Out::Coord& p, int quadIndex, const typename In::VecCoord* points);

    virtual int addPointInTetra(const int tetraIndex, const SReal* baryCoords);
    virtual int setPointInTetra(const int pointIndex, const int tetraIndex, const SReal* baryCoords);
    virtual int createPointInTetra(const typename Out::Coord& p, int tetraIndex, const typename In::VecCoord* points);

    virtual int addPointInCube(const int cubeIndex, const SReal* baryCoords);
    virtual int setPointInCube(const int pointIndex, const int cubeIndex, const SReal* baryCoords);
    virtual int createPointInCube(const typename Out::Coord& p, int cubeIndex, const typename In::VecCoord* points);

    virtual void setToTopology( topology::PointSetTopologyContainer* toTopology) {this->m_toTopology = toTopology;}
    const topology::PointSetTopologyContainer *getToTopology() const {return m_toTopology;}

    virtual void updateForceMask(){/*mask is already filled in the mapper's applyJT*/}
    virtual void resize( core::State<Out>* toModel ) = 0;

protected:
    TopologyBarycentricMapper(core::topology::BaseMeshTopology* fromTopology,
                              topology::PointSetTopologyContainer* toTopology = nullptr)
        : m_fromTopology(fromTopology)
        , m_toTopology(toTopology)
    {}

    virtual ~TopologyBarycentricMapper() override {}

    core::topology::BaseMeshTopology*    m_fromTopology;
    topology::PointSetTopologyContainer* m_toTopology;
};

#if !defined(SOFA_COMPONENT_MAPPING_TOPOLOGYBARYCENTRICMAPPER_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3dTypes, ExtVec3fTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3fTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3fTypes, ExtVec3fTypes >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3dTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3fTypes, Vec3dTypes >;
#endif
#endif
#endif

}

using _topologybarycentricmapper_::TopologyBarycentricMapper;

}}}


#endif
