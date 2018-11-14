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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERQUADSETTOPOLOGY_H
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERQUADSETTOPOLOGY_H

#include <SofaBaseMechanics/BarycentricMappers/BarycentricMapperTopologyContainer.h>
#include <SofaBaseTopology/QuadSetGeometryAlgorithms.h>
#include <SofaBaseTopology/QuadSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using sofa::defaulttype::Mat3x3d;
using sofa::defaulttype::Vector3;

typedef typename sofa::core::topology::BaseMeshTopology::Quad Quad;

/// Class allowing barycentric mapping computation on a QuadSetTopology
template<class In, class Out>
class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology : public BarycentricMapperTopologyContainer<In,Out,typename BarycentricMapper<In,Out>::MappingData2D, Quad>
{
    typedef typename BarycentricMapper<In,Out>::MappingData2D MappingData;

public:

    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapperQuadSetTopology,In,Out),SOFA_TEMPLATE4(BarycentricMapperTopologyContainer,In,Out,MappingData,Quad));
    typedef typename Inherit1::Real Real;

protected:

    topology::QuadSetTopologyContainer*			m_fromContainer;
    topology::QuadSetGeometryAlgorithms<In>*	m_fromGeomAlgo;

    using Inherit1::d_map;
    using Inherit1::m_fromTopology;
    using Inherit1::m_matrixJ;
    using Inherit1::m_updateJ;

    BarycentricMapperQuadSetTopology(topology::QuadSetTopologyContainer* fromTopology, topology::PointSetTopologyContainer* toTopology)
        : Inherit1(fromTopology, toTopology),
          m_fromContainer(fromTopology),
          m_fromGeomAlgo(NULL)
    {}

    virtual ~BarycentricMapperQuadSetTopology() {}

    virtual helper::vector<Quad> getElements() override;
    virtual helper::vector<SReal> getBaryCoef(const Real* f) override;
    helper::vector<SReal> getBaryCoef(const Real fx, const Real fy);
    virtual void computeBase(Mat3x3d& base, const typename In::VecCoord& in, const Quad& element) override;
    virtual void computeCenter(Vector3& center, const typename In::VecCoord& in, const Quad& element) override;
    virtual void computeDistance(double& d, const Vector3& v) override;
    virtual void addPointInElement(const int elementIndex, const SReal* baryCoords) override;

public:

    virtual int addPointInQuad(const int index, const SReal* baryCoords) override;
    virtual int createPointInQuad(const typename Out::Coord& p, int index, const typename In::VecCoord* points) override;
};


using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::ExtVec3fTypes;


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERQUADSETTOPOLOGY_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< Vec3dTypes, ExtVec3fTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< Vec3fTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< Vec3fTypes, ExtVec3fTypes >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< Vec3dTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< Vec3fTypes, Vec3dTypes >;
#endif
#endif
#endif

}}}

#endif
