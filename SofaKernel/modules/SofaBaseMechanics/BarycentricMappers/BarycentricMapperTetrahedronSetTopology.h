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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTETRAHEDRONSETTOPOLOGY_H
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTETRAHEDRONSETTOPOLOGY_H

#include <SofaBaseMechanics/BarycentricMappers/BarycentricMapperTopologyContainer.h>
#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.h>

namespace sofa
{

namespace component
{

namespace mapping
{


using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::ExtVec3Types;
using sofa::defaulttype::Mat3x3d;
using sofa::defaulttype::Vector3;
typedef typename sofa::core::topology::BaseMeshTopology::Tetrahedron Tetrahedron;

/// Class allowing barycentric mapping computation on a TetrahedronSetTopology
template<class In, class Out>
class BarycentricMapperTetrahedronSetTopology : public BarycentricMapperTopologyContainer<In,Out,typename BarycentricMapper<In,Out>::MappingData3D,Tetrahedron>
{
    typedef typename BarycentricMapper<In,Out>::MappingData3D MappingData;

public:
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapperTetrahedronSetTopology,In,Out),SOFA_TEMPLATE4(BarycentricMapperTopologyContainer,In,Out,MappingData,Tetrahedron));
    typedef typename Inherit1::Real Real;
    typedef typename In::VecCoord VecCoord;

    virtual int addPointInTetra(const int index, const SReal* baryCoords) override ;

protected:
    BarycentricMapperTetrahedronSetTopology(topology::TetrahedronSetTopologyContainer* fromTopology,
                                            topology::PointSetTopologyContainer* toTopology);
    virtual ~BarycentricMapperTetrahedronSetTopology() override {}

    virtual helper::vector<Tetrahedron> getElements() override;
    virtual helper::vector<SReal> getBaryCoef(const Real* f) override;
    helper::vector<SReal> getBaryCoef(const Real fx, const Real fy, const Real fz);
    virtual void computeBase(Mat3x3d& base, const typename In::VecCoord& in, const Tetrahedron& element) override;
    virtual void computeCenter(Vector3& center, const typename In::VecCoord& in, const Tetrahedron& element) override;
    virtual void computeDistance(double& d, const Vector3& v) override;
    virtual void addPointInElement(const int elementIndex, const SReal* baryCoords) override;

    topology::TetrahedronSetTopologyContainer*      m_fromContainer {nullptr};
    topology::TetrahedronSetGeometryAlgorithms<In>*	m_fromGeomAlgo  {nullptr};

    using Inherit1::d_map;
    using Inherit1::m_fromTopology;
    using Inherit1::m_matrixJ;
    using Inherit1::m_updateJ;
};

#if !defined(SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTETRAHEDRONSETTOPOLOGY_CPP)
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< Vec3dTypes, ExtVec3Types >;


#endif

}}}

#endif
