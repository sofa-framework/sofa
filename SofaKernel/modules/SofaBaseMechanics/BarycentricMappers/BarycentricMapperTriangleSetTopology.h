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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTRIANGLESETTOPOLOGY_H
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTRIANGLESETTOPOLOGY_H
#include <SofaBaseMechanics/BarycentricMappers/BarycentricMapperTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>

namespace sofa
{

namespace component
{

namespace mapping
{


using sofa::defaulttype::Mat3x3d;
using sofa::defaulttype::Vector3;
typedef typename sofa::core::topology::BaseMeshTopology::Triangle Triangle;


/// Class allowing barycentric mapping computation on a TriangleSetTopology
template<class In, class Out>
class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology : public BarycentricMapperTopologyContainer<In,Out,typename BarycentricMapper<In,Out>::MappingData2D,Triangle>
{
    typedef typename BarycentricMapper<In,Out>::MappingData2D MappingData;
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapperTriangleSetTopology,In,Out),
               SOFA_TEMPLATE4(BarycentricMapperTopologyContainer,In,Out,MappingData,Triangle));
    typedef typename Inherit1::Real Real;

    virtual ~BarycentricMapperTriangleSetTopology() {}

    virtual int addPointInTriangle(const int triangleIndex, const SReal* baryCoords) override;
    int createPointInTriangle(const typename Out::Coord& p, int triangleIndex, const typename In::VecCoord* points) override;

protected:
    BarycentricMapperTriangleSetTopology(topology::TriangleSetTopologyContainer* fromTopology,
                                         topology::PointSetTopologyContainer* toTopology);

    topology::TriangleSetTopologyContainer*			m_fromContainer;
    topology::TriangleSetGeometryAlgorithms<In>*	m_fromGeomAlgo;

    virtual helper::vector<Triangle> getElements() override;
    virtual helper::vector<SReal> getBaryCoef(const Real* f) override;
    helper::vector<SReal> getBaryCoef(const Real fx, const Real fy);
    virtual void computeBase(Mat3x3d& base, const typename In::VecCoord& in, const Triangle& element) override;
    virtual void computeCenter(Vector3& center, const typename In::VecCoord& in, const Triangle& element) override;
    virtual void computeDistance(double& d, const Vector3& v) override;
    virtual void addPointInElement(const int elementIndex, const SReal* baryCoords) override;

    using Inherit1::d_map;
    using Inherit1::m_fromTopology;
    using Inherit1::m_matrixJ;
    using Inherit1::m_updateJ;
};


using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::ExtVec3fTypes;

#if !defined(SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTRIANGLESETTOPOLOGY_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< Vec3dTypes, ExtVec3fTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< Vec3fTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< Vec3fTypes, ExtVec3fTypes >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< Vec3dTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< Vec3fTypes, Vec3dTypes >;
#endif
#endif
#endif

}}}


#endif
