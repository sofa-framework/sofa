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
#include <sofa/component/collision/detection/intersection/config.h>

#include <sofa/component/collision/detection/intersection/BaseProximityIntersection.h>

namespace sofa::component::collision::detection::intersection
{

/**
 * Basic intersection methods using proximities
 * It uses proximities on cubes and spheres, but supported pairs of collision models can
 * be extended. For example, see MeshMinProximityIntersection which adds support for
 * additional types of intersection.
 *
 * Supported by default:
 * - Cube/Cube
 * - Sphere/Sphere (rigid or vec3)
 * MeshMinProximityIntersection adds support for:
 * - Point/Point (if d_usePointPoint is true)
 * - Line/Point (if d_useLinePoint is true)
 * - Line/Line (if d_useLineLine is true)
 * - Triangle/Point
 * - Sphere/Point (if d_useSphereTriangle is true)
 * - RigidSphere/Point (if d_useSphereTriangle is true)
 * - Triangle/Sphere (if d_useSphereTriangle is true)
 * - Triangle/RigidSphere (if d_useSphereTriangle is true)
 * - Line/Sphere (if d_useSphereTriangle is true)
 * - Line/RigidSphere (if d_useSphereTriangle is true)
 * Note that MeshMinProximityIntersection ignores Triangle/Line and Triangle/Triangle intersections.
 * Datas can be set to ignore some pairs of collision models (d_useSphereTriangle, d_usePointPoint, etc).
 */
class SOFA_COMPONENT_COLLISION_DETECTION_INTERSECTION_API MinProximityIntersection : public BaseProximityIntersection
{
public:
    SOFA_CLASS(MinProximityIntersection,BaseProximityIntersection);

    Data<bool> d_useSphereTriangle; ///< activate Sphere-Triangle intersection tests
    Data<bool> d_usePointPoint; ///< activate Point-Point intersection tests
    Data<bool> d_useSurfaceNormals; ///< Compute the norms of the Detection Outputs by considering the normals of the surfaces involved.
    Data<bool> d_useLinePoint; ///< activate Line-Point intersection tests
    Data<bool> d_useLineLine; ///< activate Line-Line  intersection tests

protected:
    MinProximityIntersection();
public:
    typedef core::collision::IntersectorFactory<MinProximityIntersection> IntersectorFactory;

    void init() override;

    bool getUseSurfaceNormals() const;
        
    bool testIntersection(collision::geometry::Cube& cube1, collision::geometry::Cube& cube2, const core::collision::Intersection* currentIntersection) override;
    int computeIntersection(collision::geometry::Cube& cube1, collision::geometry::Cube& cube2, OutputVector* contacts, const core::collision::Intersection* currentIntersection) override;

    template<typename SphereType1, typename SphereType2>
    bool testIntersection(SphereType1& sph1, SphereType2& sph2, const core::collision::Intersection* currentIntersection)
    {
        const auto alarmDist = currentIntersection->getAlarmDistance() + sph1.getContactDistance() + sph2.getContactDistance();
        return DiscreteIntersection::testIntersectionSphere(sph1, sph2, alarmDist);
    }
    template<typename SphereType1, typename SphereType2>
    int computeIntersection(SphereType1& sph1, SphereType2& sph2, OutputVector* contacts, const core::collision::Intersection* currentIntersection)
    {
        const auto alarmDist = currentIntersection->getAlarmDistance() + sph1.getContactDistance() + sph2.getContactDistance();
        const auto contactDist = currentIntersection->getContactDistance() + sph1.getContactDistance() + sph2.getContactDistance();
        return DiscreteIntersection::computeIntersectionSphere(sph1, sph2, contacts, alarmDist, contactDist);
    }
};

} // namespace sofa::component::collision::detection::intersection

namespace sofa::core::collision
{
#if !defined(SOFA_COMPONENT_COLLISION_MINPROXIMITYINTERSECTION_CPP)
extern template class SOFA_COMPONENT_COLLISION_DETECTION_INTERSECTION_API IntersectorFactory<component::collision::detection::intersection::MinProximityIntersection>;
#endif
} // namespace sofa::core::collision
