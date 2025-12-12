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
#define SOFA_COMPONENT_COLLISION_CCDTightInclusionIntersection_CPP
#include <sofa/component/collision/detection/intersection/CCDTightInclusionIntersection.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/geometry/proximity/SegmentTriangle.h>
#include <sofa/simulation/Node.h>

#include <sofa/core/collision/Intersection.inl>

#include <sofa/component/collision/geometry/LineModel.h>
#include <sofa/component/collision/geometry/TriangleModel.h>
#include <sofa/linearalgebra/EigenVector.h>

#include <tight_inclusion/ccd.hpp>

#define EMIT_EXTRA_DEBUG_MESSAGE false

namespace sofa::core::collision
{
    template class SOFA_COMPONENT_COLLISION_DETECTION_INTERSECTION_API IntersectorFactory<component::collision::detection::intersection::CCDTightInclusionIntersection>;

} // namespace sofa::core::collision

namespace sofa::component::collision::detection::intersection
{

using namespace sofa::core::collision;
using namespace helper;
using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::component::collision::geometry;
using core::topology::BaseMeshTopology;

void registerCCDTightInclusionIntersection(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("A set of methods to compute (for constraint methods) if two primitives are close enough to consider they collide")
        .add< CCDTightInclusionIntersection >());
}

CCDTightInclusionIntersection::CCDTightInclusionIntersection()
: BaseProximityIntersection()
, d_continuousCollisionType(initData(&d_continuousCollisionType, helper::OptionsGroup({"None", "Inertia", "FreeMotion"}).setSelectedItem(0), "continuousCollisionType",
    "Data used for continuous collision detection taken into {'None','Inertia','FreeMotion'}. If 'None' then no CCD is used, if 'Inertia' then only inertia will be used to compute the collision detection and if 'FreeMotion' then the free motion will be used. Note that if 'FreeMotion' is selected, you cannot use the option 'parallelCollisionDetectionAndFreeMotion' in the FreeMotionAnimationLoop"))
, d_tolerance(initData(&d_tolerance,1e-10,"tolerance","tolerance used by the tight inclusion CCD algorithm"))
, d_maxIterations(initData(&d_maxIterations,(long) 1000,"maxIterations","maxIterations used by the tight inclusion CCD algorithm"))
{

}

void CCDTightInclusionIntersection::init()
{
    intersectors.add<CubeCollisionModel, CubeCollisionModel, CCDTightInclusionIntersection>(this);
    intersectors.add<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>, CCDTightInclusionIntersection>(this);
    intersectors.add<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>, CCDTightInclusionIntersection>(this);

    intersectors.ignore<SphereCollisionModel<sofa::defaulttype::Vec3Types>,     SphereCollisionModel<sofa::defaulttype::Vec3Types>>();
    intersectors.ignore<SphereCollisionModel<sofa::defaulttype::Vec3Types>,     PointCollisionModel<sofa::defaulttype::Vec3Types>>();
    intersectors.ignore<PointCollisionModel<sofa::defaulttype::Vec3Types>,      PointCollisionModel<sofa::defaulttype::Vec3Types>>();
    intersectors.ignore<LineCollisionModel<sofa::defaulttype::Vec3Types>,       PointCollisionModel<sofa::defaulttype::Vec3Types>>();
    intersectors.ignore<LineCollisionModel<sofa::defaulttype::Vec3Types>,       SphereCollisionModel<sofa::defaulttype::Vec3Types>>();
    intersectors.ignore<TriangleCollisionModel<sofa::defaulttype::Vec3Types>,   LineCollisionModel<sofa::defaulttype::Vec3Types>>();
    intersectors.ignore<TriangleCollisionModel<sofa::defaulttype::Vec3Types>,   TriangleCollisionModel<sofa::defaulttype::Vec3Types>>();
    intersectors.ignore<TriangleCollisionModel<sofa::defaulttype::Vec3Types>,   SphereCollisionModel<sofa::defaulttype::Vec3Types>>();
    intersectors.ignore<RayCollisionModel,                                      TriangleCollisionModel<sofa::defaulttype::Vec3Types>>();
    intersectors.ignore<RayCollisionModel,                                      SphereCollisionModel<sofa::defaulttype::Vec3Types>>();
    intersectors.ignore<RayCollisionModel,                                      PointCollisionModel<sofa::defaulttype::Vec3Types>>();
    intersectors.ignore<RayCollisionModel,                                      LineCollisionModel<sofa::defaulttype::Vec3Types>>();



    //By default, all th previous pairs of collision models are supported,
    //but other C++ components are able to add a list of pairs to be supported.
    //In the following function, all the C++ components that registered to
    //CCDTightInclusionIntersection are created. In their constructors, they add
    //new supported pairs of collision models.
    IntersectorFactory::getInstance()->addIntersectors(this);

    BaseProximityIntersection::init();
}

bool CCDTightInclusionIntersection::useContinuous() const
{
    return d_continuousCollisionType.getValue().getSelectedId();
}

core::CollisionModel::ContinuousIntersectionTypeFlag CCDTightInclusionIntersection::continuousIntersectionType() const
{
    if (d_continuousCollisionType.getValue().getSelectedId()<= 3 )
        return static_cast<core::CollisionModel::ContinuousIntersectionTypeFlag>(d_continuousCollisionType.getValue().getSelectedId());
    else
        return core::CollisionModel::ContinuousIntersectionTypeFlag::None;
}



bool CCDTightInclusionIntersection::testIntersection(Cube &cube1, Cube &cube2, const core::collision::Intersection* currentIntersection)
{
    return Inherit1::testIntersection(cube1, cube2, currentIntersection);
}

int CCDTightInclusionIntersection::computeIntersection(Cube&, Cube&, OutputVector* /*contacts*/, const core::collision::Intersection* )
{
    return 0; /// \todo
}

bool CCDTightInclusionIntersection::testIntersection(Line& e1, Line& e2, const core::collision::Intersection* currentIntersection)
{
    if(!e1.isActive(e2.getCollisionModel()) || !e2.isActive(e1.getCollisionModel()))
    {
        return false;
    }


    const Eigen::Map<Eigen::Vector3<SReal>> Line1ABegin(const_cast<double*>(e1.p1().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> Line1AEnd(const_cast<double*>(e1.p1Free().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> Line1BBegin(const_cast<double*>(e1.p2().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> Line1BEnd(const_cast<double*>(e1.p2Free().elems.data()));

    const Eigen::Map<Eigen::Vector3<SReal>> Line2ABegin(const_cast<double*>(e2.p1().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> Line2AEnd(const_cast<double*>(e2.p1Free().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> Line2BBegin(const_cast<double*>(e2.p2().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> Line2BEnd(const_cast<double*>(e2.p2Free().elems.data()));


    const Eigen::Vector3d err {-1.0, -1.0, -1.0};
    const double maxSeparation = currentIntersection->getContactDistance() + e1.getContactDistance() + e2.getContactDistance(); // 0.00001
    SReal toi = 2.0;
    const SReal tmax = 1.0;
    SReal outputTolerance = 0.0;


    return ticcd::edgeEdgeCCD(
        Line1ABegin, Line2ABegin, Line1BBegin, Line2BBegin,
        Line1AEnd,   Line2AEnd,   Line1BEnd,   Line2BEnd,
        err,maxSeparation, toi,d_tolerance.getValue(), tmax, d_maxIterations.getValue(), outputTolerance);

}

int CCDTightInclusionIntersection::computeIntersection(Line& e1, Line& e2, OutputVector* contacts, const core::collision::Intersection* currentIntersection)
{
    if(!e1.isActive(e2.getCollisionModel()) || !e2.isActive(e1.getCollisionModel()))
    {
        dmsg_info_when(EMIT_EXTRA_DEBUG_MESSAGE)
            <<" not activated" ;
        return 0;
    }

    const Eigen::Map<Eigen::Vector3<SReal>> Line1ABegin(const_cast<double*>(e1.p1().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> Line1AEnd(const_cast<double*>(e1.p1Free().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> Line1BBegin(const_cast<double*>(e1.p2().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> Line1BEnd(const_cast<double*>(e1.p2Free().elems.data()));

    const Eigen::Map<Eigen::Vector3<SReal>> Line2ABegin(const_cast<double*>(e2.p1().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> Line2AEnd(const_cast<double*>(e2.p1Free().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> Line2BBegin(const_cast<double*>(e2.p2().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> Line2BEnd(const_cast<double*>(e2.p2Free().elems.data()));


    const Eigen::Vector3d err {-1.0, -1.0, -1.0};
    const auto maxSeparation = currentIntersection->getContactDistance() + e1.getContactDistance() + e2.getContactDistance(); // 0.00001
    SReal toi = 2.0;
    const SReal tmax = 1.0;
    SReal outputTolerance = 0.0;


    const auto result =  ticcd::edgeEdgeCCD(
        Line1ABegin, Line2ABegin, Line1BBegin, Line2BBegin,
        Line1AEnd,   Line2AEnd,   Line1BEnd,   Line2BEnd,
        err,maxSeparation, toi,d_tolerance.getValue(), tmax, d_maxIterations.getValue(), outputTolerance);


    const Vec3 Line1AToi = (1-toi) * e1.p1() + e1.p1Free() * toi;
    const Vec3 Line1BToi = (1-toi) * e1.p2() + e1.p2Free() * toi;
    const Vec3 Line2AToi = (1-toi) * e2.p1() + e2.p1Free() * toi;
    const Vec3 Line2BToi = (1-toi) * e2.p2() + e2.p2Free() * toi;
    type::Vec2 baryCoords(type::NOINIT);
    sofa::geometry::Edge::closestPointWithEdge(Line1AToi, Line1BToi, Line2AToi, Line2BToi, baryCoords);

    contacts->resize(contacts->size() +1 );
    sofa::core::collision::DetectionOutput detection ;
    detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection.id = (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex();
    detection.point[0] = (1-baryCoords[0]) * e1.p1() + baryCoords[0] * e1.p2();
    detection.point[1] = (1-baryCoords[1]) * e2.p1() + baryCoords[1] * e2.p2();
    detection.normal = (1-baryCoords[1]) * Line2AToi + baryCoords[1] * Line2BToi - ((1-baryCoords[0]) * Line1AToi + baryCoords[0] * Line1BToi);
    detection.value = detection.normal.norm();
    detection.normal /= detection.value;
    detection.value -= maxSeparation;
    contacts->push_back(detection);

     return 1;
}

bool CCDTightInclusionIntersection::testIntersection(Triangle& triangle, Point& point, const core::collision::Intersection* currentIntersection)
{
    if(!triangle.isActive(point.getCollisionModel()) || !point.isActive(triangle.getCollisionModel()))
    {
        return false;
    }


    const Eigen::Map<Eigen::Vector3<SReal>> TriangleABegin(const_cast<double*>(triangle.p1().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> TriangleAEnd(const_cast<double*>(triangle.p1Free().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> TriangleBBegin(const_cast<double*>(triangle.p2().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> TriangleBEnd(const_cast<double*>(triangle.p2Free().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> TriangleCBegin(const_cast<double*>(triangle.p3().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> TriangleCEnd(const_cast<double*>(triangle.p3Free().elems.data()));

    const Eigen::Map<Eigen::Vector3<SReal>> PointBegin(const_cast<double*>(point.p().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> PointEnd(const_cast<double*>(point.pFree().elems.data()));


    const Eigen::Vector3d err {-1.0, -1.0, -1.0};
    const double maxSeparation = currentIntersection->getContactDistance() + triangle.getContactDistance() + point.getContactDistance(); // 0.00001
    SReal toi = 2.0;
    const SReal tmax = 1.0;
    SReal outputTolerance = 0.0;


    return ticcd::vertexFaceCCD(
        PointBegin, TriangleABegin, TriangleBBegin, TriangleCBegin,
        PointEnd,   TriangleAEnd,   TriangleBEnd,   TriangleCEnd,
        err,maxSeparation, toi,d_tolerance.getValue(), tmax, d_maxIterations.getValue(), outputTolerance);

}

int CCDTightInclusionIntersection::computeIntersection(Triangle& triangle, Point& point, OutputVector* contacts, const core::collision::Intersection* currentIntersection)
{

    if(!triangle.isActive(point.getCollisionModel()) || !point.isActive(triangle.getCollisionModel()))
    {
        dmsg_info_when(EMIT_EXTRA_DEBUG_MESSAGE)
            <<" not activated" ;
        return 0;
    }
    const Eigen::Map<Eigen::Vector3<SReal>> TriangleABegin(const_cast<double*>(triangle.p1().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> TriangleAEnd(const_cast<double*>(triangle.p1Free().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> TriangleBBegin(const_cast<double*>(triangle.p2().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> TriangleBEnd(const_cast<double*>(triangle.p2Free().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> TriangleCBegin(const_cast<double*>(triangle.p3().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> TriangleCEnd(const_cast<double*>(triangle.p3Free().elems.data()));

    const Eigen::Map<Eigen::Vector3<SReal>> PointBegin(const_cast<double*>(point.p().elems.data()));
    const Eigen::Map<Eigen::Vector3<SReal>> PointEnd(const_cast<double*>(point.pFree().elems.data()));

    const Eigen::Vector3d err {-1.0, -1.0, -1.0};
    const double maxSeparation = currentIntersection->getContactDistance() + triangle.getContactDistance() + point.getContactDistance(); // 0.00001
    SReal toi = std::numeric_limits<SReal>::infinity();
    const SReal tmax = 1.0;
    SReal outputTolerance = 0.0;

    const bool result = ticcd::vertexFaceCCD(
        PointBegin, TriangleABegin, TriangleBBegin, TriangleCBegin,
        PointEnd,   TriangleAEnd,   TriangleBEnd,   TriangleCEnd,
        err,maxSeparation, toi,d_tolerance.getValue(), tmax, d_maxIterations.getValue(), outputTolerance);

    if (!result)
        return 0;

    const Vec3 TriangleAToi = (1-toi) * triangle.p1() + triangle.p1Free() * toi;
    const Vec3 TriangleBToi = (1-toi) * triangle.p2() + triangle.p2Free() * toi;
    const Vec3 TriangleCToi = (1-toi) * triangle.p3() + triangle.p3Free() * toi;
    const Vec3 PointToi = (1-toi) * point.p() + point.pFree() * toi;

    const Vec3 TriangleBaryToi = (TriangleAToi + TriangleBToi + TriangleCToi) /3.0;
    const Vec3 TriangleNormalBaryToi = (TriangleBToi - TriangleAToi).cross(TriangleCToi - TriangleAToi).normalized();
    const Vec3 ProjOnTriangleToi = TriangleBaryToi + (PointToi - TriangleBaryToi) - TriangleNormalBaryToi * dot(PointToi - TriangleBaryToi,TriangleNormalBaryToi) ;

    type::Vec3 baryCoords = sofa::geometry::Triangle::getBarycentricCoordinates(ProjOnTriangleToi, TriangleAToi, TriangleBToi, TriangleCToi);

    contacts->resize(contacts->size() +1 );
    auto * detection = &contacts->back();
    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(triangle, point);
    detection->id = (triangle.getCollisionModel()->getSize() > point.getCollisionModel()->getSize()) ? triangle.getIndex() : point.getIndex();
    detection->point[0] = triangle.p1() * baryCoords[0] + triangle.p2() * baryCoords[1] + triangle.p3() * baryCoords[2] ;
    detection->point[1] = point.p();
    detection->normal = TriangleNormalBaryToi;
    detection->value = dot(detection->point[1] - detection->point[0] , detection->normal) ;
    detection->value -= maxSeparation;

     return 1;
}

} //namespace sofa::component::collision::detection::intersection
