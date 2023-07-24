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
#define SOFA_COMPONENT_COLLISION_LOCALMINDISTANCE_CPP
#include <sofa/component/collision/detection/intersection/LocalMinDistance.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/Intersection.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/geometry/proximity/SegmentTriangle.h>
#include <sofa/simulation/Node.h>

#define EMIT_EXTRA_DEBUG_MESSAGE false

namespace sofa::core::collision
{
    template class SOFA_COMPONENT_COLLISION_DETECTION_INTERSECTION_API IntersectorFactory<component::collision::detection::intersection::LocalMinDistance>;

} // namespace sofa::core::collision

namespace sofa::component::collision::detection::intersection
{

using namespace sofa::core::collision;
using namespace helper;
using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::component::collision::geometry;
using core::topology::BaseMeshTopology;


int LocalMinDistanceClass = core::RegisterObject("A set of methods to compute (for constraint methods) if two primitives are close enough to consider they collide")
        .add< LocalMinDistance >()
        ;

LocalMinDistance::LocalMinDistance()
    : BaseProximityIntersection()
    , filterIntersection(initData(&filterIntersection, true, "filterIntersection","Activate LMD filter"))
    , angleCone(initData(&angleCone, 0.0, "angleCone","Filtering cone extension angle"))
    , coneFactor(initData(&coneFactor, 0.5, "coneFactor", "Factor for filtering cone angle computation"))
    , useLMDFilters(initData(&useLMDFilters, false, "useLMDFilters", "Use external cone computation (Work in Progress)"))
{
}

void LocalMinDistance::init()
{
    intersectors.add<CubeCollisionModel, CubeCollisionModel, LocalMinDistance>(this);
    intersectors.add<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>, LocalMinDistance>(this); // sphere-sphere is always activated
    intersectors.add<SphereCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>, LocalMinDistance>(this); // sphere-point is always activated

    intersectors.add<PointCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>, LocalMinDistance>(this); // point-point is always activated
    intersectors.add<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>, LocalMinDistance>(this);
    intersectors.add<LineCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>, LocalMinDistance>(this);
    intersectors.add<LineCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>, LocalMinDistance>(this);
    intersectors.add<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>, LocalMinDistance>(this);
    intersectors.add<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>, LocalMinDistance>(this);

    intersectors.ignore<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>>();			// never the case with LMD
    intersectors.ignore<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>>();		// never the case with LMD

    intersectors.ignore<RayCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>>();
    intersectors.ignore<RayCollisionModel, LineCollisionModel<sofa::defaulttype::Vec3Types>>();
    intersectors.add<RayCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LocalMinDistance>(this);
    intersectors.add<RayCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>, LocalMinDistance>(this);

    //By default, all the previous pairs of collision models are supported,
    //but other C++ components are able to add a list of pairs to be supported.
    //In the following function, all the C++ components that registered to
    //LocalMinDistance are created. In their constructors, they add
    //new supported pairs of collision models.
    IntersectorFactory::getInstance()->addIntersectors(this);

    BaseProximityIntersection::init();
}

bool LocalMinDistance::testIntersection(Cube &cube1, Cube &cube2)
{
    return Inherit1::testIntersection(cube1, cube2);
}

int LocalMinDistance::computeIntersection(Cube&, Cube&, OutputVector* /*contacts*/)
{
    return 0; /// \todo
}

bool LocalMinDistance::testIntersection(Line& e1, Line& e2)
{
    if(!e1.isActive(e2.getCollisionModel()) || !e2.isActive(e1.getCollisionModel()))
    {
        return false;
    }

    const SReal alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

    const Line::Coord AB = e1.p2()-e1.p1();
    const Line::Coord CD = e2.p2()-e2.p1();
    const Line::Coord AC = e2.p1()-e1.p1();

    MatNoInit<2, 2, Line::Coord::value_type> A;
    VecNoInit<2, Line::Coord::value_type> b;

    A[0][0] = AB*AB;
    A[1][1] = CD*CD;
    A[0][1] = A[1][0] = -CD*AB;
    b[0] = AB*AC;
    b[1] = -CD*AC;

    const Line::Coord::value_type det = type::determinant(A);

    Line::Coord::value_type alpha = 0.5;
    Line::Coord::value_type beta = 0.5;

    if (det < -1.0e-30 || det > 1.0e-30)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        if (alpha < 1e-15 || alpha > (1.0-1e-15) ||
            beta  < 1e-15  || beta  > (1.0-1e-15) )
            return false;
    }

    const Line::Coord PQ = AC+CD*beta-AB*alpha;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        // filter for LMD

        if (!useLMDFilters.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            const Line::Coord QP = -PQ;
            return testValidity(e2, QP);
        }

        return true;
    }

    return false;
}

int LocalMinDistance::computeIntersection(Line& e1, Line& e2, OutputVector* contacts)
{

    if(!e1.isActive(e2.getCollisionModel()) || !e2.isActive(e1.getCollisionModel()))
    {
        dmsg_info_when(EMIT_EXTRA_DEBUG_MESSAGE)
            <<" not activated" ;
        return 0;
    }

    const SReal alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

    // E1 => A-->B
    // E2 => C-->D
    const Line::Coord AB = e1.p2()-e1.p1();
    const Line::Coord CD = e2.p2()-e2.p1();
    const Line::Coord AC = e2.p1()-e1.p1();
    Matrix2 A;
    Vec2 b;

    A[0][0] = AB*AB;
    A[1][1] = CD*CD;
    A[0][1] = A[1][0] = -CD*AB;
    b[0] = AB*AC;
    b[1] = -CD*AC;
    const double det = type::determinant(A);

    double alpha;
    double beta;

    // If lines are not parallel
    if (det < -1.0e-30 || det > 1.0e-30)
    {
        // compute the parametric coordinates along the line
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;

        // if parameters are outside of ]0,1[ then there is no intersection
        // the intersection is outside of the edge supporting point.
        if (alpha < 1e-15 || alpha > (1.0-1e-15) ||
            beta  < 1e-15  || beta  > (1.0-1e-15) )
            return 0;
    }
    else
    {
        // lines are parallel,
        // the alpha/beta parameters are set to 0.5 so the collision
        // output is in the middle of the line
        alpha = 0.5;
        beta = 0.5;
    }


    Vec3 P,Q,PQ;
    P = e1.p1() + AB * alpha;
    Q = e2.p1() + CD * beta;
    PQ = Q-P;

    // If the geometric distance between P and Q is higher than the alarm distance/
    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    // filter for LMD //

    if (!useLMDFilters.getValue())
    {
        if (!testValidity(e1, PQ))
        {
            dmsg_info_when(EMIT_EXTRA_DEBUG_MESSAGE)
                 <<" testValidity rejected for the first segment" ;
            return 0;
        }

        const Vec3 QP = -PQ;

        if (!testValidity(e2, QP))
        {
            dmsg_info_when(EMIT_EXTRA_DEBUG_MESSAGE)
                <<" testValidity rejected for the second segment";
            return 0;
        }
    }

    contacts->resize(contacts->size() + 1);
    DetectionOutput *detection = &*(contacts->end() - 1);

#ifdef SOFA_DETECTIONOUTPUT_FREEMOTION

    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        type::Vec3 Pfree, Qfree, ABfree, CDfree;
        ABfree = e1.p2Free()-e1.p1Free();
        CDfree = e2.p2Free()-e2.p1Free();
        Pfree = e1.p1Free() + ABfree * alpha;
        Qfree = e2.p1Free() + CDfree * beta;
        detection->freePoint[0] = Pfree;
        detection->freePoint[1] = Qfree;
    }

#endif

    const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->id = (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex();
    detection->point[0] = P;
    detection->point[1] = Q;
    detection->normal = PQ;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;

    return 1;
}

bool LocalMinDistance::testIntersection(Triangle& e2, Point& e1)
{
    if(!e1.isActive(e2.getCollisionModel()))
        return false;

    const SReal alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

    const Vec3 AB = e2.p2()-e2.p1();
    const Vec3 AC = e2.p3()-e2.p1();
    const Vec3 AP = e1.p() -e2.p1();
    Matrix2 A;
    Vec2 b;

    // We want to find alpha,beta so that:
    // AQ = AB*alpha+AC*beta
    // PQ.AB = 0 and PQ.AC = 0
    // (AQ-AP).AB = 0 and (AQ-AP).AC = 0
    // AQ.AB = AP.AB and AQ.AC = AP.AC
    //
    // (AB*alpha+AC*beta).AB = AP.AB and
    // (AB*alpha+AC*beta).AC = AP.AC
    //
    // AB.AB*alpha + AC.AB*beta = AP.AB and
    // AB.AC*alpha + AC.AC*beta = AP.AC
    //
    // A . [alpha beta] = b
    A[0][0] = AB*AB;
    A[1][1] = AC*AC;
    A[0][1] = A[1][0] = AB*AC;
    b[0] = AP*AB;
    b[1] = AP*AC;
    const double det = type::determinant(A);

    double alpha = 0.5;
    double beta = 0.5;


    alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
    beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
    if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
        return false;

    const Vec3 PQ = AB * alpha + AC * beta - AP;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        //filter for LMD
        if (!useLMDFilters.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            const Vec3 QP = -PQ;
            return testValidity(e2, QP);
        }
        else
        {
            return true;
        }
        // end filter
    }
    else
        return false;
}

int LocalMinDistance::computeIntersection(Triangle& e2, Point& e1, OutputVector* contacts)
{
    if(!e1.isActive(e2.getCollisionModel()))
        return 0;

    const SReal alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

    static_assert(std::is_same_v<Triangle::Coord, Point::Coord>, "Data mismatch");
    using Real = Triangle::Coord::value_type;

    const Triangle::Coord AB = e2.p2()-e2.p1();
    const Triangle::Coord AC = e2.p3()-e2.p1();
    const auto AP = e1.p() -e2.p1();
    MatNoInit<2, 2, Real> A;
    VecNoInit<2, Real> b;

    A[0][0] = AB*AB;
    A[1][1] = AC*AC;
    A[0][1] = A[1][0] = AB*AC;
    b[0] = AP*AB;
    b[1] = AP*AC;

    const Real det = type::determinant(A);

    const Real alpha=(b[0]*A[1][1]-b[1]*A[0][1])/det;
    const Real beta=(b[1]*A[0][0]-b[0]*A[1][0])/det;

    if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
        return 0;

    const Point::Coord P = e1.p();
    const Triangle::Coord Q = e2.p1()+AB*alpha+AC*beta;
    const auto PQ = Q-P;

    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    const auto QP = -PQ;

    // filter for LMD

    if (!useLMDFilters.getValue())
    {
        if (!testValidity(e1, PQ))
            return 0;

        if (!testValidity(e2, QP))
            return 0;
    }

    //end filter

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

#ifdef SOFA_DETECTIONOUTPUT_FREEMOTION
    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        type::Vec3 Pfree,Qfree,ABfree,ACfree;
        ABfree = e2.p2Free()-e2.p1Free();
        ACfree = e2.p3Free()-e2.p1Free();
        Pfree = e1.pFree();
        Qfree = e2.p1Free() + ABfree * alpha + ACfree * beta;

        detection->freePoint[0] = Qfree;
        detection->freePoint[1] = Pfree;
    }
#endif

    const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
    detection->id = e1.getIndex();
    detection->point[0] = Q;
    detection->point[1] = P;
    detection->normal = QP;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;
    return 1;
}


bool LocalMinDistance::testIntersection(Triangle& e2, Sphere& e1)
{
    if (!e1.isActive(e2.getCollisionModel()))
        return false;

    const SReal alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    static_assert(std::is_same_v<Triangle::Coord, Sphere::Coord>, "Data mismatch");
    using Real = Triangle::Coord::value_type;

    const Triangle::Coord AB = e2.p2()-e2.p1();
    const Triangle::Coord AC = e2.p3()-e2.p1();
    const auto AP = e1.p() -e2.p1();
    MatNoInit<2, 2, Real> A;
    VecNoInit<2, Real> b;

    // We want to find alpha,beta so that:
    // AQ = AB*alpha+AC*beta
    // PQ.AB = 0 and PQ.AC = 0
    // (AQ-AP).AB = 0 and (AQ-AP).AC = 0
    // AQ.AB = AP.AB and AQ.AC = AP.AC
    //
    // (AB*alpha+AC*beta).AB = AP.AB and
    // (AB*alpha+AC*beta).AC = AP.AC
    //
    // AB.AB*alpha + AC.AB*beta = AP.AB and
    // AB.AC*alpha + AC.AC*beta = AP.AC
    //
    // A . [alpha beta] = b
    A[0][0] = AB*AB;
    A[1][1] = AC*AC;
    A[0][1] = A[1][0] = AB*AC;
    b[0] = AP*AB;
    b[1] = AP*AC;
    const Real det = type::determinant(A);

    const Real alpha=(b[0]*A[1][1]-b[1]*A[0][1])/det;
    const Real beta=(b[1]*A[0][0]-b[0]*A[1][0])/det;
    if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
        return false;


    const auto PQ = AB * alpha + AC * beta - AP;

    if (PQ.norm2() < alarmDist*alarmDist)
    {

        //filter for LMD

        if (!useLMDFilters.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            const Vec3 QP = -PQ;
            return testValidity(e2, QP);
        }
        else
        {
            return true;
        }

        // end filter
    }
    else
        return false;
}

int LocalMinDistance::computeIntersection(Triangle& e2, Sphere& e1, OutputVector* contacts)
{
    if (!e1.isActive(e2.getCollisionModel()))
        return false;

    const SReal alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    static_assert(std::is_same_v<Triangle::Coord, Sphere::Coord>, "Data mismatch");
    using Real = Triangle::Coord::value_type;

    const Triangle::Coord AB = e2.p2()-e2.p1();
    const Triangle::Coord AC = e2.p3()-e2.p1();
    const auto AP = e1.p() -e2.p1();
    MatNoInit<2, 2, Real> A;
    VecNoInit<2, Real> b;

    A[0][0] = AB*AB;
    A[1][1] = AC*AC;
    A[0][1] = A[1][0] = AB*AC;
    b[0] = AP*AB;
    b[1] = AP*AC;

    const Real det = type::determinant(A);

    const Real alpha=(b[0]*A[1][1]-b[1]*A[0][1])/det;
    const Real beta=(b[1]*A[0][0]-b[0]*A[1][0])/det;
    if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
        return 0;

    const Sphere::Coord& P = e1.p();
    const Triangle::Coord Q = e2.p1()+AB*alpha+AC*beta;
    const auto PQ = Q-P;

    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    const auto QP = -PQ;

    // filter for LMD

    if (!useLMDFilters.getValue())
    {
        if (!testValidity(e1, PQ))
            return 0;

        if (!testValidity(e2, QP))
            return 0;
    }

    //end filter

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

#ifdef SOFA_DETECTIONOUTPUT_FREEMOTION
    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        type::Vec3 Pfree,Qfree,ABfree,ACfree;
        ABfree = e2.p2Free()-e2.p1Free();
        ACfree = e2.p3Free()-e2.p1Free();
        Pfree = e1.pFree();
        Qfree = e2.p1Free() + ABfree * alpha + ACfree * beta;

        detection->freePoint[0] = Qfree;
        detection->freePoint[1] = Pfree;
    }
#endif

    const double contactDist = getContactDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
    detection->id = e1.getIndex();
    detection->point[0] = Q;
    detection->point[1] = P;
    detection->normal = QP;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;
    return 1;
}

bool LocalMinDistance::testIntersection(Line& e2, Point& e1)
{
    static_assert(std::is_same_v<Line::Coord, Point::Coord>, "Data mismatch");
    using Real = Triangle::Coord::value_type;

    if(!e1.isActive(e2.getCollisionModel()) || !e2.isActive(e1.getCollisionModel()))
        return false;

    const SReal alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const Line::Coord AB = e2.p2()-e2.p1();
    const Line::Coord AP = e1.p()-e2.p1();

    const Real A = AB*AB;
    const Real b = AP*AB;

    const Real alpha = b / A;

    if (alpha < 0.000001 || alpha > 0.999999)
        return false;

    const Point::Coord& P = e1.p();
    const Line::Coord Q = e2.p1()+AB*alpha;
    const auto PQ = Q - P;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        // filter for LMD

        if (!useLMDFilters.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            const Vec3 QP = -PQ;
            return testValidity(e2, QP);
        }
        else
        {
            return true;
        }

        // end filter
    }
    else
        return false;
}

int LocalMinDistance::computeIntersection(Line& e2, Point& e1, OutputVector* contacts)
{
    static_assert(std::is_same_v<Line::Coord, Point::Coord>, "Data mismatch");
    using Real = Triangle::Coord::value_type;

    if(!e1.isActive(e2.getCollisionModel()) || !e2.isActive(e1.getCollisionModel()))
        return 0;

    const SReal alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const Line::Coord AB = e2.p2()-e2.p1();
    const Line::Coord AP = e1.p()-e2.p1();

    if (AB.norm()<0.000000000001*AP.norm())
    {
        return 0;
    }

    const Real A=AB*AB;
    const Real b=AP*AB;

    const Real alpha = b / A;
    if (alpha < 0.000001 || alpha > 0.999999)
        return 0;

    const Point::Coord& P = e1.p();
    const Line::Coord Q = e2.p1() + AB * alpha;
    const auto PQ = Q - P;

    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    const auto QP = -PQ;

    // filter for LMD
    if (!useLMDFilters.getValue())
    {
        if (!testValidity(e1, PQ))
            return 0;

        if (!testValidity(e2, QP))
            return 0;
    }

    // end filter

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

#ifdef SOFA_DETECTIONOUTPUT_FREEMOTION
    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        type::Vec3 ABfree = e2.p2Free() - e2.p1Free();
        type::Vec3 Pfree = e1.pFree();
        type::Vec3 Qfree = e2.p1Free() + ABfree * alpha;

        detection->freePoint[0] = Qfree;
        detection->freePoint[1] = Pfree;
    }
#endif

    const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
    detection->id = e1.getIndex();
    detection->point[0]=Q;
    detection->point[1]=P;
    detection->normal=QP;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;

    return 1;
}


bool LocalMinDistance::testIntersection(Line& e2, Sphere& e1)
{
    static_assert(std::is_same_v<Line::Coord, Sphere::Coord>, "Data mismatch");
    using Real = Triangle::Coord::value_type;

    const SReal alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();
    const Line::Coord AB = e2.p2()-e2.p1();
    const Line::Coord AP = e1.p()-e2.p1();

    const Real A = AB * AB;
    const Real b = AP * AB;

    const Real alpha = b / A;
    if (alpha < 0.000001 || alpha > 0.999999)
        return false;

    const Sphere::Coord& P = e1.p();
    const Line::Coord Q = e2.p1() + AB * alpha;
    const auto PQ = Q - P;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        // filter for LMD

        if (!useLMDFilters.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            const auto QP = -PQ;
            return testValidity(e2, QP);
        }
        else
        {
            return true;
        }

        // end filter
    }
    else
        return false;
}

int LocalMinDistance::computeIntersection(Line& e2, Sphere& e1, OutputVector* contacts)
{
    static_assert(std::is_same_v<Line::Coord, Sphere::Coord>, "Data mismatch");
    using Real = Triangle::Coord::value_type;

    const SReal alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();
    const Line::Coord AB = e2.p2()-e2.p1();
    const Line::Coord AP = e1.p()-e2.p1();

    if (AB.norm()<0.000000000001*AP.norm())
    {
        return 0;
    }

    const Real A = AB * AB;
    const Real b = AP * AB;

    const Real alpha = b / A;
    if (alpha < 0.000001 || alpha > 0.999999)
        return 0;

    const Sphere::Coord& P = e1.p();
    const Line::Coord Q = e2.p1()+AB*alpha;
    const auto PQ = Q - P;
    const auto QP = -PQ;

    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    // filter for LMD
    if (!useLMDFilters.getValue())
    {
        if (!testValidity(e1, PQ))
            return 0;

        if (!testValidity(e2, QP))
            return 0;
    }


    // end filter

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

#ifdef SOFA_DETECTIONOUTPUT_FREEMOTION
    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        type::Vec3 ABfree = e2.p2Free() - e2.p1Free();
        type::Vec3 Pfree = e1.pFree();
        type::Vec3 Qfree = e2.p1Free() + ABfree * alpha;

        detection->freePoint[0] = Qfree;
        detection->freePoint[1] = Pfree;
    }
#endif

    const double contactDist = getContactDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
    detection->id = e1.getIndex();
    detection->point[0]=Q;
    detection->point[1]=P;
    detection->normal=QP;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;

    return 1;
}

bool LocalMinDistance::testIntersection(Point& e1, Point& e2)
{
    if(!e1.isActive(e2.getCollisionModel()) || !e2.isActive(e1.getCollisionModel()))
        return false;

    const SReal alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

    const Point::Coord PQ = e2.p()-e1.p();

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        // filter for LMD

        if (!useLMDFilters.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            const Point::Coord QP = -PQ;
            return testValidity(e2, QP);
        }
        else
        {
            return true;
        }

        // end filter
    }
    else
        return false;
}

int LocalMinDistance::computeIntersection(Point& e1, Point& e2, OutputVector* contacts)
{
    if(!e1.isActive(e2.getCollisionModel()) || !e2.isActive(e1.getCollisionModel()))
        return 0;

    const SReal alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

    const Point::Coord& P = e1.p();
    const Point::Coord& Q = e2.p();
    const Point::Coord PQ = Q - P;

    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    // filter for LMD

    if (!useLMDFilters.getValue())
    {
        if (!testValidity(e1, PQ))
            return 0;

        const Point::Coord QP = -PQ;

        if (!testValidity(e2, QP))
            return 0;
    }

    // end filter

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

#ifdef SOFA_DETECTIONOUTPUT_FREEMOTION
    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        type::Vec3 Pfree,Qfree;
        Pfree = e1.pFree();
        Qfree = e2.pFree();

        detection->freePoint[0] = Pfree;
        detection->freePoint[1] = Qfree;
    }
#endif

    const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->id = (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex();
    detection->point[0]=P;
    detection->point[1]=Q;
    detection->normal=PQ;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;
    return 1;
}

bool LocalMinDistance::testIntersection(Sphere& e1, Point& e2)
{
    const SReal alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    static_assert(std::is_same_v<Sphere::Coord, Point::Coord>, "Data mismatch");
    const auto PQ = e2.p()-e1.p();

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        // filter for LMD

        if (!useLMDFilters.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            const auto QP = -PQ;
            return testValidity(e2, QP);
        }
        else
        {
            return true;
        }

        // end filter
    }
    else
        return false;
}

int LocalMinDistance::computeIntersection(Sphere& e1, Point& e2, OutputVector* contacts)
{
    const SReal alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();
    static_assert(std::is_same_v<Sphere::Coord, Point::Coord>, "Data mismatch");

    const Sphere::Coord& P = e1.p();
    const Point::Coord& Q = e2.p();
    const auto PQ = Q - P;

    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    // filter for LMD

    if (!useLMDFilters.getValue())
    {
        if (!testValidity(e1, PQ))
            return 0;

        const auto QP = -PQ;

        if (!testValidity(e2, QP))
            return 0;
    }

    // end filter

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

#ifdef SOFA_DETECTIONOUTPUT_FREEMOTION
    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        type::Vec3 Pfree,Qfree;
        Pfree = e1.pFree();
        Qfree = e2.pFree();

        detection->freePoint[0] = Pfree;
        detection->freePoint[1] = Qfree;
    }
#endif

    const double contactDist = getContactDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->id = (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex();
    detection->point[0]=P;
    detection->point[1]=Q;
    detection->normal=PQ;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;
    return 1;
}

bool LocalMinDistance::testIntersection(Sphere& e1, Sphere& e2)
{
    const SReal alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.r() + e2.getProximity();

    const Sphere::Coord PQ = e2.p()-e1.p();
    if (PQ.norm2() < alarmDist*alarmDist)
    {
        // filter for LMD

        if (!useLMDFilters.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            const Sphere::Coord QP = -PQ;
            return testValidity(e2, QP);
        }
        else
        {
            return true;
        }

        // end filter
    }
    else
        return false;
}

int LocalMinDistance::computeIntersection(Sphere& e1, Sphere& e2, OutputVector* contacts)
{
    const SReal alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.r() + e2.getProximity();

    const Sphere::Coord& P = e1.p();
    const Sphere::Coord& Q = e2.p();
    const Sphere::Coord PQ = Q - P;

    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    // filter for LMD

    if (!useLMDFilters.getValue())
    {
        if (!testValidity(e1, PQ))
            return 0;

        const Sphere::Coord QP = -PQ;
        if (!testValidity(e2, QP))
            return 0;
    }

    // end filter

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

#ifdef SOFA_DETECTIONOUTPUT_FREEMOTION
    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        type::Vec3 Pfree,Qfree;
        Pfree = e1.pFree();
        Qfree = e2.pFree();

        detection->freePoint[0] = Pfree;
        detection->freePoint[1] = Qfree;
    }
#endif

    const double contactDist = getContactDistance() + e1.r() + e1.getProximity() + e2.r() + e2.getProximity();

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->id = (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex();
    detection->point[0]=P;
    detection->point[1]=Q;
    detection->normal=PQ;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;
    return 1;
}


bool LocalMinDistance::testIntersection(Ray &t1,Triangle &t2)
{
    type::Vec3 P,Q;

    const SReal alarmDist = getAlarmDistance() + t1.getProximity() + t2.getProximity();

    if (fabs(t2.n() * t1.direction()) < 0.000001)
        return false; // no intersection for edges parallel to the triangle

    const Vec3 A = t1.origin();
    const Vec3 B = A + t1.direction() * t1.l();

    const auto r = sofa::geometry::proximity::computeClosestPointsSegmentAndTriangle(t2.p1(), t2.p2(), t2.p3(), A, B,P,Q);
    msg_warning_when(!r, "RayNewProximityIntersection") << "Failed to compute distance between ray ["
        << A << "," << B <<"] and triangle [" << t2.p1() << ", " << t2.p2() << ", " << t2.p3() << "]";

    const auto PQ=Q-P;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        return true;
    }
    else
        return false;
}

int LocalMinDistance::computeIntersection(Ray &t1, Triangle &t2, OutputVector* contacts)
{
    const SReal alarmDist = getAlarmDistance() + t1.getProximity() + t2.getProximity();


    if (fabs(t2.n() * t1.direction()) < 0.000001)
        return false; // no intersection for edges parallel to the triangle

    const Vec3 A = t1.origin();
    const Vec3 B = A + t1.direction() * t1.l();

    Vec3 P,Q;

    const auto r = sofa::geometry::proximity::computeClosestPointsSegmentAndTriangle(t2.p1(), t2.p2(), t2.p3(), A, B,P,Q);
    msg_warning_when(!r, "RayNewProximityIntersection") << "Failed to compute distance between ray ["
        << A << "," << B <<"] and triangle [" << t2.p1() << ", " << t2.p2() << ", " << t2.p3() << "]";

    const Vec3 PQ=Q-P;

    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    const double contactDist = alarmDist;
    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(t1, t2);
    detection->id = t1.getIndex();
    detection->point[1]=P;
    detection->point[0]=Q;
#ifdef SOFA_DETECTIONOUTPUT_FREEMOTION
    detection->freePoint[1] = P;
    detection->freePoint[0] = Q;
#endif
    detection->normal=-t2.n();
    detection->value = PQ.norm();
    detection->value -= contactDist;
    return 1;
}


bool LocalMinDistance::testIntersection(Ray &ray1,Sphere &sph2)
{
    // Center of the sphere
    const Vec3 sph2Pos(sph2.center());
    // Radius of the sphere
    const double radius1 = sph2.r();

    const Vec3 ray1Origin(ray1.origin());
    const Vec3 ray1Direction(ray1.direction());
    const double length2 = ray1.l();
    const Vec3 tmp = sph2Pos - ray1Origin;
    const double rayPos = tmp*ray1Direction;
    const double rayPosInside = std::max(std::min(rayPos,length2),0.0);
    const double dist2 = tmp.norm2() - (rayPosInside*rayPosInside);
    return (dist2 < (radius1*radius1));
}

int LocalMinDistance::computeIntersection(Ray &ray1, Sphere &sph2, OutputVector* contacts)
{
    // Center of the sphere
    const Vec3 sph2Pos(sph2.center());
    // Radius of the sphere
    const double radius1 = sph2.r();

    const Vec3 ray1Origin(ray1.origin());
    const Vec3 ray1Direction(ray1.direction());
    const double length2 = ray1.l();
    const Vec3 tmp = sph2Pos - ray1Origin;
    const double rayPos = tmp*ray1Direction;
    const double rayPosInside = std::max(std::min(rayPos,length2),0.0);
    const double dist2 = tmp.norm2() - (rayPosInside*rayPosInside);
    if (dist2 >= (radius1*radius1))
        return 0;

    const double dist = sqrt(dist2);

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

    detection->point[0] = ray1Origin + ray1Direction*rayPosInside;
    detection->normal = sph2Pos - detection->point[0];
    detection->normal /= dist;
    detection->point[1] = sph2Pos - detection->normal * radius1;
    detection->value = dist - radius1;
    detection->elem.first = ray1;
    detection->elem.second = sph2;
    detection->id = ray1.getIndex();
    return 1;
}


bool LocalMinDistance::testValidity(Point &p, const Vec3 &PQ) const
{
    if (!filterIntersection.getValue())
        return true;

    const Vec3 pt = p.p();

    const sofa::simulation::Node* node = dynamic_cast<sofa::simulation::Node*>(p.getCollisionModel()->getContext());
    if ( !(node->get< LineCollisionModel<sofa::defaulttype::Vec3Types> >()) )
        return true;

    BaseMeshTopology* topology = p.getCollisionModel()->getCollisionTopology();
    const auto& x =(p.getCollisionModel()->getMechanicalState()->read(core::ConstVecCoordId::position())->getValue());

    const auto& trianglesAroundVertex = topology->getTrianglesAroundVertex(p.getIndex());
    const auto& edgesAroundVertex = topology->getEdgesAroundVertex(p.getIndex());
    Vec3 nMean;

    for (unsigned int i=0; i<trianglesAroundVertex.size(); i++)
    {
        const unsigned int t = trianglesAroundVertex[i];
        const auto& ptr = topology->getTriangle(t);
        Vec3 nCur = (x[ptr[1]]-x[ptr[0]]).cross(x[ptr[2]]-x[ptr[0]]);
        nCur.normalize();
        nMean += nCur;
    }

    if (trianglesAroundVertex.empty())
    {
        for (unsigned int i=0; i<edgesAroundVertex.size(); i++)
        {
            const unsigned int e = edgesAroundVertex[i];
            const auto& ped = topology->getEdge(e);
            Vec3 l = (pt - x[ped[0]]) + (pt - x[ped[1]]);
            l.normalize();
            nMean += l;
        }
    }



    if (nMean.norm()> 0.0000000001)
    {
        /// validity test with nMean, except if bothSide
        const PointCollisionModel<sofa::defaulttype::Vec3Types> *pM = p.getCollisionModel();
        const bool bothSide_computation = pM->bothSide.getValue();
        nMean.normalize();
        if (dot(nMean, PQ) < -angleCone.getValue()*PQ.norm() && !bothSide_computation)
        {
            return false;
        }
    }

    for (unsigned int i=0; i<edgesAroundVertex.size(); i++)
    {
        const unsigned int e = edgesAroundVertex[i];
        const auto& ped = topology->getEdge(e);
        Vec3 l = (pt - x[ped[0]]) + (pt - x[ped[1]]);
        l.normalize();
        double computedAngleCone = dot(nMean , l) * coneFactor.getValue();
        if (computedAngleCone<0)
            computedAngleCone=0.0;
        computedAngleCone+=angleCone.getValue();
        if (dot(l , PQ) < -computedAngleCone*PQ.norm())
        {
            return false;
        }
    }

    return true;
}

bool LocalMinDistance::testValidity(Line &l, const Vec3 &PQ) const
{
    if (!filterIntersection.getValue())
        return true;

    const LineCollisionModel<sofa::defaulttype::Vec3Types> *lM = l.getCollisionModel();
    const bool bothSide_computation = lM->bothSide.getValue();

    Vec3 n1;

    const Line::Coord &pt1 = l.p1();
    const Line::Coord &pt2 = l.p2();

    Line::Coord AB = pt2 - pt1;
    AB.normalize();

    BaseMeshTopology* topology = l.getCollisionModel()->getCollisionTopology();
    const auto& x =(l.getCollisionModel()->getMechanicalState()->read(core::ConstVecCoordId::position())->getValue());
    const auto& trianglesAroundEdge = topology->getTrianglesAroundEdge(l.getIndex());

    if ( trianglesAroundEdge.size() == 2)
    {
        // which triangle is left ?
        const BaseMeshTopology::Triangle& triangle0 = topology->getTriangle(trianglesAroundEdge[0]);
        bool triangle0_is_left=false;
        if ( (l.i1()==triangle0[0]&&l.i2()==triangle0[1]) || (l.i1()==triangle0[1]&&l.i2()==triangle0[2]) || (l.i1()==triangle0[2]&&l.i2()==triangle0[0]) )
        {
            triangle0_is_left=true;
        }


        // compute the normal of the triangle situated on the right
        const BaseMeshTopology::Triangle& triangleRight = triangle0_is_left ? topology->getTriangle(trianglesAroundEdge[1]): topology->getTriangle(trianglesAroundEdge[0]);
        n1 = cross(x[triangleRight[1]]-x[triangleRight[0]], x[triangleRight[2]]-x[triangleRight[0]]);
        n1.normalize();
        Vec3 nMean=n1;
        Vec3 t1=cross(n1, AB);
        t1.normalize(); // necessary ?

        // compute the normal of the triangle situated on the left
        const BaseMeshTopology::Triangle& triangleLeft = triangle0_is_left ? topology->getTriangle(trianglesAroundEdge[0]): topology->getTriangle(trianglesAroundEdge[1]);
        Vec3 n2=cross(x[triangleLeft[1]]-x[triangleLeft[0]], x[triangleLeft[2]]-x[triangleLeft[0]]);
        n2.normalize();
        nMean += n2;
        Vec3 t2=cross(AB, n2);
        t2.normalize(); // necessary ?

        nMean.normalize();

        if ((nMean*PQ) < 0  && !bothSide_computation) // test
        {
            msg_info_when(EMIT_EXTRA_DEBUG_MESSAGE)
                    <<" rejected because of nMean: "<<nMean ;
            return false;
        }

        // compute the angle for the cone to filter contacts using the normal of the triangle situated on the right
        double computedAngleCone = (nMean * t1) * coneFactor.getValue();
        if (computedAngleCone<0)
            computedAngleCone=0.0;
        computedAngleCone+=angleCone.getValue();

        if (t1*PQ < -computedAngleCone*PQ.norm())
        {
            msg_info_when(EMIT_EXTRA_DEBUG_MESSAGE)
                    <<" rejected because of right triangle normal: "<<n1<<" tang "<< t1 ;
            return false;
        }

        // compute the angle for the cone to filter contacts using the normal of the triangle situated on the left
        computedAngleCone = (nMean * t2) * coneFactor.getValue();
        if (computedAngleCone<0)
            computedAngleCone=0.0;
        computedAngleCone+=angleCone.getValue();

        if (t2*PQ < -computedAngleCone*PQ.norm())
        {
            msg_info_when(EMIT_EXTRA_DEBUG_MESSAGE)
                <<" rejected because of left triangle normal: "<<n2 ;

            return false;
        }

    }
    else
    {
        n1 = PQ;
        n1.normalize();
        if (fabs(dot(AB,n1)) > angleCone.getValue() + 0.0001 )		// dot(AB,n1) should be equal to 0
        {
            // means that proximity was detected with a null determinant
            // in function computeIntersection
            msg_info_when(EMIT_EXTRA_DEBUG_MESSAGE)
                <<"bad case detected  -  abs(dot(AB,n1)) ="<<fabs(dot(AB,n1)) ;
            return false;
        }
    }
    return true;
}

bool LocalMinDistance::testValidity(Triangle &t, const Vec3 &PQ) const
{
    const TriangleCollisionModel<sofa::defaulttype::Vec3Types> *tM = t.getCollisionModel();
    const bool bothSide_computation = tM->d_bothSide.getValue();

    if (!filterIntersection.getValue()  || bothSide_computation)
        return true;

    const Vec3& pt1 = t.p1();
    const Vec3& pt2 = t.p2();
    const Vec3& pt3 = t.p3();

    const Vec3 n = cross(pt2-pt1,pt3-pt1);

    return n * PQ >= 0.0;
}

} //namespace sofa::component::collision::detection::intersection
