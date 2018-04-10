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
#define SOFA_COMPONENT_COLLISION_LOCALMINDISTANCE_CPP
#include <SofaConstraint/LocalMinDistance.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/proximity.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/collision/Intersection.inl>
#include <sofa/core/visual/VisualParams.h>

#define DYNAMIC_CONE_ANGLE_COMPUTATION

/// To emit extra debug message set this to true.
#define EMIT_EXTRA_DEBUG_MESSAGE false

namespace sofa
{

namespace core
{
    namespace collision
    {
        template class SOFA_CONSTRAINT_API IntersectorFactory<component::collision::LocalMinDistance>;
    }
}

namespace component
{

namespace collision
{

using namespace sofa::core::collision;
using namespace helper;
using namespace sofa::defaulttype;

using core::topology::BaseMeshTopology;


SOFA_DECL_CLASS(LocalMinDistance)

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
    intersectors.add<CubeModel, CubeModel, LocalMinDistance>(this);

    //intersectors.ignore<SphereModel, PointModel>();		// SphereModel are not supported yet
    //intersectors.ignore<LineModel, SphereModel>();
    //intersectors.ignore<TriangleModel, SphereModel>();

    intersectors.add<SphereModel, SphereModel, LocalMinDistance>(this); // sphere-sphere is always activated
    intersectors.add<SphereModel, PointModel, LocalMinDistance>(this); // sphere-point is always activated

    intersectors.add<PointModel, PointModel, LocalMinDistance>(this); // point-point is always activated
    intersectors.add<LineModel, LineModel, LocalMinDistance>(this);
    intersectors.add<LineModel, PointModel, LocalMinDistance>(this);
    intersectors.add<LineModel, SphereModel, LocalMinDistance>(this);
    intersectors.add<TriangleModel, PointModel, LocalMinDistance>(this);
    intersectors.add<TriangleModel, SphereModel, LocalMinDistance>(this);

    intersectors.ignore<TriangleModel, LineModel>();			// never the case with LMD
    intersectors.ignore<TriangleModel, TriangleModel>();		// never the case with LMD

    intersectors.ignore<RayModel, PointModel>();
    intersectors.ignore<RayModel, LineModel>();
    intersectors.add<RayModel, TriangleModel, LocalMinDistance>(this);
    intersectors.add<RayModel, SphereModel, LocalMinDistance>(this);
    IntersectorFactory::getInstance()->addIntersectors(this);

    BaseProximityIntersection::init();
}

bool LocalMinDistance::testIntersection(Cube &cube1, Cube &cube2)
{
    const Vector3& minVect1 = cube1.minVect();
    const Vector3& minVect2 = cube2.minVect();
    const Vector3& maxVect1 = cube1.maxVect();
    const Vector3& maxVect2 = cube2.maxVect();

    const double alarmDist = getAlarmDistance() + cube1.getProximity() + cube2.getProximity();

    for (int i=0; i<3; i++)
    {
        if ( minVect1[i] > maxVect2[i] + alarmDist || minVect2[i]> maxVect1[i] + alarmDist )
            return false;
    }

    return true;
}

int LocalMinDistance::computeIntersection(Cube&, Cube&, OutputVector* /*contacts*/)
{
    return 0; /// \todo
}

bool LocalMinDistance::testIntersection(Line& e1, Line& e2)
{
    if(!e1.activated(e2.getCollisionModel()) || !e2.activated(e1.getCollisionModel()))
    {
        return false;
    }

    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

    const Vector3 AB = e1.p2()-e1.p1();
    const Vector3 CD = e2.p2()-e2.p1();
    const Vector3 AC = e2.p1()-e1.p1();
    Matrix2 A;
    Vector2 b;

    A[0][0] = AB*AB;
    A[1][1] = CD*CD;
    A[0][1] = A[1][0] = -CD*AB;
    b[0] = AB*AC;
    b[1] = -CD*AC;

    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

    if (det < -1.0e-30 || det > 1.0e-30)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        if (alpha < 1e-15 || alpha > (1.0-1e-15) ||
            beta  < 1e-15  || beta  > (1.0-1e-15) )
            return false;
    }

    Vector3 PQ = AC + CD * beta - AB * alpha;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        // filter for LMD

        if (!useLMDFilters.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            Vector3 QP = -PQ;
            return testValidity(e2, QP);
        }
        else
        {
            /*
            core::collision::ContactFiltrationAlgorithm *e1_cfa = e1.getCollisionModel()->getContactFiltrationAlgorithm();
            if (e1_cfa != 0)
            {
                if (!e1_cfa->validate(e1, PQ))
                    return false;
            }

            core::collision::ContactFiltrationAlgorithm *e2_cfa = e2.getCollisionModel()->getContactFiltrationAlgorithm();
            if (e2_cfa != 0)
            {
                Vector3 QP = -PQ;
                return e2_cfa->validate(e2, QP);
            }
            */

            return true;
        }

        // end filter

    }
    else
        return false;
}

int LocalMinDistance::computeIntersection(Line& e1, Line& e2, OutputVector* contacts)
{

    if(!e1.activated(e2.getCollisionModel()) || !e2.activated(e1.getCollisionModel()))
    {
        dmsg_info_when(EMIT_EXTRA_DEBUG_MESSAGE)
            <<" not activated" ;
        return 0;
    }

    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

    // E1 => A-->B
    // E2 => C-->D
    const Vector3 AB = e1.p2()-e1.p1();
    const Vector3 CD = e2.p2()-e2.p1();
    const Vector3 AC = e2.p1()-e1.p1();
    Matrix2 A;
    Vector2 b;

    A[0][0] = AB*AB;
    A[1][1] = CD*CD;
    A[0][1] = A[1][0] = -CD*AB;
    b[0] = AB*AC;
    b[1] = -CD*AC;
    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

    if (det < -1.0e-30 || det > 1.0e-30)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        if (alpha < 1e-15 || alpha > (1.0-1e-15) ||
            beta  < 1e-15  || beta  > (1.0-1e-15) )
            return 0;
    }
    else
    {
        // several possibilities :
        // -one point in common (auto-collision) => return false !
        // -no point in common but line are // => we can continue to test

        sout<<"WARNING det is null"<<sendl;
    }

    Vector3 P,Q,PQ;
    P = e1.p1() + AB * alpha;
    Q = e2.p1() + CD * beta;
    PQ = Q-P;

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

        Vector3 QP = -PQ;

        if (!testValidity(e2, QP))
        {
            dmsg_info_when(EMIT_EXTRA_DEBUG_MESSAGE)
                <<" testValidity rejected for the second segment";
            return 0;
        }
    }

    contacts->resize(contacts->size() + 1);
    DetectionOutput *detection = &*(contacts->end() - 1);

#ifdef DETECTIONOUTPUT_FREEMOTION

    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        Vector3 Pfree, Qfree, ABfree, CDfree;
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
    detection->baryCoords[0][0] = alpha;
    detection->baryCoords[1][0] = beta;
#endif
    detection->normal = PQ;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;

    return 1;
}

bool LocalMinDistance::testIntersection(Triangle& e2, Point& e1)
{
    if(!e1.activated(e2.getCollisionModel()))
        return false;

    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

    const Vector3 AB = e2.p2()-e2.p1();
    const Vector3 AC = e2.p3()-e2.p1();
    const Vector3 AP = e1.p() -e2.p1();
    Matrix2 A;
    Vector2 b;

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
    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;


    alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
    beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
    if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
        return false;

    const Vector3 PQ = AB * alpha + AC * beta - AP;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        //filter for LMD
        if (!useLMDFilters.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            Vector3 QP = -PQ;
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
    if(!e1.activated(e2.getCollisionModel()))
        return 0;

    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

    const Vector3 AB = e2.p2()-e2.p1();
    const Vector3 AC = e2.p3()-e2.p1();
    const Vector3 AP = e1.p() -e2.p1();
    Matrix2 A;
    Vector2 b;

    A[0][0] = AB*AB;
    A[1][1] = AC*AC;
    A[0][1] = A[1][0] = AB*AC;
    b[0] = AP*AB;
    b[1] = AP*AC;



    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;


    alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
    beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
    if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
        return 0;


    Vector3 P,Q; //PQ
    P = e1.p();
    Q = e2.p1() + AB * alpha + AC * beta;
    Vector3 PQ = Q-P;
    Vector3 QP = -PQ;

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

    //end filter

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

#ifdef DETECTIONOUTPUT_FREEMOTION
    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        Vector3 Pfree,Qfree,ABfree,ACfree;
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
    detection->baryCoords[0][0] = 0;
    detection->baryCoords[1][0] = alpha;
    detection->baryCoords[1][1] = beta;
#endif
    detection->normal = QP;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;
    return 1;
}


bool LocalMinDistance::testIntersection(Triangle& e2, Sphere& e1)
{
    const double alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    const Vector3 AB = e2.p2()-e2.p1();
    const Vector3 AC = e2.p3()-e2.p1();
    const Vector3 AP = e1.p() -e2.p1();
    Matrix2 A;
    Vector2 b;

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
    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;


    alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
    beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
    if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
        return false;


    const Vector3 PQ = AB * alpha + AC * beta - AP;

    if (PQ.norm2() < alarmDist*alarmDist)
    {

        //filter for LMD

        if (!useLMDFilters.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            Vector3 QP = -PQ;
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

    const double alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    const Vector3 AB = e2.p2()-e2.p1();
    const Vector3 AC = e2.p3()-e2.p1();
    const Vector3 AP = e1.p() -e2.p1();
    Matrix2 A;
    Vector2 b;

    A[0][0] = AB*AB;
    A[1][1] = AC*AC;
    A[0][1] = A[1][0] = AB*AC;
    b[0] = AP*AB;
    b[1] = AP*AC;



    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

    alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
    beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
    if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
        return 0;

    Vector3 P,Q; //PQ
    P = e1.p();
    Q = e2.p1() + AB * alpha + AC * beta;
    Vector3 PQ = Q-P;
    Vector3 QP = -PQ;

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

    //end filter

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

#ifdef DETECTIONOUTPUT_FREEMOTION
    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        Vector3 Pfree,Qfree,ABfree,ACfree;
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
    detection->baryCoords[0][0] = 0;
    detection->baryCoords[1][0] = alpha;
    detection->baryCoords[1][1] = beta;
#endif
    detection->normal = QP;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;
    return 1;
}

bool LocalMinDistance::testIntersection(Line& e2, Point& e1)
{

    if(!e1.activated(e2.getCollisionModel()) || !e2.activated(e1.getCollisionModel()))
        return false;

    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const Vector3 AB = e2.p2()-e2.p1();
    const Vector3 AP = e1.p()-e2.p1();

    double A;
    double b;
    A = AB*AB;
    b = AP*AB;

    double alpha = 0.5;


    alpha = b/A;
    if (alpha < 0.000001 || alpha > 0.999999)
        return false;


    Vector3 P,Q,PQ;
    P = e1.p();
    Q = e2.p1() + AB * alpha;
    PQ = Q-P;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        // filter for LMD

        if (!useLMDFilters.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            Vector3 QP = -PQ;
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
    if(!e1.activated(e2.getCollisionModel()) || !e2.activated(e1.getCollisionModel()))
        return 0;

    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const Vector3 AB = e2.p2()-e2.p1();
    const Vector3 AP = e1.p()-e2.p1();

    if (AB.norm()<0.000000000001*AP.norm())
    {
        return 0;
    }


    double A;
    double b;
    A = AB*AB;
    b = AP*AB;

    double alpha = 0.5;

    alpha = b/A;
    if (alpha < 0.000001 || alpha > 0.999999)
        return 0;

    Vector3 P,Q;
    P = e1.p();
    Q = e2.p1() + AB * alpha;
    Vector3 PQ = Q - P;
    Vector3 QP = -PQ;

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

#ifdef DETECTIONOUTPUT_FREEMOTION
    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        Vector3 ABfree = e2.p2Free() - e2.p1Free();
        Vector3 Pfree = e1.pFree();
        Vector3 Qfree = e2.p1Free() + ABfree * alpha;

        detection->freePoint[0] = Qfree;
        detection->freePoint[1] = Pfree;
    }
#endif

    const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
    detection->id = e1.getIndex();
    detection->point[0]=Q;
    detection->point[1]=P;
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
    detection->baryCoords[0][0]=0;
    detection->baryCoords[1][0]=alpha;
#endif
    detection->normal=QP;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;

    return 1;
}


bool LocalMinDistance::testIntersection(Line& e2, Sphere& e1)
{

    const double alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();
    const Vector3 AB = e2.p2()-e2.p1();
    const Vector3 AP = e1.p()-e2.p1();

    double A;
    double b;
    A = AB*AB;
    b = AP*AB;

    double alpha = 0.5;


    alpha = b/A;
    if (alpha < 0.000001 || alpha > 0.999999)
        return false;


    Vector3 P,Q,PQ;
    P = e1.p();
    Q = e2.p1() + AB * alpha;
    PQ = Q-P;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        // filter for LMD

        if (!useLMDFilters.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            Vector3 QP = -PQ;
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
    const double alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();
    const Vector3 AB = e2.p2()-e2.p1();
    const Vector3 AP = e1.p()-e2.p1();

    if (AB.norm()<0.000000000001*AP.norm())
    {
        return 0;
    }


    double A;
    double b;
    A = AB*AB;
    b = AP*AB;

    double alpha = 0.5;


    alpha = b/A;
    if (alpha < 0.000001 || alpha > 0.999999)
        return 0;


    Vector3 P,Q;
    P = e1.p();
    Q = e2.p1() + AB * alpha;
    Vector3 PQ = Q - P;
    Vector3 QP = -PQ;

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

#ifdef DETECTIONOUTPUT_FREEMOTION
    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        Vector3 ABfree = e2.p2Free() - e2.p1Free();
        Vector3 Pfree = e1.pFree();
        Vector3 Qfree = e2.p1Free() + ABfree * alpha;

        detection->freePoint[0] = Qfree;
        detection->freePoint[1] = Pfree;
    }
#endif

    const double contactDist = getContactDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
    detection->id = e1.getIndex();
    detection->point[0]=Q;
    detection->point[1]=P;
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
    detection->baryCoords[0][0]=0;
    detection->baryCoords[1][0]=alpha;
#endif
    detection->normal=QP;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;

    return 1;
}

bool LocalMinDistance::testIntersection(Point& e1, Point& e2)
{
    if(!e1.activated(e2.getCollisionModel()) || !e2.activated(e1.getCollisionModel()))
        return 0;

    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

    Vector3 PQ = e2.p()-e1.p();

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        // filter for LMD

        if (!useLMDFilters.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            Vector3 QP = -PQ;
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
    if(!e1.activated(e2.getCollisionModel()) || !e2.activated(e1.getCollisionModel()))
        return 0;

    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

    Vector3 P,Q,PQ;
    P = e1.p();
    Q = e2.p();
    PQ = Q-P;


    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    // filter for LMD

    if (!useLMDFilters.getValue())
    {
        if (!testValidity(e1, PQ))
            return 0;

        Vector3 QP = -PQ;

        if (!testValidity(e2, QP))
            return 0;
    }

    // end filter

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

#ifdef DETECTIONOUTPUT_FREEMOTION
    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        Vector3 Pfree,Qfree;
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
    detection->baryCoords[0][0]=0;
    detection->baryCoords[1][0]=0;
#endif
    detection->normal=PQ;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;
    return 1;
}

bool LocalMinDistance::testIntersection(Sphere& e1, Point& e2)
{
    const double alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    Vector3 PQ = e2.p()-e1.p();

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        // filter for LMD

        if (!useLMDFilters.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            Vector3 QP = -PQ;
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
    const double alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    Vector3 P,Q,PQ;
    P = e1.p();
    Q = e2.p();
    PQ = Q-P;


    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    // filter for LMD

    if (!useLMDFilters.getValue())
    {
        if (!testValidity(e1, PQ))
            return 0;

        Vector3 QP = -PQ;

        if (!testValidity(e2, QP))
            return 0;
    }

    // end filter

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

#ifdef DETECTIONOUTPUT_FREEMOTION
    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        Vector3 Pfree,Qfree;
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
    detection->baryCoords[0][0]=0;
    detection->baryCoords[1][0]=0;
#endif
    detection->normal=PQ;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;
    return 1;
}

bool LocalMinDistance::testIntersection(Sphere& e1, Sphere& e2)
{
    const double alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.r() + e2.getProximity();

    Vector3 PQ = e2.p()-e1.p();

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        // filter for LMD

        if (!useLMDFilters.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            Vector3 QP = -PQ;
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
    const double alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.r() + e2.getProximity();

    Vector3 P,Q,PQ;
    P = e1.p();
    Q = e2.p();
    PQ = Q-P;


    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    // filter for LMD

    if (!useLMDFilters.getValue())
    {
        if (!testValidity(e1, PQ))
            return 0;

        Vector3 QP = -PQ;

        if (!testValidity(e2, QP))
            return 0;
    }

    // end filter

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

#ifdef DETECTIONOUTPUT_FREEMOTION
    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        Vector3 Pfree,Qfree;
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
    detection->baryCoords[0][0]=0;
    detection->baryCoords[1][0]=0;
#endif
    detection->normal=PQ;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;
    return 1;
}


bool LocalMinDistance::testIntersection(Ray &t1,Triangle &t2)
{
    Vector3 P,Q,PQ;
    static DistanceSegTri proximitySolver;

    const double alarmDist = getAlarmDistance() + t1.getProximity() + t2.getProximity();

    if (fabs(t2.n() * t1.direction()) < 0.000001)
        return false; // no intersection for edges parallel to the triangle

    Vector3 A = t1.origin();
    Vector3 B = A + t1.direction() * t1.l();

    proximitySolver.NewComputation( t2.p1(), t2.p2(), t2.p3(), A, B,P,Q);
    PQ = Q-P;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        return true;
    }
    else
        return false;
}

int LocalMinDistance::computeIntersection(Ray &t1, Triangle &t2, OutputVector* contacts)
{
    const double alarmDist = getAlarmDistance() + t1.getProximity() + t2.getProximity();


    if (fabs(t2.n() * t1.direction()) < 0.000001)
        return false; // no intersection for edges parallel to the triangle

    Vector3 A = t1.origin();
    Vector3 B = A + t1.direction() * t1.l();

    Vector3 P,Q,PQ;
    static DistanceSegTri proximitySolver;

    proximitySolver.NewComputation( t2.p1(), t2.p2(), t2.p3(), A,B,P,Q);
    PQ = Q-P;

    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    const double contactDist = alarmDist;
    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(t1, t2);
    detection->id = t1.getIndex();
    detection->point[1]=P;
    detection->point[0]=Q;
#ifdef DETECTIONOUTPUT_FREEMOTION
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
    const Vector3 sph2Pos(sph2.center());
    // Radius of the sphere
    const double radius1 = sph2.r();

    const Vector3 ray1Origin(ray1.origin());
    const Vector3 ray1Direction(ray1.direction());
    const double length2 = ray1.l();
    const Vector3 tmp = sph2Pos - ray1Origin;
    const double rayPos = tmp*ray1Direction;
    const double rayPosInside = std::max(std::min(rayPos,length2),0.0);
    const double dist2 = tmp.norm2() - (rayPosInside*rayPosInside);
    return (dist2 < (radius1*radius1));
}

int LocalMinDistance::computeIntersection(Ray &ray1, Sphere &sph2, OutputVector* contacts)
{
    // Center of the sphere
    const Vector3 sph2Pos(sph2.center());
    // Radius of the sphere
    const double radius1 = sph2.r();

    const Vector3 ray1Origin(ray1.origin());
    const Vector3 ray1Direction(ray1.direction());
    const double length2 = ray1.l();
    const Vector3 tmp = sph2Pos - ray1Origin;
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


bool LocalMinDistance::testValidity(Point &p, const Vector3 &PQ)
{
    if (!filterIntersection.getValue())
        return true;

    Vector3 pt = p.p();

    sofa::simulation::Node* node = dynamic_cast<sofa::simulation::Node*>(p.getCollisionModel()->getContext());
    if ( !(node->get< LineModel >()) )
        return true;

    BaseMeshTopology* topology = p.getCollisionModel()->getMeshTopology();
    const helper::vector<Vector3>& x =(p.getCollisionModel()->getMechanicalState()->read(core::ConstVecCoordId::position())->getValue());

    const helper::vector <unsigned int>& trianglesAroundVertex = topology->getTrianglesAroundVertex(p.getIndex());
    const helper::vector <unsigned int>& edgesAroundVertex = topology->getEdgesAroundVertex(p.getIndex());
    Vector3 nMean;

    for (unsigned int i=0; i<trianglesAroundVertex.size(); i++)
    {
        unsigned int t = trianglesAroundVertex[i];
        const fixed_array<unsigned int,3>& ptr = topology->getTriangle(t);
        Vector3 nCur = (x[ptr[1]]-x[ptr[0]]).cross(x[ptr[2]]-x[ptr[0]]);
        nCur.normalize();
        nMean += nCur;
    }

    if (trianglesAroundVertex.size()==0)
    {
        for (unsigned int i=0; i<edgesAroundVertex.size(); i++)
        {
            unsigned int e = edgesAroundVertex[i];
            const fixed_array<unsigned int,2>& ped = topology->getEdge(e);
            Vector3 l = (pt - x[ped[0]]) + (pt - x[ped[1]]);
            l.normalize();
            nMean += l;
        }
    }



    if (nMean.norm()> 0.0000000001)
    {
        /// validity test with nMean, except if bothSide
        PointModel *pM = p.getCollisionModel();
        bool bothSide_computation = pM->bothSide.getValue();
        nMean.normalize();
        if (dot(nMean, PQ) < -angleCone.getValue()*PQ.norm() && !bothSide_computation)
        {
            return false;
        }
    }

    for (unsigned int i=0; i<edgesAroundVertex.size(); i++)
    {
        unsigned int e = edgesAroundVertex[i];
        const fixed_array<unsigned int,2>& ped = topology->getEdge(e);
        Vector3 l = (pt - x[ped[0]]) + (pt - x[ped[1]]);
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

bool LocalMinDistance::testValidity(Line &l, const Vector3 &PQ)
{
    if (!filterIntersection.getValue())
        return true;

    LineModel *lM = l.getCollisionModel();
    bool bothSide_computation = lM->bothSide.getValue();

    Vector3 nMean;
    Vector3 n1, n2;
    Vector3 t1, t2;

    const Vector3 &pt1 = l.p1();
    const Vector3 &pt2 = l.p2();

    Vector3 AB = pt2 - pt1;
    AB.normalize();

    BaseMeshTopology* topology = l.getCollisionModel()->getMeshTopology();
    const helper::vector<Vector3>& x =(l.getCollisionModel()->getMechanicalState()->read(core::ConstVecCoordId::position())->getValue());
    const sofa::helper::vector<unsigned int>& trianglesAroundEdge = topology->getTrianglesAroundEdge(l.getIndex());

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
        //const BaseMeshTopology::Triangle& triangleRight = topology->getTriangle(trianglesAroundEdge[0]);
        n1 = cross(x[triangleRight[1]]-x[triangleRight[0]], x[triangleRight[2]]-x[triangleRight[0]]);
        n1.normalize();
        nMean = n1;
        t1 = cross(n1, AB);
        t1.normalize(); // necessary ?

        // compute the normal of the triangle situated on the left
        const BaseMeshTopology::Triangle& triangleLeft = triangle0_is_left ? topology->getTriangle(trianglesAroundEdge[0]): topology->getTriangle(trianglesAroundEdge[1]);
        //const fixed_array<PointID,3>& triangleLeft = topology->getTriangle(trianglesAroundEdge[1]);
        n2 = cross(x[triangleLeft[1]]-x[triangleLeft[0]], x[triangleLeft[2]]-x[triangleLeft[0]]);
        n2.normalize();
        nMean += n2;
        t2 = cross(AB, n2);
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
        //
        ///////// ??? /////////
        if (fabs(dot(AB,n1)) > angleCone.getValue() + 0.0001 )		// dot(AB,n1) should be equal to 0
        {
            // means that proximity was detected with a null determinant
            // in function computeIntersection
            msg_info_when(EMIT_EXTRA_DEBUG_MESSAGE)
                <<"bad case detected  -  abs(dot(AB,n1)) ="<<fabs(dot(AB,n1)) ;
            return false;
        }
        //////////////////////

    }
    //sout<<"trianglesAroundEdge.size()"<<trianglesAroundEdge.size()<<sendl;
    return true;
}

bool LocalMinDistance::testValidity(Triangle &t, const Vector3 &PQ)
{
    TriangleModel *tM = t.getCollisionModel();
    bool bothSide_computation = tM->bothSide.getValue();

    if (!filterIntersection.getValue()  || bothSide_computation)
        return true;

    const Vector3& pt1 = t.p1();
    const Vector3& pt2 = t.p2();
    const Vector3& pt3 = t.p3();

    Vector3 n = cross(pt2-pt1,pt3-pt1);

    return ( (n*PQ) >= 0.0);
}


void LocalMinDistance::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowCollisionModels())
        return;
}

} // namespace collision

} // namespace component

} // namespace sofa

