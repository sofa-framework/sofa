#include "Common/FnDispatcher.h"
#include "Intersection.h"
//#include "ContinuousTriangleIntersection.h"

#include <iostream>
#include <algorithm>

namespace Sofa
{

namespace Components
{

namespace Intersections
{

using namespace Common;
using namespace Collision;

Intersection Intersection::instance;

Intersection::Intersection()
{
    FnCollisionDetection::Add<Cube,Cube,intersectionCubeCube,false>();
    FnCollisionDetection::Add<Sphere,Sphere,intersectionSphereSphere,false>();
    //FnCollisionDetection::Add<Sphere,Triangle,intersectionSphereTriangle,true>();
    //FnCollisionDetection::Add<Triangle,Triangle,intersectionSphereSphere,false>();
    FnCollisionDetection::Add<Sphere,Ray,intersectionSphereRay,true>();

    //FnCollisionDetectionOutput::Add<Cube,Cube,distCorrectionCubeCube,false>();
    FnCollisionDetectionOutput::Add<Sphere,Sphere,distCorrectionSphereSphere,false>();
    FnCollisionDetectionOutput::Add<Sphere,Ray,distCorrectionSphereRay,true>();
    //FnCollisionDetectionOutput::Add<Sphere,Triangle,distCorrectionSphereTriangle,true>();
    //FnCollisionDetectionOutput::Add<Triangle,Triangle,distCorrectionSphereSphere,false>();
}

/*
static void projectOntoAxis (const Triangle rkTri, const Vector3& rkAxis, double& rfMin, double& rfMax)
{
	double fDot0 = rkAxis.Dot(*rkTri.p1);
    double fDot1 = rkAxis.Dot(*rkTri.p2);
    double fDot2 = rkAxis.Dot(*rkTri.p3);

    rfMin = fDot0;
    rfMax = rfMin;

    if ( fDot1 < rfMin )
        rfMin = fDot1;
    else if ( fDot1 > rfMax )
        rfMax = fDot1;

    if ( fDot2 < rfMin )
        rfMin = fDot2;
    else if ( fDot2 > rfMax )
        rfMax = fDot2;
}
*/

bool intersectionSphereSphere(Sphere &sph1 ,Sphere &sph2)
{
    //std::cout<<"Collision between Sphere - Sphere"<<std::endl;
    Vector3 sph1Pos(sph1.center());
    Vector3 sph2Pos(sph2.center());
    double radius1 = sph1.r(), radius2 = sph2.r();
    Vector3 tmp = sph1Pos - sph2Pos;

    return (tmp.norm2() < (radius1 + radius2) * (radius1 + radius2));
}

bool intersectionCubeCube(Cube &cube1, Cube &cube2)
{
    const Vector3& minVect1 = cube1.minVect();
    const Vector3& minVect2 = cube2.minVect();
    const Vector3& maxVect1 = cube1.maxVect();
    const Vector3& maxVect2 = cube2.maxVect();
    for (int i=0; i<3; i++)
    {
        // Why so complicated?
        //if (!((((minVect1[i] >= minVect2[i]) && (minVect1[i] <= maxVect2[i]))
        //	|| ((maxVect1[i] >= minVect2[i]) && (maxVect1[i] <= maxVect2[i])))
        //	|| (((minVect2[i] >= minVect1[i]) && (minVect2[i] <= maxVect1[i]))
        //	|| ((maxVect2[i] >= minVect1[i]) && (maxVect2[i] <= maxVect1[i])))))

        if (minVect1[i] > maxVect2[i] || minVect2[i] > maxVect1[i])
            return false;
    }
    //std::cout << "Box <"<<minVect1[0]<<","<<minVect1[1]<<","<<minVect1[2]<<">-<"<<maxVect1[0]<<","<<maxVect1[1]<<","<<maxVect1[2]
    //  <<"> collide with Box "<<minVect2[0]<<","<<minVect2[1]<<","<<minVect2[2]<<">-<"<<maxVect2[0]<<","<<maxVect2[1]<<","<<maxVect2[2]<<">"<<std::endl;
    return true;
}

bool intersectionSphereRay(Sphere &sph1 ,Ray &ray2)
{
    //std::cout<<"Collision between Sphere - Ray"<<std::endl;

    const Vector3 sph1Pos(sph1.center());
    const double radius1 = sph1.r();
    const Vector3 ray2Origin(ray2.origin());
    const Vector3 ray2Direction(ray2.direction());
    const double length2 = ray2.l();
    const Vector3 tmp = sph1Pos - ray2Origin;
    const double rayPos = tmp*ray2Direction;
    const double rayPosInside = std::max(std::min(rayPos,length2),0.0);
    const double dist2 = tmp.norm2() - (rayPosInside*rayPosInside);
    return (dist2 < (radius1*radius1));
}

/*
bool intersectionSphereTriangle(Sphere &, Triangle &)
{
	std::cout<<"Collision between Sphere - Triangle"<<std::endl;
	return false;
}

bool intersectionTriangleTriangle (Triangle& t1, Triangle& t2)
{
	 Vector3 akE0[3](
	 	*(t1.p2) - *(t1.p1),
        *(t1.p3) - *(t1.p2),
        *(t1.p1) - *(t1.p3)
    );

    // get normal vector of triangle0
	 Vector3 kN0 = akE0[0].UnitCross(akE0[1]);

    // project triangle1 onto normal line of triangle0, test for separation
    double fN0dT0V0 = kN0.Dot(*(t1.p1));
    double fMin1, fMax1;
    projectOntoAxis(t2, kN0, fMin1, fMax1);
    if ( fN0dT0V0 < fMin1 || fN0dT0V0 > fMax1 )
        return false;

    // get edge vectors for triangle1
    Vector3 akE1[3] =
    {
        *(t2.p2) - *(t2.p1),
        *(t2.p3) - *(t2.p2),
        *(t2.p1) - *(t2.p3)
    };

    // get normal vector of triangle1
    Vector3 kN1 = akE1[0].UnitCross(akE1[1]);

    Vector3 kDir;
    double fMin0, fMax0;
    int i0, i1;

    Vector3 kN0xN1 = kN0.UnitCross(kN1);
    if ( kN0xN1.Dot(kN0xN1) >= 1e-08 )
    {
        // triangles are not parallel

        // Project triangle0 onto normal line of triangle1, test for
        // separation.
        double fN1dT1V0 = kN1.Dot(*t2.p1);
        projectOntoAxis(t1, kN1, fMin0, fMax0);
        if ( fN1dT1V0 < fMin0 || fN1dT1V0 > fMax0 )
            return false;

        // directions E0[i0]xE1[i1]
        for (i1 = 0; i1 < 3; i1++)
        {
            for (i0 = 0; i0 < 3; i0++)
            {
                kDir = akE0[i0].UnitCross(akE1[i1]);
                projectOntoAxis(t1, kDir, fMin0, fMax0);
                projectOntoAxis(t2, kDir, fMin1, fMax1);
                if ( fMax0 < fMin1 || fMax1 < fMin0 )
                    return false;
            }
        }
    }
    else  // triangles are parallel (and, in fact, coplanar)
    {
        // directions N0xE0[i0]
        for (i0 = 0; i0 < 3; i0++)
        {
            kDir = kN0.UnitCross(akE0[i0]);
            projectOntoAxis(t1, kDir, fMin0, fMax0);
            projectOntoAxis(t2, kDir, fMin1, fMax1);
            if ( fMax0 < fMin1 || fMax1 < fMin0 )
                return false;
        }

        // directions N1xE1[i1]
        for (i1 = 0; i1 < 3; i1++)
        {
            kDir = kN1.UnitCross(akE1[i1]);
            projectOntoAxis(t1, kDir, fMin0, fMax0);
            projectOntoAxis(t2, kDir, fMin1, fMax1);
            if ( fMax0 < fMin1 || fMax1 < fMin0 )
                return false;
        }
    }
	return true;
	//return false;
}
*/

DetectionOutput* distCorrectionSphereSphere(Sphere &sph1 ,Sphere &sph2)
{
    DetectionOutput *detection = new DetectionOutput();
    double distSph1Sph2 = (sph2.center() - sph1.center()).norm();
    double t1 = sph1.r() / distSph1Sph2;
    double t2 = (distSph1Sph2 - sph2.r()) / distSph1Sph2;

    detection->point[0] = sph1.center() + ((sph2.center() - sph1.center()) * t1);
    detection->point[1] = sph1.center() + ((sph2.center() - sph1.center()) * t2);

    detection->distance = ((distSph1Sph2 - (sph1.r() + sph2.r())) >= 0);
    detection->elem.first = &sph1;
    detection->elem.second = &sph2;

    return detection;
}

DetectionOutput* distCorrectionSphereRay(Sphere &sph1 ,Ray &ray2)
{
    const Vector3 sph1Pos(sph1.center());
    const double radius1 = sph1.r();
    const Vector3 ray2Origin(ray2.origin());
    const Vector3 ray2Direction(ray2.direction());
    const double length2 = ray2.l();
    const Vector3 tmp = sph1Pos - ray2Origin;
    const double rayPos = tmp*ray2Direction;
    const double rayPosInside = std::max(std::min(rayPos,length2),0.0);
    const double dist = sqrt(tmp.norm2() - (rayPosInside*rayPosInside));

    DetectionOutput *detection = new DetectionOutput();
    double t1 = radius1 / dist;

    detection->point[1] = ray2Origin + ray2Direction*rayPosInside;
    detection->point[0] = sph1Pos + ((detection->point[1] - sph1Pos) * t1);

    detection->distance = ((dist - radius1) >= 0);
    detection->elem.first = &sph1;
    detection->elem.second = &ray2;

    return detection;
}

/*
DetectionOutput* distCorrectionSphereTriangle(Sphere &, Triangle &)
{
	std::cout<<"Distance correction between Sphere - Triangle"<<std::endl;
	return new DetectionOutput();
}

DetectionOutput* distCorrectionTriangleTriangle (Triangle &t1, Triangle &t2)
{
	ContinuousTriangleIntersection intersectionT(t1, t2);

	//std::cout<<"Distance correction between Triangle - Triangle"<<std::endl;
	return intersectionT.computeDetectionOutput(); //new DetectionOutput();
}
*/

} // namespace Intersections

} // namespace Components

} // namespace Sofa
