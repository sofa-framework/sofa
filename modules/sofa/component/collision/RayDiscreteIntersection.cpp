/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/collision/RayDiscreteIntersection.inl>
#include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.inl>
#include <sofa/component/collision/DiscreteIntersection.inl>
#include <sofa/core/collision/Intersection.inl>
//#include <sofa/component/collision/ProximityIntersection.h>
#include <sofa/helper/proximity.h>
#include <iostream>
#include <algorithm>
#include <sofa/core/collision/IntersectorFactory.h>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;

SOFA_DECL_CLASS(RayDiscreteIntersection)

IntersectorCreator<DiscreteIntersection, RayDiscreteIntersection> RayDiscreteIntersectors("Ray");

RayDiscreteIntersection::RayDiscreteIntersection(DiscreteIntersection* object)
    : intersection(object)
{
    intersection->intersectors.add<RayModel, SphereModel,                     RayDiscreteIntersection>  (this);
    intersection->intersectors.add<RayModel, TriangleModel,                   RayDiscreteIntersection>  (this);
    intersection->intersectors.add<RayModel, TetrahedronModel,                RayDiscreteIntersection>  (this);
    intersection->intersectors.add<RayModel, RigidDistanceGridCollisionModel, RayDiscreteIntersection>  (this);
    intersection->intersectors.add<RayModel, FFDDistanceGridCollisionModel,   RayDiscreteIntersection>  (this);

    intersection->intersectors.ignore<RayModel, PointModel>();
    intersection->intersectors.ignore<RayModel, LineModel>();
}

bool RayDiscreteIntersection::testIntersection(Ray&, Triangle&)
{
    return true;
}

int RayDiscreteIntersection::computeIntersection(Ray& e1, Triangle& e2, OutputVector* contacts)
{
    Vector3 A = e2.p1();
    Vector3 AB = e2.p2()-A;
    Vector3 AC = e2.p3()-A;
    Vector3 P = e1.origin();
    Vector3 PQ = e1.direction();
    Matrix3 M, Minv;
    Vector3 right;
    for (int i=0; i<3; i++)
    {
        M[i][0] = AB[i];
        M[i][1] = AC[i];
        M[i][2] = -PQ[i];
        right[i] = P[i]-A[i];
    }
    if (!Minv.invert(M))
        return 0;
    Vector3 baryCoords = Minv * right;
    if (baryCoords[0] < 0 || baryCoords[1] < 0 || baryCoords[0]+baryCoords[1] > 1)
        return 0; // out of the triangle
    if (baryCoords[2] < 0 || baryCoords[2] > e1.l())
        return 0; // out of the line

    Vector3 X = P+PQ*baryCoords[2];

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    detection->point[0] = X;
    detection->point[1] = X;
    detection->normal = -e2.n();
    detection->value = 0;
    detection->elem.first = e1;
    detection->elem.second = e2;
    detection->id = e1.getIndex();
    return 1;
}

bool RayDiscreteIntersection::testIntersection(Ray&, Tetrahedron&)
{
    return true;
}

int RayDiscreteIntersection::computeIntersection(Ray& e1, Tetrahedron& e2, OutputVector* contacts)
{
    Vector3 P = e1.origin();
    Vector3 PQ = e1.direction();
    Vector3 b0 = e2.getBary(P);
    Vector3 bdir = e2.getDBary(PQ);
    //sout << "b0 = "<<b0<<" \tbdir = "<<bdir<<sendl;
    double l0 = 0;
    double l1 = e1.l();
    for (int c=0; c<3; ++c)
    {
        if (b0[c] < 0 && bdir[c] < 0) return 0; // no intersection
        //if (b0[c] > 1 && bdir[c] > 0) return 0; // no intersection
        if (bdir[c] < -1.0e-10)
        {
            double l = -b0[c]/bdir[c];
            if (l < l1) l1 = l;
        }
        else if (bdir[c] > 1.0e-10)
        {
            double l = -b0[c]/bdir[c];
            if (l > l0) l0 = l;
        }
    }
    // 4th plane : bx+by+bz = 1
    {
        double bd = bdir[0]+bdir[1]+bdir[2];
        if (bd > 1.0e-10)
        {
            double l = (1-(b0[0]+b0[1]+b0[2]))/bd;
            if (l < l1) l1 = l;
        }
        else if (bd < -1.0e-10)
        {
            double l = (1-(b0[0]+b0[1]+b0[2]))/bd;
            if (l > l0) l0 = l;
        }
    }
    if (l0 > l1) return 0; // empty intersection
    double l = l0; //(l0+l1)/2;
    Vector3 X = P+PQ*l;

    //sout << "tetra "<<e2.getIndex()<<": b0 = "<<b0<<" \tbdir = "<<bdir<<sendl;
    //sout << "l0 = "<<l0<<" \tl1 = "<<l1<<" \tX = "<<X<<" \tbX = "<<e2.getBary(X)<<" \t?=? "<<(b0+bdir*l)<<sendl;
    //sout << "b1 = "<<e2.getBary(e2.p1())<<" \nb2 = "<<e2.getBary(e2.p2())<<" \nb3 = "<<e2.getBary(e2.p3())<<" \nb4 = "<<e2.getBary(e2.p4())<<sendl;

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    detection->point[0] = X;
    detection->point[1] = X;
    PQ.normalize();
    detection->normal = PQ;
    detection->value = 0;
    detection->elem.first = e1;
    detection->elem.second = e2;
    detection->id = e1.getIndex();
    return 1;
}

bool RayDiscreteIntersection::testIntersection(Ray& /*e2*/, RigidDistanceGridCollisionElement& /*e1*/)
{
    return true;
}

int RayDiscreteIntersection::computeIntersection(Ray& e2, RigidDistanceGridCollisionElement& e1, OutputVector* contacts)
{
    Vector3 rayOrigin(e2.origin());
    Vector3 rayDirection(e2.direction());
    const double rayLength = e2.l();

    int nc = 0;
    DistanceGrid* grid1 = e1.getGrid();
    bool useXForm = e1.isTransformed();

    if (useXForm)
    {
        const Vector3& t1 = e1.getTranslation();
        const Matrix3& r1 = e1.getRotation();
        rayOrigin = r1.multTranspose(rayOrigin-t1);
        rayDirection = r1.multTranspose(rayDirection);
        // now ray infos are in grid1 space
    }

    double l0 = 0;
    double l1 = rayLength;
    Vector3 r0 = rayOrigin;
    Vector3 r1 = rayOrigin + rayDirection*l1;

    DistanceGrid::Coord bbmin = grid1->getBBMin(), bbmax = grid1->getBBMax();
    // clip along each axis
    for (int c=0; c<3 && l1>l0; c++)
    {
        if (rayDirection[c] > 0)
        {
            // test if the ray is inside
            if (r1[c] < bbmin[c] || r0[c] > bbmax[c])
            { l1 = 0; break; }
            if (r0[c] < bbmin[c])
            {
                // intersect with p[c] == bbmin[c] plane
                double l = (bbmin[c]-rayOrigin[c]) / rayDirection[c];
                if(l0 < l)
                {
                    l0 = l;
                    r0 = rayOrigin + rayDirection*l0;
                }
            }
            if (r1[c] > bbmax[c])
            {
                // intersect with p[c] == bbmax[c] plane
                double l = (bbmax[c]-rayOrigin[c]) / rayDirection[c];
                if(l1 > l)
                {
                    l1 = l;
                    r1 = rayOrigin + rayDirection*l1;
                }
            }
        }
        else
        {
            // test if the ray is inside
            if (r0[c] < bbmin[c] || r1[c] > bbmax[c])
            { l1 = 0; break; }
            if (r0[c] > bbmax[c])
            {
                // intersect with p[c] == bbmax[c] plane
                double l = (bbmax[c]-rayOrigin[c]) / rayDirection[c];
                if(l0 < l)
                {
                    l0 = l;
                    r0 = rayOrigin + rayDirection*l0;
                }
            }
            if (r1[c] < bbmin[c])
            {
                // intersect with p[c] == bbmin[c] plane
                double l = (bbmin[c]-rayOrigin[c]) / rayDirection[c];
                if(l1 > l)
                {
                    l1 = l;
                    r1 = rayOrigin + rayDirection*l1;
                }
            }
        }

    }

    if (l0 < l1)
    {
        // some part of the ray is inside the grid
        Vector3 p = rayOrigin + rayDirection*l0;
        double dist = grid1->interp(p);
        double epsilon = grid1->getCellWidth().norm()*0.1f;
        while (l0 < l1 && (dist > epsilon || dist < -epsilon))
        {
            l0 += dist;
            p = rayOrigin + rayDirection*l0;
            dist = grid1->interp(p);
            //sout << "p="<<p<<" dist="<<dist<<" l0="<<l0<<" l1="<<l1<<" epsilon="<<epsilon<<sendl;
        }
        if (dist < epsilon)
        {
            // intersection found

            contacts->resize(contacts->size()+1);
            DetectionOutput *detection = &*(contacts->end()-1);

            detection->point[0] = e2.origin() + e2.direction()*l0;
            detection->point[1] = p;
            detection->normal = e2.direction(); // normal in global space from p1's surface
            detection->value = dist;
            detection->elem.first = e2;
            detection->elem.second = e1;
            detection->id = e2.getIndex();
            ++nc;
        }
    }
    return nc;
}

bool RayDiscreteIntersection::testIntersection(Ray& /*e1*/, FFDDistanceGridCollisionElement& /*e2*/)
{
    return true;
}

int RayDiscreteIntersection::computeIntersection(Ray& e2, FFDDistanceGridCollisionElement& e1, OutputVector* contacts)
{
    Vector3 rayOrigin(e2.origin());
    Vector3 rayDirection(e2.direction());
    const double rayLength = e2.l();

    DistanceGrid* grid1 = e1.getGrid();
    FFDDistanceGridCollisionModel::DeformedCube& c1 = e1.getCollisionModel()->getDeformCube(e1.getIndex());

    // Center of the sphere
    const Vector3 center1 = c1.center;
    // Radius of the sphere
    const double radius1 = c1.radius;

    const Vector3 tmp = center1 - rayOrigin;
    double rayPos = tmp*rayDirection;
    const double dist2 = tmp.norm2() - (rayPos*rayPos);
    if (dist2 >= (radius1*radius1))
        return 0;

    double l0 = rayPos - sqrt(radius1*radius1 - dist2);
    double l1 = rayPos + sqrt(radius1*radius1 - dist2);
    if (l0 < 0) l0 = 0;
    if (l1 > rayLength) l1 = rayLength;
    if (l0 > l1) return 0; // outside of ray
    //const double dist = sqrt(dist2);
    //double epsilon = grid1->getCellWidth().norm()*0.1f;

    c1.updateFaces();
    DistanceGrid::Coord p1;
    const SReal cubesize = c1.invDP.norm();
    for(int i=0; i<100; i++)
    {
        rayPos = l0 + (l1-l0)*(i*0.01);
        p1 = rayOrigin + rayDirection*rayPos;
        // estimate the barycentric coordinates
        DistanceGrid::Coord b = c1.undeform0(p1);
        // refine the estimate until we are very close to the p2 or we are sure p2 cannot intersect with the object
        int iter;
        SReal err1 = 1000.0f;
        bool found = false;
        for(iter=0; iter<5; ++iter)
        {
            DistanceGrid::Coord pdeform = c1.deform(b);
            DistanceGrid::Coord diff = p1-pdeform;
            SReal err = diff.norm();
            //if (iter>3)
            //    sout << "Iter"<<iter<<": "<<err1<<" -> "<<err<<" b = "<<b<<" diff = "<<diff<<" d = "<<grid1->interp(c1.initpos(b))<<""<<sendl;
            SReal berr = err*cubesize; if (berr>0.5f) berr=0.5f;
            if (b[0] < -berr || b[0] > 1+berr
                || b[1] < -berr || b[1] > 1+berr
                || b[2] < -berr || b[2] > 1+berr)
                break; // far from the cube
            if (err < 0.001f)
            {
                // we found the corresponding point, but is is only valid if inside the current cube
                if (b[0] > -0.1f && b[0] < 1.1f
                    && b[1] > -0.1f && b[1] < 1.1f
                    && b[2] > -0.1f && b[2] < 1.1f)
                {
                    found = true;
                }
                break;
            }
            err1 = err;
            b += c1.undeformDir( b, diff );
        }
        if (found)
        {
            SReal d = grid1->interp(c1.initpos(b));
            if (d < 0)
            {
                // intersection found

                contacts->resize(contacts->size()+1);
                DetectionOutput *detection = &*(contacts->end()-1);

                detection->point[0] = e2.origin() + e2.direction()*rayPos;
                detection->point[1] = c1.initpos(b);
                detection->normal = e2.direction(); // normal in global space from p1's surface
                detection->value = d;
                detection->elem.first = e2;
                detection->elem.second = e1;
                detection->id = e2.getIndex();
                return 1;
            }
        }
        // else move along the ray
        //if (dot(Vector3(grid1->grad(c1.initpos(b))),rayDirection) < 0)
        //    rayPos += 0.5*d;
        //else
        //    rayPos -= 0.5*d;
    }
    return 0;
}


} // namespace collision

} // namespace component

} // namespace sofa

