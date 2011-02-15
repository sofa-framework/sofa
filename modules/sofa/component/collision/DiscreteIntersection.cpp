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
#include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.inl>
#include <sofa/component/collision/DiscreteIntersection.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/Intersection.inl>
//#include <sofa/component/collision/ProximityIntersection.h>
#include <sofa/helper/proximity.h>
#include <iostream>
#include <algorithm>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;

SOFA_DECL_CLASS(DiscreteIntersection)

int DiscreteIntersectionClass = core::RegisterObject("TODO-DiscreteIntersectionClass")
        .add< DiscreteIntersection >()
        ;


DiscreteIntersection::DiscreteIntersection()
{
    intersectors.add<CubeModel,       CubeModel,         DiscreteIntersection> (this);
    intersectors.add<SphereModel,     SphereModel,       DiscreteIntersection> (this);
    intersectors.add<SphereTreeModel, SphereTreeModel,   DiscreteIntersection> (this);
    intersectors.add<SphereTreeModel, CubeModel,         DiscreteIntersection>  (this);
    intersectors.add<SphereTreeModel, TriangleModel,     DiscreteIntersection>  (this);
    //intersectors.add<SphereTreeModel, SphereModel,       DiscreteIntersection>  (this);
    //intersectors.add<SphereModel,     TriangleModel,     DiscreteIntersection>  (this);
    intersectors.add<TriangleModel,     LineModel,       DiscreteIntersection>  (this);
    //intersectors.add<TriangleModel,   TriangleModel,     DiscreteIntersection> (this);
    intersectors.add<TetrahedronModel, PointModel,       DiscreteIntersection>  (this);
    intersectors.add<RigidDistanceGridCollisionModel, PointModel,                      DiscreteIntersection>  (this);
    intersectors.add<RigidDistanceGridCollisionModel, SphereModel,                     DiscreteIntersection>  (this);
    intersectors.add<RigidDistanceGridCollisionModel, TriangleModel,                   DiscreteIntersection>  (this);
    intersectors.add<RigidDistanceGridCollisionModel, RigidDistanceGridCollisionModel, DiscreteIntersection> (this);
    intersectors.add<FFDDistanceGridCollisionModel, PointModel,                        DiscreteIntersection>  (this);
    intersectors.add<FFDDistanceGridCollisionModel, SphereModel,                       DiscreteIntersection>  (this);
    intersectors.add<FFDDistanceGridCollisionModel, TriangleModel,                     DiscreteIntersection>  (this);
    intersectors.add<FFDDistanceGridCollisionModel,   RigidDistanceGridCollisionModel, DiscreteIntersection>  (this);
    intersectors.add<FFDDistanceGridCollisionModel,   FFDDistanceGridCollisionModel,   DiscreteIntersection> (this);

    intersectors.add<RayModel, SphereModel,                     DiscreteIntersection>  (this);
    intersectors.add<RayModel, SphereTreeModel,                 DiscreteIntersection>  (this);
    intersectors.ignore<RayModel, PointModel>();
    intersectors.ignore<RayModel, LineModel>();
    intersectors.add<RayModel, TriangleModel,                   DiscreteIntersection>  (this);
    intersectors.add<RayModel, TetrahedronModel,                DiscreteIntersection>  (this);
    intersectors.add<RayModel, RigidDistanceGridCollisionModel, DiscreteIntersection>  (this);
    intersectors.add<RayModel, FFDDistanceGridCollisionModel,   DiscreteIntersection>  (this);

}

/// Return the intersector class handling the given pair of collision models, or NULL if not supported.
ElementIntersector* DiscreteIntersection::findIntersector(core::CollisionModel* object1, core::CollisionModel* object2, bool& swapModels)
{
    return intersectors.get(object1, object2, swapModels);
}

bool DiscreteIntersection::testIntersection(Cube& cube1, Cube& cube2)
{
    const Vector3& minVect1 = cube1.minVect();
    const Vector3& minVect2 = cube2.minVect();
    const Vector3& maxVect1 = cube1.maxVect();
    const Vector3& maxVect2 = cube2.maxVect();

    for (int i=0; i<3; i++)
    {
        if (minVect1[i] > maxVect2[i] || minVect2[i] > maxVect1[i])
            return false;
    }

    //sout << "Box <"<<minVect1[0]<<","<<minVect1[1]<<","<<minVect1[2]<<">-<"<<maxVect1[0]<<","<<maxVect1[1]<<","<<maxVect1[2]
    //  <<"> collide with Box "<<minVect2[0]<<","<<minVect2[1]<<","<<minVect2[2]<<">-<"<<maxVect2[0]<<","<<maxVect2[1]<<","<<maxVect2[2]<<">"<<sendl;
    return true;
}

//bool DiscreteIntersection::testIntersection(Triangle& t1, Triangle& t2)
//{
//	sout<<"Collision between Triangle - Triangle"<<sendl;
//	return false;
//}

int DiscreteIntersection::computeIntersection(Cube&, Cube&, OutputVector*)
{
    return 0; /// \todo
}

//int DiscreteIntersection::computeIntersection(Triangle&, Triangle&, OutputVector*)
//{
//	sout<<"Distance correction between Triangle - Triangle"<<sendl;
//	return 0;
//}

bool DiscreteIntersection::testIntersection(Triangle&, Line&)
{
    return true;
}

int DiscreteIntersection::computeIntersection(Triangle& e1, Line& e2, OutputVector* contacts)
{
    Vector3 A = e1.p1();
    Vector3 AB = e1.p2()-A;
    Vector3 AC = e1.p3()-A;
    Vector3 P = e2.p1();
    Vector3 PQ = e2.p2()-P;
    Matrix3 M, Minv;
    Vector3 right;
    for (int i=0; i<3; i++)
    {
        M[i][0] = AB[i];
        M[i][1] = AC[i];
        M[i][2] = -PQ[i];
        right[i] = P[i]-A[i];
    }
    //sout << "M="<<M<<sendl;
    if (!Minv.invert(M))
        return 0;
    Vector3 baryCoords = Minv * right;
    if (baryCoords[0] < 0 || baryCoords[1] < 0 || baryCoords[0]+baryCoords[1] > 1)
        return 0; // out of the triangle
    if (baryCoords[2] < 0 || baryCoords[2] > 1)
        return 0; // out of the line

    Vector3 X = P+PQ*baryCoords[2];

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    detection->point[0] = X;
    detection->point[1] = X;
    detection->normal = e1.n();
    detection->value = 0;
    detection->elem.first = e1;
    detection->elem.second = e2;
    detection->id = e2.getIndex();
    return 1;
}

bool DiscreteIntersection::testIntersection(Ray&, Triangle&)
{
    return true;
}

int DiscreteIntersection::computeIntersection(Ray& e1, Triangle& e2, OutputVector* contacts)
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

bool DiscreteIntersection::testIntersection(Ray&, Tetrahedron&)
{
    return true;
}

int DiscreteIntersection::computeIntersection(Ray& e1, Tetrahedron& e2, OutputVector* contacts)
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

bool DiscreteIntersection::testIntersection(Tetrahedron&, Point&)
{
    return true;
}

int DiscreteIntersection::computeIntersection(Tetrahedron& e1, Point& e2, OutputVector* contacts)
{
    Vector3 n = e2.n();
    if (n == Vector3()) return 0; // no normal on point -> either an internal points or normal is not available

    if (e1.getCollisionModel()->getMechanicalState() == e2.getCollisionModel()->getMechanicalState())
    {
        // self-collisions: make sure the point is not one of the vertices of the tetrahedron
        int i = e2.getIndex();
        if (i == e1.p1Index() || i == e1.p2Index() || i == e1.p3Index() || i == e1.p4Index())
            return 0;
    }

    Vector3 P = e2.p();
    Vector3 b0 = e1.getBary(P);
    if (b0[0] < 0 || b0[1] < 0 || b0[2] < 0 || (b0[0]+b0[1]+b0[2]) > 1)
        return 1; // out of tetrahedron

    // Find the point on the surface of the tetrahedron in the direction of -n
    Vector3 bdir = e1.getDBary(-n);
    //sout << "b0 = "<<b0<<" \tbdir = "<<bdir<<sendl;
    double l1 = 1.0e10;
    for (int c=0; c<3; ++c)
    {
        if (bdir[c] < -1.0e-10)
        {
            double l = -b0[c]/bdir[c];
            if (l < l1) l1 = l;
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
    }
    if (l1 >= 1.0e9) l1 = 0;
    double l = l1;
    Vector3 X = P-n*l;

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    detection->point[0] = X;
    detection->point[1] = P;
    detection->normal = -n;
    detection->value = -l;
    detection->elem.first = e1;
    detection->elem.second = e2;
    detection->id = e2.getIndex();
    return 1;
}

bool DiscreteIntersection::testIntersection(RigidDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&)
{
    return true;
}

//#define DEBUG_XFORM

int DiscreteIntersection::computeIntersection(RigidDistanceGridCollisionElement& e1, RigidDistanceGridCollisionElement& e2, OutputVector* contacts)
{
    int nc = 0;
    DistanceGrid* grid1 = e1.getGrid();
    DistanceGrid* grid2 = e2.getGrid();
    bool useXForm = e1.isTransformed() || e2.isTransformed();
    const Vector3& t1 = e1.getTranslation();
    const Matrix3& r1 = e1.getRotation();
    const Vector3& t2 = e2.getTranslation();
    const Matrix3& r2 = e2.getRotation();

    const double d0 = e1.getProximity() + e2.getProximity() + (this->getContactDistance() == 0.0 ? 0.001 : this->getContactDistance());
    //const SReal margin = 0.001f + (SReal)d0;
    const SReal margin = (SReal)((e1.getProximity() + e2.getProximity() + (this->getAlarmDistance() == 0.0 ? 0.001 : this->getAlarmDistance()))/2);

    // transform from grid1 to grid2
    Vec3f translation;
    Mat3x3f rotation;

    if (useXForm)
    {
        // p = t1+r1*p1 = t2+r2*p2
        // r2*p2 = t1-t2+r1*p1
        // p2 = r2t*(t1-p2) + r2t*r1*p1
        translation = r2.multTranspose(t1-t2);
        rotation = r2.multTranspose ( r1 );
    }
    else rotation.identity();

    // For the cube-cube case, we need to detect cases where cubes are stacked
    // One way is to find if a pair of faces exists that are nearly parallel and
    // that are near each other
    // if such pair is found, the corresponding face will be stored in these variables

    enum { FACE_NONE=-1,FACE_XNEG=0,FACE_XPOS,FACE_YNEG,FACE_YPOS,FACE_ZNEG,FACE_ZPOS };
    int face_e1 = FACE_NONE;
    int face_e2 = FACE_NONE;

    if (grid2->isCube() && grid1->isCube())
    {
        const SReal cubeDim1 = grid1->getCubeDim();
        const SReal cubeDim2 = grid2->getCubeDim();
        // current distance found
        // we allow only 10% penetration
        SReal dist = (SReal)((cubeDim1 + cubeDim2) * 0.1);
        // a nearly perpendicular pair would be visible by an entry close to 1 in the rotation matrix
        for (int f2 = 0; f2 < 3; f2++)
        {
            for (int f1 = 0; f1 < 3; f1++)
            {
                if (rotation[f2][f1] < -0.99 || rotation[f2][f1] > 0.99)
                {
                    // found a match
                    // translation is the position of cube1 center in cube2 space
                    // so the pair of faces are close if |translation[f2]| is close dim1+dim2
                    SReal d = rabs(rabs(translation[f2])-(cubeDim1+cubeDim2));
                    // we should favor normals that are perpendicular to the relative velocity
                    // however we don't have this information currently, so for now we favor the horizontal face
                    if (rabs(r2[f2][2]) > 0.99 && d < (cubeDim1 + cubeDim2) * 0.1) d = 0;
                    if (d < dist)
                    {
                        dist = d;
                        if (translation[f2] > 0)
                        {
                            // positive side on cube 2
                            face_e2 = 2*f2+1;
                            if (rotation[f2][f1] > 0)
                            {
                                // cubes have same axis orientation -> negative side on cube 1
                                face_e1 = 2*f1;
                            }
                            else
                            {
                                // cubes have same opposite orientation -> positive side on cube 1
                                face_e1 = 2*f1+1;
                            }
                        }
                        else
                        {
                            // negative side on cube 2
                            face_e2 = 2*f2;
                            if (rotation[f2][f1] > 0)
                            {
                                // cubes have same axis orientation -> positive side on cube 1
                                face_e1 = 2*f1+1;
                            }
                            else
                            {
                                // cubes have same opposite orientation -> negative side on cube 1
                                face_e1 = 2*f1;
                            }
                        }
                    }
                }
            }
        }
    }

    // first points of e1 against distance field of e2
    const DistanceGrid::VecCoord& x1 = grid1->meshPts;
    if (!x1.empty() && e1.getCollisionModel()->usePoints.getValue())
    {
        if (grid2->isCube() && grid1->isCube())
        {
            const SReal cubeDim2 = grid2->getCubeDim();
            const SReal cubeDim2Margin = cubeDim2+margin;

            if (face_e2 != FACE_NONE)
            {
                // stacked cubes
                DistanceGrid::Coord normal;
                normal[face_e2/2] = (face_e2&1)?1.0f:-1.0f;
                Vector3 gnormal = r2 * -normal;
                // special case: if normal in global frame is nearly vertical or horizontal, make it so
                if (gnormal[0] < -0.99f) gnormal = Vector3(-1.0f, 0.0f,  0.0f);
                else if (gnormal[0] >  0.99f) gnormal = Vector3( 1.0f, 0.0f,  0.0f);
                if (gnormal[1] < -0.99f) gnormal = Vector3( 0.0f,-1.0f,  0.0f);
                else if (gnormal[1] >  0.99f) gnormal = Vector3( 0.0f, 1.0f,  0.0f);
                if (gnormal[2] < -0.99f) gnormal = Vector3( 0.0f, 0.0f, -1.0f);
                else if (gnormal[2] >  0.99f) gnormal = Vector3( 0.0f, 0.0f,  1.0f);
                for (unsigned int i=0; i<x1.size(); i++)
                {
                    DistanceGrid::Coord p1 = x1[i];
                    DistanceGrid::Coord p2 = translation + rotation*p1;
                    if (p2[0] < -cubeDim2Margin || p2[0] > cubeDim2Margin ||
                        p2[1] < -cubeDim2Margin || p2[1] > cubeDim2Margin ||
                        p2[2] < -cubeDim2Margin || p2[2] > cubeDim2Margin)
                        continue;
                    double d = p2*normal - cubeDim2;

                    p2 -= normal * d; // push p2 to the surface

                    contacts->resize(contacts->size()+1);
                    DetectionOutput *detection = &*(contacts->end()-1);

                    detection->point[0] = Vector3(p1);
                    detection->point[1] = Vector3(p2);
                    detection->normal = gnormal;
                    detection->value = d - d0;
                    detection->elem.first = e1;
                    detection->elem.second = e2;
                    detection->id = i;
                    ++nc;
                }
            }
            else
            {
                // general case
                for (unsigned int i=0; i<x1.size(); i++)
                {
                    DistanceGrid::Coord p1 = x1[i];
                    DistanceGrid::Coord p2 = translation + rotation*p1;
                    if (p2[0] < -cubeDim2Margin || p2[0] > cubeDim2Margin ||
                        p2[1] < -cubeDim2Margin || p2[1] > cubeDim2Margin ||
                        p2[2] < -cubeDim2Margin || p2[2] > cubeDim2Margin)
                        continue;
                    //double d = p2*normal - cubeDim2;

                    DistanceGrid::Coord normal;
                    normal[0] = rabs(p2[0]) - cubeDim2;
                    normal[1] = rabs(p2[1]) - cubeDim2;
                    normal[2] = rabs(p2[2]) - cubeDim2;

                    SReal d;
                    // find the smallest penetration
                    int axis;
                    if (normal[0] > normal[1])
                        if (normal[0] > normal[2]) axis = 0;
                        else                       axis = 2;
                    else if (normal[1] > normal[2]) axis = 1;
                    else                       axis = 2;

                    SReal sign = (p2[axis]<0)?-1.0f:1.0f;
                    d = normal[axis];
                    p2[axis] = sign*cubeDim2;
                    Vector3 gnormal = r2.col(axis) * -sign;

                    //p2 -= normal * d; // push p2 to the surface

                    contacts->resize(contacts->size()+1);
                    DetectionOutput *detection = &*(contacts->end()-1);

                    detection->point[0] = Vector3(p1);
                    detection->point[1] = Vector3(p2);
                    detection->normal = gnormal;
                    detection->value = d - d0;
                    detection->elem.first = e1;
                    detection->elem.second = e2;
                    detection->id = i;
                    ++nc;
                }
            }
        }
        else
        {
            for (unsigned int i=0; i<x1.size(); i++)
            {
                DistanceGrid::Coord p1 = x1[i];
                Vector3 n1 = grid1->grad(p1); // note that there are some redundant computations between interp() and grad()
                n1.normalize();
                DistanceGrid::Coord p2 = translation + rotation*(p1 + n1*margin);
#ifdef DEBUG_XFORM
                DistanceGrid::Coord p1b = rotation.multTranspose(p2-translation);
                DistanceGrid::Coord gp1 = t1+r1*p1;
                DistanceGrid::Coord gp2 = t2+r2*p2;
                if ((p1b-p1).norm2() > 0.0001f)
                    serr << "ERROR1a: " << p1 << " -> " << p2 << " -> " << p1b << sendl;
                if ((gp1-gp2).norm2() > 0.0001f)
                    serr << "ERROR1b: " << p1 << " -> " << gp1 << "    " << p2 << " -> " << gp2 << sendl;
#endif

                if (!grid2->inBBox( p2 /*, margin*/ )) continue;
                if (!grid2->inGrid( p2 ))
                {
                    serr << "WARNING: margin less than "<<margin<<" in DistanceGrid "<<e2.getCollisionModel()->getName()<<sendl;
                    continue;
                }

                SReal d = grid2->interp(p2);
                if (d >= 0 /* margin */ ) continue;

                Vector3 grad = grid2->grad(p2); // note that there are some redundant computations between interp() and grad()
                grad.normalize();

                //p2 -= grad * d; // push p2 back to the surface

                contacts->resize(contacts->size()+1);
                DetectionOutput *detection = &*(contacts->end()-1);

                detection->point[0] = Vector3(p1);
                detection->point[1] = Vector3(p2) - grad * d;
                detection->normal = r2 * -grad; // normal in global space from p1's surface
                detection->value = d + margin - d0;
                detection->elem.first = e1;
                detection->elem.second = e2;
                detection->id = i;
                ++nc;
            }
        }
    }

    // then points of e2 against distance field of e1
    const DistanceGrid::VecCoord& x2 = grid2->meshPts;
    const int i0 = x1.size();
    if (!x2.empty() && e2.getCollisionModel()->usePoints.getValue())
    {
        if (grid1->isCube() && grid2->isCube())
        {
            const SReal cubeDim1 = grid1->getCubeDim();
            const SReal cubeDim1Margin = cubeDim1+margin;

            if (face_e1 != FACE_NONE)
            {
                // stacked cubes
                DistanceGrid::Coord normal;
                normal[face_e1/2] = (face_e1&1)?1.0f:-1.0f;
                Vector3 gnormal = r1 * normal;
                // special case: if normal in global frame is nearly vertical or horizontal, make it so
                if (gnormal[0] < -0.99f) gnormal = Vector3(-1.0f, 0.0f,  0.0f);
                else if (gnormal[0] >  0.99f) gnormal = Vector3( 1.0f, 0.0f,  0.0f);
                if (gnormal[1] < -0.99f) gnormal = Vector3( 0.0f,-1.0f,  0.0f);
                else if (gnormal[1] >  0.99f) gnormal = Vector3( 0.0f, 1.0f,  0.0f);
                if (gnormal[2] < -0.99f) gnormal = Vector3( 0.0f, 0.0f, -1.0f);
                else if (gnormal[2] >  0.99f) gnormal = Vector3( 0.0f, 0.0f,  1.0f);
                for (unsigned int i=0; i<x2.size(); i++)
                {
                    DistanceGrid::Coord p2 = x2[i];
                    DistanceGrid::Coord p1 = rotation.multTranspose(p2-translation);
                    if (p1[0] < -cubeDim1Margin || p1[0] > cubeDim1Margin ||
                        p1[1] < -cubeDim1Margin || p1[1] > cubeDim1Margin ||
                        p1[2] < -cubeDim1Margin || p1[2] > cubeDim1Margin)
                        continue;
                    double d = p1*normal - cubeDim1;

                    p1 -= normal * d; // push p2 to the surface

                    contacts->resize(contacts->size()+1);
                    DetectionOutput *detection = &*(contacts->end()-1);

                    detection->point[0] = Vector3(p1);
                    detection->point[1] = Vector3(p2);
                    detection->normal = gnormal;
                    detection->value = d - d0;
                    detection->elem.first = e1;
                    detection->elem.second = e2;
                    detection->id = i + i0;
                    ++nc;
                }
            }
            else
            {
                // general case
                for (unsigned int i=0; i<x2.size(); i++)
                {
                    DistanceGrid::Coord p2 = x2[i];
                    DistanceGrid::Coord p1 = rotation.multTranspose(p2-translation);
                    if (p1[0] < -cubeDim1Margin || p1[0] > cubeDim1Margin ||
                        p1[1] < -cubeDim1Margin || p1[1] > cubeDim1Margin ||
                        p1[2] < -cubeDim1Margin || p1[2] > cubeDim1Margin)
                        continue;

                    DistanceGrid::Coord normal;
                    normal[0] = rabs(p1[0]) - cubeDim1;
                    normal[1] = rabs(p1[1]) - cubeDim1;
                    normal[2] = rabs(p1[2]) - cubeDim1;

                    SReal d;
                    // find the smallest penetration
                    int axis;
                    if (normal[0] > normal[1])
                        if (normal[0] > normal[2]) axis = 0;
                        else                       axis = 2;
                    else if (normal[1] > normal[2]) axis = 1;
                    else                       axis = 2;

                    SReal sign = (p1[axis]<0)?-1.0f:1.0f;
                    d = normal[axis];
                    p1[axis] = sign*cubeDim1;
                    Vector3 gnormal = r1.col(axis) * sign;

                    contacts->resize(contacts->size()+1);
                    DetectionOutput *detection = &*(contacts->end()-1);

                    detection->point[0] = Vector3(p1);
                    detection->point[1] = Vector3(p2);
                    detection->normal = gnormal;
                    detection->value = d - d0;
                    detection->elem.first = e1;
                    detection->elem.second = e2;
                    detection->id = i + i0;
                    ++nc;
                }
            }
#if 0
#if 0
            // -rotationT*translation is the position of cube2 center in cube1 space
            // we use its largest component as the dominant contact face normal
            /// \TODO use the relative velocity as an additionnal factor
            Vector3 normal = rotation.multTranspose(-translation);
            //normal[2] *= 1.1f; // we like Z contact better ;)
            if (rabs(normal[0]) > rabs(normal[1]))
            {
                if (rabs(normal[0]) > rabs(normal[2]))
                    normal = Vector3(normal[0]>0.0f?1.0f:-1.0f,0.0f,0.0f);
                else
                    normal = Vector3(0.0f,0.0f,normal[2]>0.0f?1.0f:-1.0f);
            }
            else
            {
                if (rabs(normal[1]) > rabs(normal[2]))
                    normal = Vector3(0.0f,normal[1]>0.0f?1.0f:-1.0f,0.0f);
                else
                    normal = Vector3(0.0f,0.0f,normal[2]>0.0f?1.0f:-1.0f);
            }

            Vector3 gnormal = r1 * normal; // normal in global space from p1's surface
            // special case: if normal in global frame is nearly vertical, make it so
            if (gnormal[2] < -0.99f) gnormal = Vector3(0.0f, 0.0f, -1.0f);
            else if (gnormal[2] >  0.99f) gnormal = Vector3(0.0f, 0.0f,  1.0f);
#endif
            Vector3 gnormal[3]; // X/Y/Z normals from p1 in global space
            for (int i=0; i<3; i++)
            {
                gnormal[i] = r1.col(i);
                // special case: if normal in global frame is nearly vertical or horizontal, make it so
                if (gnormal[i][0] < -0.99f) gnormal[i] = Vector3(-1.0f, 0.0f,  0.0f);
                else if (gnormal[i][0] >  0.99f) gnormal[i] = Vector3( 1.0f, 0.0f,  0.0f);
                if (gnormal[i][1] < -0.99f) gnormal[i] = Vector3( 0.0f,-1.0f,  0.0f);
                else if (gnormal[i][1] >  0.99f) gnormal[i] = Vector3( 0.0f, 1.0f,  0.0f);
                if (gnormal[i][2] < -0.99f) gnormal[i] = Vector3( 0.0f, 0.0f, -1.0f);
                else if (gnormal[i][2] >  0.99f) gnormal[i] = Vector3( 0.0f, 0.0f,  1.0f);
            }
            for (unsigned int i=0; i<x2.size(); i++)
            {
                DistanceGrid::Coord p2 = x2[i];
                DistanceGrid::Coord p1 = rotation.multTranspose(p2-translation);
                if (p1[0] < -cubeDim1Margin || p1[0] > cubeDim1Margin ||
                    p1[1] < -cubeDim1Margin || p1[1] > cubeDim1Margin ||
                    p1[2] < -cubeDim1Margin || p1[2] > cubeDim1Margin)
                    continue;

                DistanceGrid::Coord p2normal = rotation.multTranspose(grid2->grad(p2)); // normal of p2, in p1's space

                DistanceGrid::Coord p1normal;

                p1normal[0] = (cubeDim1Margin - rabs(p1[0]))/(0.000001+rabs(p2normal[0]));
                p1normal[1] = (cubeDim1Margin - rabs(p1[1]))/(0.000001+rabs(p2normal[1]));
                p1normal[2] = (cubeDim1Margin - rabs(p1[2]))/(0.000001+rabs(p2normal[2]));

                SReal d;
                Vector3 normal;
                // find the smallest penetration
                int axis;
                //if (p1normal[0]*p2normal[0] < p1normal[1]*p2normal[1])
                if (p1normal[0] < p1normal[1])
                {
                    if (p1normal[0] < p1normal[2])
                        axis = 0;
                    else
                        axis = 2;
                }
                else
                {
                    if (p1normal[1] < p1normal[2])
                        axis = 1;
                    else
                        axis = 2;
                }
                if (p1[axis]<0)
                {
                    d = -cubeDim1 - p1[axis]; // p2normal[axis];
                    p1[axis] = -cubeDim1;
                    normal = -gnormal[axis];
                }
                else
                {
                    d = p1[axis] - cubeDim1; // -p2normal[axis];
                    p1[axis] = cubeDim1;
                    normal = gnormal[axis];
                }


                contacts->resize(contacts->size()+1);
                DetectionOutput *detection = &*(contacts->end()-1);

                detection->point[0] = Vector3(p1); // - normal * d;
                detection->point[1] = Vector3(p2);
                detection->normal = normal;
                detection->value = d - d0;
                detection->elem.first = e1;
                detection->elem.second = e2;
                detection->id = i0+i;
                ++nc;
            }
#endif
        }
        else
        {
            for (unsigned int i=0; i<x2.size(); i++)
            {
                DistanceGrid::Coord p2 = x2[i];
                Vector3 n2 = grid2->grad(p2); // note that there are some redundant computations between interp() and grad()
                n2.normalize();

                DistanceGrid::Coord p1 = rotation.multTranspose(p2 + n2*margin - translation);
#ifdef DEBUG_XFORM
                DistanceGrid::Coord p2b = translation + rotation*p1;
                DistanceGrid::Coord gp1 = t1+r1*p1;
                DistanceGrid::Coord gp2 = t2+r2*p2;
                if ((p2b-p2).norm2() > 0.0001f)
                    serr << "ERROR2a: " << p2 << " -> " << p1 << " -> " << p2b << sendl;
                else if ((gp1-gp2).norm2() > 0.0001f)
                    serr << "ERROR2b: " << p1 << " -> " << gp1 << "    " << p2 << " -> " << gp2 << sendl;
#endif

                if (!grid1->inBBox( p1 /*, margin*/ )) continue;
                if (!grid1->inGrid( p1 ))
                {
                    serr << "WARNING: margin less than "<<margin<<" in DistanceGrid "<<e1.getCollisionModel()->getName()<<sendl;
                    continue;
                }

                SReal d = grid1->interp(p1);
                if (d >= 0 /* margin */ ) continue;

                Vector3 grad = grid1->grad(p1); // note that there are some redundant computations between interp() and grad()
                grad.normalize();

                //p1 -= grad * d; // push p1 back to the surface

                contacts->resize(contacts->size()+1);
                DetectionOutput *detection = &*(contacts->end()-1);

                detection->point[0] = Vector3(p1) - grad * d;
                detection->point[1] = Vector3(p2);
                detection->normal = r1 * grad; // normal in global space from p1's surface
                detection->value = d + margin - d0;
                detection->elem.first = e1;
                detection->elem.second = e2;
                detection->id = i0+i;
                ++nc;
            }
        }
    }
    return nc;
}

bool DiscreteIntersection::testIntersection(RigidDistanceGridCollisionElement&, Point&)
{
    return true;
}

int DiscreteIntersection::computeIntersection(RigidDistanceGridCollisionElement& e1, Point& e2, OutputVector* contacts)
{
    DistanceGrid* grid1 = e1.getGrid();
    bool useXForm = e1.isTransformed();
    const Vector3& t1 = e1.getTranslation();
    const Matrix3& r1 = e1.getRotation();
    const bool flipped = e1.isFlipped();

    const double d0 = e1.getProximity() + e2.getProximity() + getContactDistance();
    const SReal margin = 0.001f + (SReal)d0;


    Vector3 p2 = e2.p();
    DistanceGrid::Coord p1;

    if (useXForm)
    {
        p1 = r1.multTranspose(p2-t1);
    }
    else p1 = p2;

    if (flipped)
    {
        if (!grid1->inGrid( p1 )) return 0;
    }
    else
    {
        if (!grid1->inBBox( p1, margin )) return 0;
        if (!grid1->inGrid( p1 ))
        {
            serr << "WARNING: margin less than "<<margin<<" in DistanceGrid "<<e1.getCollisionModel()->getName()<<sendl;
            return 0;
        }
    }

    SReal d = grid1->interp(p1);
    if (flipped) d = -d;
    if (d >= margin) return 0;

    Vector3 grad = grid1->grad(p1); // note that there are some redundant computations between interp() and grad()
    if (flipped) grad = -grad;
    grad.normalize();

    //p1 -= grad * d; // push p1 back to the surface

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

    detection->point[0] = Vector3(p1) - grad * d;
    detection->point[1] = Vector3(p2);
    detection->normal = (useXForm) ? r1 * grad : grad; // normal in global space from p1's surface
    detection->value = d - d0;
    detection->elem.first = e1;
    detection->elem.second = e2;
    detection->id = e2.getIndex();
    return 1;
}

bool DiscreteIntersection::testIntersection(RigidDistanceGridCollisionElement&, Triangle&)
{
    return true;
}

int DiscreteIntersection::computeIntersection(RigidDistanceGridCollisionElement& e1, Triangle& e2, OutputVector* contacts)
{
    const int f2 = e2.flags();
    if (!(f2&TriangleModel::FLAG_POINTS)) return 0; // no points associated with this triangle
    DistanceGrid* grid1 = e1.getGrid();
    bool useXForm = e1.isTransformed();
    const Vector3& t1 = e1.getTranslation();
    const Matrix3& r1 = e1.getRotation();

    const double d0 = e1.getProximity() + e2.getProximity() + getContactDistance();
    const SReal margin = 0.001f + (SReal)d0;

    if (f2&TriangleModel::FLAG_P1)
    {
        Vector3 p2 = e2.p1();
        DistanceGrid::Coord p1;

        if (useXForm)
        {
            p1 = r1.multTranspose(p2-t1);
        }
        else p1 = p2;

        if (grid1->inBBox( p1, margin ))
        {
            if (!grid1->inGrid( p1 ))
            {
                serr << "WARNING: margin less than "<<margin<<" in DistanceGrid "<<e1.getCollisionModel()->getName()<<sendl;
            }
            else
            {
                SReal d = grid1->interp(p1);
                if (d >= margin) return 0;

                Vector3 grad = grid1->grad(p1); // note that there are some redundant computations between interp() and grad()
                grad.normalize();

                //p1 -= grad * d; // push p1 back to the surface

                contacts->resize(contacts->size()+1);
                DetectionOutput *detection = &*(contacts->end()-1);

                detection->point[0] = Vector3(p1) - grad * d;
                detection->point[1] = Vector3(p2);
                detection->normal = (useXForm) ? r1 * grad : grad; // normal in global space from p1's surface
                detection->value = d - d0;
                detection->elem.first = e1;
                detection->elem.second = e2;
                detection->id = e2.getIndex()*3+0;
            }
        }
    }

    if (f2&TriangleModel::FLAG_P2)
    {
        Vector3 p2 = e2.p2();
        DistanceGrid::Coord p1;

        if (useXForm)
        {
            p1 = r1.multTranspose(p2-t1);
        }
        else p1 = p2;

        if (grid1->inBBox( p1, margin ))
        {
            if (!grid1->inGrid( p1 ))
            {
                serr << "WARNING: margin less than "<<margin<<" in DistanceGrid "<<e1.getCollisionModel()->getName()<<sendl;
            }
            else
            {
                SReal d = grid1->interp(p1);
                if (d >= margin) return 0;

                Vector3 grad = grid1->grad(p1); // note that there are some redundant computations between interp() and grad()
                grad.normalize();

                //p1 -= grad * d; // push p1 back to the surface

                contacts->resize(contacts->size()+1);
                DetectionOutput *detection = &*(contacts->end()-1);

                detection->point[0] = Vector3(p1) - grad * d;
                detection->point[1] = Vector3(p2);
                detection->normal = (useXForm) ? r1 * grad : grad; // normal in global space from p1's surface
                detection->value = d - d0;
                detection->elem.first = e1;
                detection->elem.second = e2;
                detection->id = e2.getIndex()*3+1;
            }
        }
    }

    if (f2&TriangleModel::FLAG_P3)
    {
        Vector3 p2 = e2.p3();
        DistanceGrid::Coord p1;

        if (useXForm)
        {
            p1 = r1.multTranspose(p2-t1);
        }
        else p1 = p2;

        if (grid1->inBBox( p1, margin ))
        {
            if (!grid1->inGrid( p1 ))
            {
                serr << "WARNING: margin less than "<<margin<<" in DistanceGrid "<<e1.getCollisionModel()->getName()<<sendl;
            }
            else
            {
                SReal d = grid1->interp(p1);
                if (d >= margin) return 0;

                Vector3 grad = grid1->grad(p1); // note that there are some redundant computations between interp() and grad()
                grad.normalize();

                //p1 -= grad * d; // push p1 back to the surface

                contacts->resize(contacts->size()+1);
                DetectionOutput *detection = &*(contacts->end()-1);

                detection->point[0] = Vector3(p1) - grad * d;
                detection->point[1] = Vector3(p2);
                detection->normal = (useXForm) ? r1 * grad : grad; // normal in global space from p1's surface
                detection->value = d - d0;
                detection->elem.first = e1;
                detection->elem.second = e2;
                detection->id = e2.getIndex()*3+2;
            }
        }
    }
    return 1;
}

bool DiscreteIntersection::testIntersection(Ray& /*e2*/, RigidDistanceGridCollisionElement& /*e1*/)
{
    return true;
}

int DiscreteIntersection::computeIntersection(Ray& e2, RigidDistanceGridCollisionElement& e1, OutputVector* contacts)
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

bool DiscreteIntersection::testIntersection(FFDDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&)
{
    return true;
}

int DiscreteIntersection::computeIntersection(FFDDistanceGridCollisionElement& e1, RigidDistanceGridCollisionElement& e2, OutputVector* contacts)
{
    int nc = 0;
    DistanceGrid* grid1 = e1.getGrid();
    DistanceGrid* grid2 = e2.getGrid();
    FFDDistanceGridCollisionModel::DeformedCube& c1 = e1.getCollisionModel()->getDeformCube(e1.getIndex());
    bool useXForm = e2.isTransformed();
    //const Vector3& t1 = e1.getTranslation();
    //const Matrix3& r1 = e1.getRotation();
    const Vector3& t2 = e2.getTranslation();
    const Matrix3& r2 = e2.getRotation();

    const double d0 = e1.getProximity() + e2.getProximity() + (this->getContactDistance() == 0.0 ? 0.001 : this->getContactDistance());
    //const SReal margin = 0.001f + (SReal)d0;
    const SReal margin = (SReal)((e1.getProximity() + e2.getProximity() + (this->getAlarmDistance() == 0.0 ? 0.001 : this->getAlarmDistance()))/2);
    //std::cout << "margin="<<margin<<std::endl;
    const bool singleContact = e1.getCollisionModel()->singleContact.getValue();

    // transform from grid1 to grid2
    Vec3f translation;
    Mat3x3f rotation;

    if (useXForm)
    {
        translation = r2.multTranspose(-t2);
        rotation = r2; rotation.transpose();
    }
    else rotation.identity();

    const DistanceGrid::Coord center2 = translation + rotation*c1.center;
    const SReal radius2 = c1.radius*c1.radius;
    const DistanceGrid::VecCoord& x2 = grid2->meshPts;
    const int i0 = x2.size();
    // first points of e1 against distance field of e2
    if (e1.getCollisionModel()->usePoints.getValue()) // && !e2.getCollisionModel()->usePoints.getValue())
    {
        if (grid2->inBBox( center2, margin + c1.radius ))
        {
            c1.updatePoints();
            const sofa::helper::vector<DistanceGrid::Coord>& x1 = c1.deformedPoints;
            const sofa::helper::vector<DistanceGrid::Coord>& n1 = c1.deformedNormals;
            bool first = true;
            for (unsigned int i=0; i<x1.size(); i++)
            {
                DistanceGrid::Coord p1 = x1[i] + n1[i]*margin;
                DistanceGrid::Coord p2 = translation + rotation*p1;

                if (!grid2->inBBox( p2, margin )) continue;
                if (!grid2->inGrid( p2 ))
                {
                    serr << "WARNING: margin less than "<<margin<<" in DistanceGrid "<<e2.getCollisionModel()->getName()<<sendl;
                    continue;
                }

                SReal d = grid2->interp(p2);
                if (d >= margin) continue;

                Vector3 grad = grid2->grad(p2); // note that there are some redundant computations between interp() and grad()
                grad.normalize();

                //p2 -= grad * d; // push p2 back to the surface
                if (!singleContact || first)
                    contacts->resize(contacts->size()+1);
                DetectionOutput *detection = &*(contacts->end()-1);
                double value = d + margin - d0;
                //std::cout << "value = d + margin - d0 = " << d << " + " << margin << " - " << d0 << " = " << value << std::endl;
                if (!singleContact || first || (value < detection->value))
                {
                    detection->point[0] = grid1->meshPts[c1.points[i].index];
                    detection->point[1] = Vector3(p2) - grad * d;
                    detection->normal = r2 * -grad; // normal in global space from p1's surface
                    detection->value = value;
                    detection->elem.first = e1;
                    detection->elem.second = e2;
                    detection->id = i0 + c1.points[i].index;
                    ++nc;
                    first = false;
                }
            }
        }
    }

    // then points of e2 against distance field of e1

    if (!x2.empty() && e2.getCollisionModel()->usePoints.getValue())
    {
        const SReal cubesize = c1.invDP.norm();
        bool first = true;
        for (unsigned int i=0; i<x2.size(); i++)
        {
            DistanceGrid::Coord p2 = x2[i];
            if ((p2-center2).norm2() >= radius2) continue;
            DistanceGrid::Coord p1 = rotation.multTranspose(p2-translation);
            // try to find the point in the undeformed cube
            {
                c1.updateFaces();
                // estimate the barycentric coordinates
                DistanceGrid::Coord b = c1.undeform0(p1);

                // refine the estimate until we are very close to the p1 or we are sure p1 cannot intersect with the object
                int iter;
                SReal err1 = 1000.0f;
                for(iter=0; iter<5; ++iter)
                {
                    DistanceGrid::Coord pdeform = c1.deform(b);
                    DistanceGrid::Coord diff = p1-pdeform;
                    SReal err = diff.norm();
                    SReal berr = err*cubesize; if (berr>0.5f) berr=0.5f;
                    if (b[0] < -berr || b[0] > 1+berr
                        || b[1] < -berr || b[1] > 1+berr
                        || b[2] < -berr || b[2] > 1+berr)
                        break; // far from the cube
                    if (iter>3)
                        sout << "Iter"<<iter<<": "<<err1<<" -> "<<err<<" b = "<<b<<" diff = "<<diff<<" d = "<<grid1->interp(c1.initpos(b))<<""<<sendl;
                    if (err < 0.005f)
                    {
                        // we found the corresponding point, but is is only valid if inside the current cube
                        if (b[0] > 0.001f && b[0] < 0.999f
                            && b[1] > 0.001f && b[1] < 0.999f
                            && b[2] > 0.001f && b[2] < 0.999f)
                        {
                            DistanceGrid::Coord pinit = c1.initpos(b);
                            SReal d = grid1->interp(pinit);
                            if (d < 2*margin)
                            {
                                DistanceGrid::Coord grad = grid1->grad(pinit); // note that there are some redundant computations between interp() and grad()
                                grad.normalize();
                                pinit -= grad*d;
                                grad = c1.deformDir(c1.baryCoords(pinit),grad);
                                grad.normalize();

                                if (!singleContact || first)
                                    contacts->resize(contacts->size()+1);
                                DetectionOutput *detection = &*(contacts->end()-1);
                                double value = d - d0;
                                if (!singleContact || first || (value < detection->value))
                                {
                                    detection->point[0] = Vector3(pinit);
                                    detection->point[1] = Vector3(p2);
                                    detection->normal = Vector3(grad); // normal in global space from p1's surface
                                    detection->value = value;
                                    detection->elem.first = e1;
                                    detection->elem.second = e2;
                                    detection->id = i;
                                    ++nc;
                                    first = false;
                                }
                            }
                        }
                        break;
                    }
                    err1 = err;
                    SReal d = grid1->interp(c1.initpos(b));
                    if (d*0.5f - err > 2*margin)
                        break; // the point is too far from the object
                    // we are solving for deform(b+db)-deform(b) = p1-deform(b)
                    // deform(b+db) ~= deform(b) + J db  -> J db = p1-deform(b) -> db = J^-1 (p1-deform(b))
                    b += c1.undeformDir( b, diff );
                }
                if (iter == 5)
                {
                    if (b[0] > 0.001f && b[0] < 0.999f
                        && b[1] > 0.001f && b[1] < 0.999f
                        && b[2] > 0.001f && b[2] < 0.999f)
                        serr << "ERROR: FFD-Rigid collision failed to converge to undeformed point: p1 = "<<p1<<" b = "<<b<<" c000 = "<<c1.corners[0]<<" c100 = "<<c1.corners[1]<<" c010 = "<<c1.corners[2]<<" c110 = "<<c1.corners[3]<<" c001 = "<<c1.corners[4]<<" c101 = "<<c1.corners[5]<<" c011 = "<<c1.corners[6]<<" c111 = "<<c1.corners[7]<<" pinit = "<<c1.initpos(b)<<" pdeform = "<<c1.deform(b)<<" err = "<<err1<<sendl;
                }
            }
        }
    }
    return nc;
}


bool DiscreteIntersection::testIntersection(FFDDistanceGridCollisionElement&, FFDDistanceGridCollisionElement&)
{
    return true;
}

int DiscreteIntersection::computeIntersection(FFDDistanceGridCollisionElement& e1, FFDDistanceGridCollisionElement& e2, OutputVector* contacts)
{
    int nc = 0;
    DistanceGrid* grid1 = e1.getGrid();
    DistanceGrid* grid2 = e2.getGrid();
    FFDDistanceGridCollisionModel::DeformedCube& c1 = e1.getCollisionModel()->getDeformCube(e1.getIndex());
    FFDDistanceGridCollisionModel::DeformedCube& c2 = e2.getCollisionModel()->getDeformCube(e2.getIndex());
    const bool usePoints1 = e1.getCollisionModel()->usePoints.getValue();
    const bool usePoints2 = e2.getCollisionModel()->usePoints.getValue();
    const bool singleContact = e1.getCollisionModel()->singleContact.getValue() || e2.getCollisionModel()->singleContact.getValue();

    if (!usePoints1 && !usePoints2) return 0; // no tests possible

    const double d0 = e1.getProximity() + e2.getProximity() + (this->getContactDistance() == 0.0 ? 0.001 : this->getContactDistance());
    //const SReal margin = 0.001f + (SReal)d0;
    const SReal margin = (SReal)((e1.getProximity() + e2.getProximity() + (this->getAlarmDistance() == 0.0 ? 0.001 : this->getAlarmDistance()))/2);

    if ((c2.center - c1.center).norm2() > (c1.radius+c2.radius)*(c1.radius+c2.radius))
        return 0; // the two enclosing spheres are not colliding

    int i0 = grid1->meshPts.size();
    if (usePoints1)
    {
        c1.updatePoints();
        c2.updateFaces();
        const SReal cubesize = c2.invDP.norm();
        const sofa::helper::vector<DistanceGrid::Coord>& x1 = c1.deformedPoints;
        const sofa::helper::vector<DistanceGrid::Coord>& n1 = c1.deformedNormals;
        bool first = true;
        for (unsigned int i=0; i<x1.size(); i++)
        {
            DistanceGrid::Coord p2 = x1[i] + n1[i]*margin;

            // estimate the barycentric coordinates
            DistanceGrid::Coord b = c2.undeform0(p2);

            // refine the estimate until we are very close to the p2 or we are sure p2 cannot intersect with the object
            int iter;
            SReal err1 = 1000.0f;
            for(iter=0; iter<5; ++iter)
            {
                DistanceGrid::Coord pdeform = c2.deform(b);
                DistanceGrid::Coord diff = p2-pdeform;
                SReal err = diff.norm();
                SReal berr = err*cubesize; if (berr>0.5f) berr=0.5f;
                if (b[0] < -berr || b[0] > 1+berr
                    || b[1] < -berr || b[1] > 1+berr
                    || b[2] < -berr || b[2] > 1+berr)
                    break; // far from the cube
                if (iter>3)
                    sout << "Iter"<<iter<<": "<<err1<<" -> "<<err<<" b = "<<b<<" diff = "<<diff<<" d = "<<grid2->interp(c2.initpos(b))<<""<<sendl;
                if (err < 0.005f)
                {
                    // we found the corresponding point, but is is only valid if inside the current cube
                    if (b[0] > 0.001f && b[0] < 0.999f
                        && b[1] > 0.001f && b[1] < 0.999f
                        && b[2] > 0.001f && b[2] < 0.999f)
                    {
                        DistanceGrid::Coord pinit = c2.initpos(b);
                        SReal d = grid2->interp(pinit);
                        if (d < margin)
                        {
                            DistanceGrid::Coord grad = grid2->grad(pinit); // note that there are some redundant computations between interp() and grad()
                            grad.normalize();
                            pinit -= grad*d;
                            grad = c2.deformDir(c2.baryCoords(pinit),grad);
                            grad.normalize();

                            if (!singleContact || first)
                                contacts->resize(contacts->size()+1);
                            DetectionOutput *detection = &*(contacts->end()-1);
                            double value = d + margin - d0;
                            if (!singleContact || first || (value < detection->value))
                            {
                                detection->point[0] = Vector3(grid1->meshPts[c1.points[i].index]);
                                detection->point[1] = Vector3(pinit);
                                detection->normal = Vector3(-grad); // normal in global space from p1's surface
                                detection->value = value;
                                detection->elem.first = e1;
                                detection->elem.second = e2;
                                detection->id = c1.points[i].index;
                                ++nc;
                                first = false;
                            }
                        }
                    }
                    break;
                }
                err1 = err;
                SReal d = grid2->interp(c2.initpos(b));
                if (d*0.5f - err > margin)
                    break; // the point is too far from the object
                // we are solving for deform(b+db)-deform(b) = p1-deform(b)
                // deform(b+db) ~= deform(b) + M db  -> M db = p1-deform(b) -> db = M^-1 (p1-deform(b))
                b += c2.undeformDir( b, diff );
            }
            if (iter == 5)
            {
                if (b[0] > 0.001f && b[0] < 0.999f
                    && b[1] > 0.001f && b[1] < 0.999f
                    && b[2] > 0.001f && b[2] < 0.999f)
                    serr << "ERROR: FFD-FFD collision failed to converge to undeformed point: p2 = "<<p2<<" b = "<<b<<" c000 = "<<c2.corners[0]<<" c100 = "<<c2.corners[1]<<" c010 = "<<c2.corners[2]<<" c110 = "<<c2.corners[3]<<" c001 = "<<c2.corners[4]<<" c101 = "<<c2.corners[5]<<" c011 = "<<c2.corners[6]<<" c111 = "<<c2.corners[7]<<" pinit = "<<c2.initpos(b)<<" pdeform = "<<c2.deform(b)<<" err = "<<err1<<sendl;
            }
        }
    }
    if (usePoints2)
    {
        c2.updatePoints();
        c1.updateFaces();
        const SReal cubesize = c1.invDP.norm();
        const sofa::helper::vector<DistanceGrid::Coord>& x2 = c2.deformedPoints;
        const sofa::helper::vector<DistanceGrid::Coord>& n2 = c2.deformedNormals;
        bool first = true;
        for (unsigned int i=0; i<x2.size(); i++)
        {
            DistanceGrid::Coord p1 = x2[i] + n2[i]*margin;

            // estimate the barycentric coordinates
            DistanceGrid::Coord b = c1.undeform0(p1);

            // refine the estimate until we are very close to the p2 or we are sure p2 cannot intersect with the object
            int iter;
            SReal err1 = 1000.0f;
            for(iter=0; iter<5; ++iter)
            {
                DistanceGrid::Coord pdeform = c1.deform(b);
                DistanceGrid::Coord diff = p1-pdeform;
                SReal err = diff.norm();
                if (iter>3)
                    sout << "Iter"<<iter<<": "<<err1<<" -> "<<err<<" b = "<<b<<" diff = "<<diff<<" d = "<<grid1->interp(c1.initpos(b))<<""<<sendl;
                SReal berr = err*cubesize; if (berr>0.5f) berr=0.5f;
                if (b[0] < -berr || b[0] > 1+berr
                    || b[1] < -berr || b[1] > 1+berr
                    || b[2] < -berr || b[2] > 1+berr)
                    break; // far from the cube
                if (err < 0.005f)
                {
                    // we found the corresponding point, but is is only valid if inside the current cube
                    if (b[0] > 0.001f && b[0] < 0.999f
                        && b[1] > 0.001f && b[1] < 0.999f
                        && b[2] > 0.001f && b[2] < 0.999f)
                    {
                        DistanceGrid::Coord pinit = c1.initpos(b);
                        SReal d = grid1->interp(pinit);
                        if (d < margin)
                        {
                            DistanceGrid::Coord grad = grid1->grad(pinit); // note that there are some redundant computations between interp() and grad()
                            grad.normalize();
                            pinit -= grad*d;
                            grad = c1.deformDir(c1.baryCoords(pinit),grad);
                            grad.normalize();

                            if (!singleContact || first)
                                contacts->resize(contacts->size()+1);
                            DetectionOutput *detection = &*(contacts->end()-1);
                            double value = d + margin - d0;
                            if (!singleContact || first || (value < detection->value))
                            {
                                detection->point[0] = Vector3(pinit);
                                detection->point[1] = Vector3(grid2->meshPts[c2.points[i].index]);
                                detection->normal = Vector3(grad); // normal in global space from p1's surface
                                detection->value = value;
                                detection->elem.first = e1;
                                detection->elem.second = e2;
                                detection->id = i0+c2.points[i].index;
                                ++nc;
                                first = false;
                            }
                        }
                    }
                    break;
                }
                err1 = err;
                SReal d = grid1->interp(c1.initpos(b));
                if (d*0.5f - err > margin)
                    break; // the point is too far from the object
                // we are solving for deform(b+db)-deform(b) = p1-deform(b)
                // deform(b+db) ~= deform(b) + M db  -> M db = p1-deform(b) -> db = M^-1 (p1-deform(b))
                b += c1.undeformDir( b, diff );
            }
            if (iter == 5)
            {
                if (b[0] > 0.001f && b[0] < 0.999f
                    && b[1] > 0.001f && b[1] < 0.999f
                    && b[2] > 0.001f && b[2] < 0.999f)
                    serr << "ERROR: FFD-FFD collision failed to converge to undeformed point: p1 = "<<p1<<" b = "<<b<<" c000 = "<<c1.corners[0]<<" c100 = "<<c1.corners[1]<<" c010 = "<<c1.corners[2]<<" c110 = "<<c1.corners[3]<<" c001 = "<<c1.corners[4]<<" c101 = "<<c1.corners[5]<<" c011 = "<<c1.corners[6]<<" c111 = "<<c1.corners[7]<<" pinit = "<<c1.initpos(b)<<" pdeform = "<<c1.deform(b)<<" err = "<<err1<<sendl;
            }
        }
    }
    return nc;
}

bool DiscreteIntersection::testIntersection(FFDDistanceGridCollisionElement&, Point&)
{
    return true;
}

int DiscreteIntersection::computeIntersection(FFDDistanceGridCollisionElement& e1, Point& e2, OutputVector* contacts)
{

    DistanceGrid* grid1 = e1.getGrid();
    FFDDistanceGridCollisionModel::DeformedCube& c1 = e1.getCollisionModel()->getDeformCube(e1.getIndex());

    const double d0 = e1.getProximity() + e2.getProximity() + getContactDistance();
    const SReal margin = 0.001f + (SReal)d0;

    c1.updateFaces();
    const SReal cubesize = c1.invDP.norm();
    int nc = 0;

    Vector3 p2 = e2.p();
    DistanceGrid::Coord p1 = p2;

    // estimate the barycentric coordinates
    DistanceGrid::Coord b = c1.undeform0(p1);

    // refine the estimate until we are very close to the p2 or we are sure p2 cannot intersect with the object
    int iter;
    SReal err1 = 1000.0f;
    for(iter=0; iter<5; ++iter)
    {
        DistanceGrid::Coord pdeform = c1.deform(b);
        DistanceGrid::Coord diff = p1-pdeform;
        SReal err = diff.norm();
        if (iter>3)
            sout << "Iter"<<iter<<": "<<err1<<" -> "<<err<<" b = "<<b<<" diff = "<<diff<<" d = "<<grid1->interp(c1.initpos(b))<<""<<sendl;
        SReal berr = err*cubesize; if (berr>0.5f) berr=0.5f;
        if (b[0] < -berr || b[0] > 1+berr
            || b[1] < -berr || b[1] > 1+berr
            || b[2] < -berr || b[2] > 1+berr)
            break; // far from the cube
        if (err < 0.005f)
        {
            // we found the corresponding point, but is is only valid if inside the current cube
            if (b[0] > 0.001f && b[0] < 0.999f
                && b[1] > 0.001f && b[1] < 0.999f
                && b[2] > 0.001f && b[2] < 0.999f)
            {
                DistanceGrid::Coord pinit = c1.initpos(b);
                SReal d = grid1->interp(pinit);
                if (d < margin)
                {
                    DistanceGrid::Coord grad = grid1->grad(pinit); // note that there are some redundant computations between interp() and grad()
                    grad.normalize();
                    pinit -= grad*d;
                    grad = c1.deformDir(c1.baryCoords(pinit),grad);
                    grad.normalize();

                    contacts->resize(contacts->size()+1);
                    DetectionOutput *detection = &*(contacts->end()-1);

                    detection->point[0] = Vector3(pinit);
                    detection->point[1] = Vector3(p2);
                    detection->normal = Vector3(grad); // normal in global space from p1's surface
                    detection->value = d - d0;
                    detection->elem.first = e1;
                    detection->elem.second = e2;
                    detection->id = e2.getIndex();
                    ++nc;
                }
            }
            break;
        }
        err1 = err;
        SReal d = grid1->interp(c1.initpos(b));
        if (d*0.5f - err > margin)
            break; // the point is too far from the object
        // we are solving for deform(b+db)-deform(b) = p1-deform(b)
        // deform(b+db) ~= deform(b) + M db  -> M db = p1-deform(b) -> db = M^-1 (p1-deform(b))
        b += c1.undeformDir( b, diff );
    }
    if (iter == 5)
    {
        if (b[0] > 0.001f && b[0] < 0.999f
            && b[1] > 0.001f && b[1] < 0.999f
            && b[2] > 0.001f && b[2] < 0.999f)
            serr << "ERROR: FFD-FFD collision failed to converge to undeformed point: p1 = "<<p1<<" b = "<<b<<" c000 = "<<c1.corners[0]<<" c100 = "<<c1.corners[1]<<" c010 = "<<c1.corners[2]<<" c110 = "<<c1.corners[3]<<" c001 = "<<c1.corners[4]<<" c101 = "<<c1.corners[5]<<" c011 = "<<c1.corners[6]<<" c111 = "<<c1.corners[7]<<" pinit = "<<c1.initpos(b)<<" pdeform = "<<c1.deform(b)<<" err = "<<err1<<sendl;
    }

    return nc;
}

bool DiscreteIntersection::testIntersection(FFDDistanceGridCollisionElement&, Triangle&)
{
    return true;
}

int DiscreteIntersection::computeIntersection(FFDDistanceGridCollisionElement& e1, Triangle& e2, OutputVector* contacts)
{
    const int f2 = e2.flags();
    if (!(f2&TriangleModel::FLAG_POINTS)) return 0; // no points associated with this triangle

    DistanceGrid* grid1 = e1.getGrid();
    FFDDistanceGridCollisionModel::DeformedCube& c1 = e1.getCollisionModel()->getDeformCube(e1.getIndex());

    const double d0 = e1.getProximity() + e2.getProximity() + getContactDistance();
    const SReal margin = 0.001f + (SReal)d0;

    c1.updateFaces();
    const SReal cubesize = c1.invDP.norm();
    int nc = 0;

    if (f2&TriangleModel::FLAG_P1)
    {
        Vector3 p2 = e2.p1();
        DistanceGrid::Coord p1 = p2;

        // estimate the barycentric coordinates
        DistanceGrid::Coord b = c1.undeform0(p1);

        // refine the estimate until we are very close to the p2 or we are sure p2 cannot intersect with the object
        int iter;
        SReal err1 = 1000.0f;
        for(iter=0; iter<5; ++iter)
        {
            DistanceGrid::Coord pdeform = c1.deform(b);
            DistanceGrid::Coord diff = p1-pdeform;
            SReal err = diff.norm();
            if (iter>3)
                sout << "Iter"<<iter<<": "<<err1<<" -> "<<err<<" b = "<<b<<" diff = "<<diff<<" d = "<<grid1->interp(c1.initpos(b))<<""<<sendl;
            SReal berr = err*cubesize; if (berr>0.5f) berr=0.5f;
            if (b[0] < -berr || b[0] > 1+berr
                || b[1] < -berr || b[1] > 1+berr
                || b[2] < -berr || b[2] > 1+berr)
                break; // far from the cube
            if (err < 0.005f)
            {
                // we found the corresponding point, but is is only valid if inside the current cube
                if (b[0] > 0.001f && b[0] < 0.999f
                    && b[1] > 0.001f && b[1] < 0.999f
                    && b[2] > 0.001f && b[2] < 0.999f)
                {
                    DistanceGrid::Coord pinit = c1.initpos(b);
                    SReal d = grid1->interp(pinit);
                    if (d < margin)
                    {
                        DistanceGrid::Coord grad = grid1->grad(pinit); // note that there are some redundant computations between interp() and grad()
                        grad.normalize();
                        pinit -= grad*d;
                        grad = c1.deformDir(c1.baryCoords(pinit),grad);
                        grad.normalize();

                        contacts->resize(contacts->size()+1);
                        DetectionOutput *detection = &*(contacts->end()-1);

                        detection->point[0] = Vector3(pinit);
                        detection->point[1] = Vector3(p2);
                        detection->normal = Vector3(grad); // normal in global space from p1's surface
                        detection->value = d - d0;
                        detection->elem.first = e1;
                        detection->elem.second = e2;
                        detection->id = e2.getIndex()*3+0;
                        ++nc;
                    }
                }
                break;
            }
            err1 = err;
            SReal d = grid1->interp(c1.initpos(b));
            if (d*0.5f - err > margin)
                break; // the point is too far from the object
            // we are solving for deform(b+db)-deform(b) = p1-deform(b)
            // deform(b+db) ~= deform(b) + M db  -> M db = p1-deform(b) -> db = M^-1 (p1-deform(b))
            b += c1.undeformDir( b, diff );
        }
        if (iter == 5)
        {
            if (b[0] > 0.001f && b[0] < 0.999f
                && b[1] > 0.001f && b[1] < 0.999f
                && b[2] > 0.001f && b[2] < 0.999f)
                serr << "ERROR: FFD-FFD collision failed to converge to undeformed point: p1 = "<<p1<<" b = "<<b<<" c000 = "<<c1.corners[0]<<" c100 = "<<c1.corners[1]<<" c010 = "<<c1.corners[2]<<" c110 = "<<c1.corners[3]<<" c001 = "<<c1.corners[4]<<" c101 = "<<c1.corners[5]<<" c011 = "<<c1.corners[6]<<" c111 = "<<c1.corners[7]<<" pinit = "<<c1.initpos(b)<<" pdeform = "<<c1.deform(b)<<" err = "<<err1<<sendl;
        }
    }

    if (f2&TriangleModel::FLAG_P2)
    {
        Vector3 p2 = e2.p2();
        DistanceGrid::Coord p1 = p2;

        // estimate the barycentric coordinates
        DistanceGrid::Coord b = c1.undeform0(p1);

        // refine the estimate until we are very close to the p2 or we are sure p2 cannot intersect with the object
        int iter;
        SReal err1 = 1000.0f;
        for(iter=0; iter<5; ++iter)
        {
            DistanceGrid::Coord pdeform = c1.deform(b);
            DistanceGrid::Coord diff = p1-pdeform;
            SReal err = diff.norm();
            if (iter>3)
                sout << "Iter"<<iter<<": "<<err1<<" -> "<<err<<" b = "<<b<<" diff = "<<diff<<" d = "<<grid1->interp(c1.initpos(b))<<""<<sendl;
            SReal berr = err*cubesize; if (berr>0.5f) berr=0.5f;
            if (b[0] < -berr || b[0] > 1+berr
                || b[1] < -berr || b[1] > 1+berr
                || b[2] < -berr || b[2] > 1+berr)
                break; // far from the cube
            if (err < 0.005f)
            {
                // we found the corresponding point, but is is only valid if inside the current cube
                if (b[0] > 0.001f && b[0] < 0.999f
                    && b[1] > 0.001f && b[1] < 0.999f
                    && b[2] > 0.001f && b[2] < 0.999f)
                {
                    DistanceGrid::Coord pinit = c1.initpos(b);
                    SReal d = grid1->interp(pinit);
                    if (d < margin)
                    {
                        DistanceGrid::Coord grad = grid1->grad(pinit); // note that there are some redundant computations between interp() and grad()
                        grad.normalize();
                        pinit -= grad*d;
                        grad = c1.deformDir(c1.baryCoords(pinit),grad);
                        grad.normalize();

                        contacts->resize(contacts->size()+1);
                        DetectionOutput *detection = &*(contacts->end()-1);

                        detection->point[0] = Vector3(pinit);
                        detection->point[1] = Vector3(p2);
                        detection->normal = Vector3(grad); // normal in global space from p1's surface
                        detection->value = d - d0;
                        detection->elem.first = e1;
                        detection->elem.second = e2;
                        detection->id = e2.getIndex()*3+1;
                        ++nc;
                    }
                }
                break;
            }
            err1 = err;
            SReal d = grid1->interp(c1.initpos(b));
            if (d*0.5f - err > margin)
                break; // the point is too far from the object
            // we are solving for deform(b+db)-deform(b) = p1-deform(b)
            // deform(b+db) ~= deform(b) + M db  -> M db = p1-deform(b) -> db = M^-1 (p1-deform(b))
            b += c1.undeformDir( b, diff );
        }
        if (iter == 5)
        {
            if (b[0] > 0.001f && b[0] < 0.999f
                && b[1] > 0.001f && b[1] < 0.999f
                && b[2] > 0.001f && b[2] < 0.999f)
                serr << "ERROR: FFD-FFD collision failed to converge to undeformed point: p1 = "<<p1<<" b = "<<b<<" c000 = "<<c1.corners[0]<<" c100 = "<<c1.corners[1]<<" c010 = "<<c1.corners[2]<<" c110 = "<<c1.corners[3]<<" c001 = "<<c1.corners[4]<<" c101 = "<<c1.corners[5]<<" c011 = "<<c1.corners[6]<<" c111 = "<<c1.corners[7]<<" pinit = "<<c1.initpos(b)<<" pdeform = "<<c1.deform(b)<<" err = "<<err1<<sendl;
        }
    }

    if (f2&TriangleModel::FLAG_P3)
    {
        Vector3 p2 = e2.p3();
        DistanceGrid::Coord p1 = p2;

        // estimate the barycentric coordinates
        DistanceGrid::Coord b = c1.undeform0(p1);

        // refine the estimate until we are very close to the p2 or we are sure p2 cannot intersect with the object
        int iter;
        SReal err1 = 1000.0f;
        for(iter=0; iter<5; ++iter)
        {
            DistanceGrid::Coord pdeform = c1.deform(b);
            DistanceGrid::Coord diff = p1-pdeform;
            SReal err = diff.norm();
            if (iter>3)
                sout << "Iter"<<iter<<": "<<err1<<" -> "<<err<<" b = "<<b<<" diff = "<<diff<<" d = "<<grid1->interp(c1.initpos(b))<<""<<sendl;
            SReal berr = err*cubesize; if (berr>0.5f) berr=0.5f;
            if (b[0] < -berr || b[0] > 1+berr
                || b[1] < -berr || b[1] > 1+berr
                || b[2] < -berr || b[2] > 1+berr)
                break; // far from the cube
            if (err < 0.005f)
            {
                // we found the corresponding point, but is is only valid if inside the current cube
                if (b[0] > 0.001f && b[0] < 0.999f
                    && b[1] > 0.001f && b[1] < 0.999f
                    && b[2] > 0.001f && b[2] < 0.999f)
                {
                    DistanceGrid::Coord pinit = c1.initpos(b);
                    SReal d = grid1->interp(pinit);
                    if (d < margin)
                    {
                        DistanceGrid::Coord grad = grid1->grad(pinit); // note that there are some redundant computations between interp() and grad()
                        grad.normalize();
                        pinit -= grad*d;
                        grad = c1.deformDir(c1.baryCoords(pinit),grad);
                        grad.normalize();

                        contacts->resize(contacts->size()+1);
                        DetectionOutput *detection = &*(contacts->end()-1);

                        detection->point[0] = Vector3(pinit);
                        detection->point[1] = Vector3(p2);
                        detection->normal = Vector3(grad); // normal in global space from p1's surface
                        detection->value = d - d0;
                        detection->elem.first = e1;
                        detection->elem.second = e2;
                        detection->id = e2.getIndex()*3+2;
                        ++nc;
                    }
                }
                break;
            }
            err1 = err;
            SReal d = grid1->interp(c1.initpos(b));
            if (d*0.5f - err > margin)
                break; // the point is too far from the object
            // we are solving for deform(b+db)-deform(b) = p1-deform(b)
            // deform(b+db) ~= deform(b) + M db  -> M db = p1-deform(b) -> db = M^-1 (p1-deform(b))
            b += c1.undeformDir( b, diff );
        }
        if (iter == 5)
        {
            if (b[0] > 0.001f && b[0] < 0.999f
                && b[1] > 0.001f && b[1] < 0.999f
                && b[2] > 0.001f && b[2] < 0.999f)
                serr << "ERROR: FFD-FFD collision failed to converge to undeformed point: p1 = "<<p1<<" b = "<<b<<" c000 = "<<c1.corners[0]<<" c100 = "<<c1.corners[1]<<" c010 = "<<c1.corners[2]<<" c110 = "<<c1.corners[3]<<" c001 = "<<c1.corners[4]<<" c101 = "<<c1.corners[5]<<" c011 = "<<c1.corners[6]<<" c111 = "<<c1.corners[7]<<" pinit = "<<c1.initpos(b)<<" pdeform = "<<c1.deform(b)<<" err = "<<err1<<sendl;
        }
    }
    return nc;
}

bool DiscreteIntersection::testIntersection(Ray& /*e1*/, FFDDistanceGridCollisionElement& /*e2*/)
{
    return true;
}

int DiscreteIntersection::computeIntersection(Ray& e2, FFDDistanceGridCollisionElement& e1, OutputVector* contacts)
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

