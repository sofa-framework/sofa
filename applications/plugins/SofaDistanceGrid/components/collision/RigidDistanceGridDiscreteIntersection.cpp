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
#include <iostream>
#include <algorithm>

#include <sofa/core/collision/Intersection.inl>
#include <sofa/core/collision/IntersectorFactory.h>
#include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.inl>
#include <sofa/helper/proximity.h>
#include <SofaBaseCollision/DiscreteIntersection.h>
#include "RigidDistanceGridDiscreteIntersection.inl"

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;

SOFA_DECL_CLASS(RigidDistanceGridDiscreteIntersection)

IntersectorCreator<DiscreteIntersection, RigidDistanceGridDiscreteIntersection> RigidDistanceGridDiscreteIntersectors("RigidDistanceGrid");

RigidDistanceGridDiscreteIntersection::RigidDistanceGridDiscreteIntersection(DiscreteIntersection* object)
    : intersection(object)
{
    intersection->intersectors.add<RigidDistanceGridCollisionModel, PointModel,                      RigidDistanceGridDiscreteIntersection>  (this);
    intersection->intersectors.add<RigidDistanceGridCollisionModel, SphereModel,                     RigidDistanceGridDiscreteIntersection>  (this);
    intersection->intersectors.add<RigidDistanceGridCollisionModel, LineModel,                       RigidDistanceGridDiscreteIntersection>  (this);
    intersection->intersectors.add<RigidDistanceGridCollisionModel, TriangleModel,                   RigidDistanceGridDiscreteIntersection>  (this);
    intersection->intersectors.add<RayModel, RigidDistanceGridCollisionModel, RigidDistanceGridDiscreteIntersection>  (this);
    intersection->intersectors.add<RigidDistanceGridCollisionModel, RigidDistanceGridCollisionModel, RigidDistanceGridDiscreteIntersection> (this);
}

bool RigidDistanceGridDiscreteIntersection::testIntersection(RigidDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&)
{
    return true;
}

//#define DEBUG_XFORM

int RigidDistanceGridDiscreteIntersection::computeIntersection(RigidDistanceGridCollisionElement& e1, RigidDistanceGridCollisionElement& e2, OutputVector* contacts)
{
    int nc = 0;
    DistanceGrid* grid1 = e1.getGrid();
    DistanceGrid* grid2 = e2.getGrid();
    bool useXForm = e1.isTransformed() || e2.isTransformed();
    const Vector3& t1 = e1.getTranslation();
    const Matrix3& r1 = e1.getRotation();
    const Vector3& t2 = e2.getTranslation();
    const Matrix3& r2 = e2.getRotation();

    const double d0 = e1.getProximity() + e2.getProximity() + (intersection->getContactDistance() == 0.0 ? 0.001 : intersection->getContactDistance());
    //const SReal margin = 0.001f + (SReal)d0;
    const SReal margin = (SReal)((e1.getProximity() + e2.getProximity() + (intersection->getAlarmDistance() == 0.0 ? 0.001 : intersection->getAlarmDistance()))/2);

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
                    SReal d = sofa::helper::rabs(sofa::helper::rabs(translation[f2])-(cubeDim1+cubeDim2));
                    // we should favor normals that are perpendicular to the relative velocity
                    // however we don't have this information currently, so for now we favor the horizontal face
                    if (sofa::helper::rabs(r2[f2][2]) > 0.99 && d < (cubeDim1 + cubeDim2) * 0.1) d = 0;
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                    detection->baryCoords[0] = Vector3(p1);
                    detection->baryCoords[1] = Vector3(p2);
#endif
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
                    normal[0] = sofa::helper::rabs(p2[0]) - cubeDim2;
                    normal[1] = sofa::helper::rabs(p2[1]) - cubeDim2;
                    normal[2] = sofa::helper::rabs(p2[2]) - cubeDim2;

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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                    detection->baryCoords[0] = Vector3(p1);
                    detection->baryCoords[1] = Vector3(p2);
#endif
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
                    intersection->serr << "WARNING: margin less than "<<margin<<" in DistanceGrid "<<e2.getCollisionModel()->getName()<<intersection->sendl;
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                detection->baryCoords[0] = Vector3(p1);
                detection->baryCoords[1] = Vector3(p2);
#endif
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                    detection->baryCoords[0] = Vector3(p1);
                    detection->baryCoords[1] = Vector3(p2);
#endif
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
                    normal[0] = sofa::helper::rabs(p1[0]) - cubeDim1;
                    normal[1] = sofa::helper::rabs(p1[1]) - cubeDim1;
                    normal[2] = sofa::helper::rabs(p1[2]) - cubeDim1;

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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                    detection->baryCoords[0] = Vector3(p1);
                    detection->baryCoords[1] = Vector3(p2);
#endif
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                detection->baryCoords[0] = Vector3(p1);
                detection->baryCoords[1] = Vector3(p2);
#endif
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
                    intersection->serr << "WARNING: margin less than "<<margin<<" in DistanceGrid "<<e1.getCollisionModel()->getName()<<intersection->sendl;
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                detection->baryCoords[0] = Vector3(p1);
                detection->baryCoords[1] = Vector3(p2);
#endif
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

bool RigidDistanceGridDiscreteIntersection::testIntersection(RigidDistanceGridCollisionElement&, Point&)
{
    return true;
}

int RigidDistanceGridDiscreteIntersection::computeIntersection(RigidDistanceGridCollisionElement& e1, Point& e2, OutputVector* contacts)
{
    DistanceGrid* grid1 = e1.getGrid();
    bool useXForm = e1.isTransformed();
    const Vector3& t1 = e1.getTranslation();
    const Matrix3& r1 = e1.getRotation();
    const bool flipped = e1.isFlipped();

    const double d0 = e1.getProximity() + e2.getProximity() + intersection->getContactDistance();
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
            intersection->serr << "WARNING: margin less than "<<margin<<" in DistanceGrid "<<e1.getCollisionModel()->getName()<<intersection->sendl;
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
    detection->baryCoords[0] = Vector3(p1);
    detection->baryCoords[1] = Vector3(0,0,0);
#endif
    detection->normal = (useXForm) ? r1 * grad : grad; // normal in global space from p1's surface
    detection->value = d - d0;
    detection->elem.first = e1;
    detection->elem.second = e2;
    detection->id = e2.getIndex();
    return 1;
}

bool RigidDistanceGridDiscreteIntersection::testIntersection(RigidDistanceGridCollisionElement&, Triangle&)
{
    return true;
}

int RigidDistanceGridDiscreteIntersection::computeIntersection(RigidDistanceGridCollisionElement& e1, Triangle& e2, OutputVector* contacts)
{
    const int f2 = e2.flags();
    if (!(f2&(TriangleModel::FLAG_POINTS|TriangleModel::FLAG_BEDGES))) return 0; // no points associated with this triangle
    DistanceGrid* grid1 = e1.getGrid();
    const bool useXForm = e1.isTransformed();
    const Vector3& t1 = e1.getTranslation();
    const Matrix3& r1 = e1.getRotation();

    const double d0 = e1.getProximity() + e2.getProximity() + intersection->getContactDistance();
    const SReal margin = 0.001f + (SReal)d0;
    int nc = 0;
    for (unsigned int iP = 0; iP < 3; ++iP)
    {
        if (!(f2&(TriangleModel::FLAG_P1 << iP))) continue;

        Vector3 p2 = e2.p(iP);
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
                intersection->serr << "WARNING: margin less than "<<margin<<" in DistanceGrid "<<e1.getCollisionModel()->getName()<<intersection->sendl;
            }
            else
            {
                SReal d = grid1->interp(p1);
                if (d >= margin) continue;

                Vector3 grad = grid1->grad(p1); // note that there are some redundant computations between interp() and grad()
                grad.normalize();

                //p1 -= grad * d; // push p1 back to the surface

                contacts->resize(contacts->size()+1);
                DetectionOutput *detection = &*(contacts->end()-1);

                detection->point[0] = Vector3(p1) - grad * d;
                detection->point[1] = Vector3(p2);
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                detection->baryCoords[0] = Vector3(p1);
                detection->baryCoords[1] = Vector3((iP == 1)?1.0:0.0,(iP == 2)?1.0:0.0,0.0);
#endif
                detection->normal = (useXForm) ? r1 * grad : grad; // normal in global space from p1's surface
                detection->value = d - d0;
                detection->elem.first = e1;
                detection->elem.second = e2;
                detection->id = e2.getIndex()*6+iP;
                ++nc;
            }
        }
    }
    for (unsigned int iE = 0; iE < 3; ++iE)
    {
        if (!(f2&(TriangleModel::FLAG_BE23 << iE))) continue;
        unsigned int iP1 = (iE+1)%3;
        unsigned int iP2 = (iE+2)%3;
        Vector3 p2 = (e2.p(iP1)+e2.p(iP2))*0.5;

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
                intersection->serr << "WARNING: margin less than "<<margin<<" in DistanceGrid "<<e1.getCollisionModel()->getName()<<intersection->sendl;
            }
            else
            {
                SReal d = grid1->interp(p1);
                if (d >= margin) continue;

                Vector3 grad = grid1->grad(p1); // note that there are some redundant computations between interp() and grad()
                grad.normalize();

                //p1 -= grad * d; // push p1 back to the surface

                contacts->resize(contacts->size()+1);
                DetectionOutput *detection = &*(contacts->end()-1);

                detection->point[0] = Vector3(p1) - grad * d;
                detection->point[1] = Vector3(p2);
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                detection->baryCoords[0] = Vector3(p1);
                detection->baryCoords[1] = Vector3(((iE != 1)?0.5:0.0),
                                                   ((iE != 2)?0.5:0.0),
                                                   0.0);
#endif
                detection->normal = (useXForm) ? r1 * grad : grad; // normal in global space from p1's surface
                detection->value = d - d0;
                detection->elem.first = e1;
                detection->elem.second = e2;
                detection->id = e2.getIndex()*6+(3+iE);
                ++nc;
            }
        }
    }

    return nc;
}

bool RigidDistanceGridDiscreteIntersection::testIntersection(RigidDistanceGridCollisionElement&, Line&)
{
    return true;
}

int RigidDistanceGridDiscreteIntersection::computeIntersection(RigidDistanceGridCollisionElement& e1, Line& e2, OutputVector* contacts)
{
    const int f2 = e2.flags();
    if (!(f2&LineModel::FLAG_POINTS)) return 0; // no points associated with this line
    DistanceGrid* grid1 = e1.getGrid();
    const bool useXForm = e1.isTransformed();
    const Vector3& t1 = e1.getTranslation();
    const Matrix3& r1 = e1.getRotation();

    const double d0 = e1.getProximity() + e2.getProximity() + intersection->getContactDistance();
    const SReal margin = 0.001f + (SReal)d0;
    int nresult = 0;
    for (unsigned int iP = 0; iP < 2; ++iP)
    {
        if (!(f2&(LineModel::FLAG_P1 << iP))) continue;

        Vector3 p2 = e2.p(iP);
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
                intersection->serr << "WARNING: margin less than "<<margin<<" in DistanceGrid "<<e1.getCollisionModel()->getName()<<intersection->sendl;
            }
            else
            {
                SReal d = grid1->interp(p1);
                if (d >= margin) continue;

                Vector3 grad = grid1->grad(p1); // note that there are some redundant computations between interp() and grad()
                grad.normalize();

                //p1 -= grad * d; // push p1 back to the surface

                contacts->resize(contacts->size()+1);
                DetectionOutput *detection = &*(contacts->end()-1);

                detection->point[0] = Vector3(p1) - grad * d;
                detection->point[1] = Vector3(p2);
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                detection->baryCoords[0] = Vector3(p1);
                detection->baryCoords[1] = Vector3((iP == 1)?1.0:0.0,0.0,0.0);
#endif
                detection->normal = (useXForm) ? r1 * grad : grad; // normal in global space from p1's surface
                detection->value = d - d0;
                detection->elem.first = e1;
                detection->elem.second = e2;
                detection->id = e2.getIndex()*2+iP;
                ++nresult;
            }
        }
    }

    return nresult;
}

bool RigidDistanceGridDiscreteIntersection::testIntersection(Ray& /*e2*/, RigidDistanceGridCollisionElement& /*e1*/)
{
    return true;
}

int RigidDistanceGridDiscreteIntersection::computeIntersection(Ray& e2, RigidDistanceGridCollisionElement& e1, OutputVector* contacts)
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
            detection->baryCoords[0] = Vector3(l0,0,0);
            detection->baryCoords[1] = p;
#endif
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


} // namespace collision

} // namespace component

} // namespace sofa

