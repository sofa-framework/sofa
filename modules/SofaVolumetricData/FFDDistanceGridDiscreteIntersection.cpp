/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaVolumetricData/FFDDistanceGridDiscreteIntersection.inl>
#include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.inl>
#include <SofaBaseCollision/DiscreteIntersection.h>
#include <sofa/core/collision/Intersection.inl>
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

SOFA_DECL_CLASS(FFDDistanceGridDiscreteIntersection)

IntersectorCreator<DiscreteIntersection, FFDDistanceGridDiscreteIntersection> FFDDistanceGridDiscreteIntersectors("FFDDistanceGrid");

FFDDistanceGridDiscreteIntersection::FFDDistanceGridDiscreteIntersection(DiscreteIntersection* object)
    : intersection(object)
{
    intersection->intersectors.add<FFDDistanceGridCollisionModel, PointModel,                        FFDDistanceGridDiscreteIntersection>  (this);
    intersection->intersectors.add<FFDDistanceGridCollisionModel, SphereModel,                       FFDDistanceGridDiscreteIntersection>  (this);
    intersection->intersectors.add<FFDDistanceGridCollisionModel, TriangleModel,                     FFDDistanceGridDiscreteIntersection>  (this);
    intersection->intersectors.add<RayModel, FFDDistanceGridCollisionModel,   FFDDistanceGridDiscreteIntersection>  (this);
    intersection->intersectors.add<FFDDistanceGridCollisionModel,   RigidDistanceGridCollisionModel, FFDDistanceGridDiscreteIntersection>  (this);
    intersection->intersectors.add<FFDDistanceGridCollisionModel,   FFDDistanceGridCollisionModel,   FFDDistanceGridDiscreteIntersection> (this);
}

bool FFDDistanceGridDiscreteIntersection::testIntersection(FFDDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&)
{
    return true;
}

int FFDDistanceGridDiscreteIntersection::computeIntersection(FFDDistanceGridCollisionElement& e1, RigidDistanceGridCollisionElement& e2, OutputVector* contacts)
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

    const double d0 = e1.getProximity() + e2.getProximity() + (intersection->getContactDistance() == 0.0 ? 0.001 : intersection->getContactDistance());
    //const SReal margin = 0.001f + (SReal)d0;
    const SReal margin = (SReal)((e1.getProximity() + e2.getProximity() + (intersection->getAlarmDistance() == 0.0 ? 0.001 : intersection->getAlarmDistance()))/2);
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
                    intersection->serr << "WARNING: margin less than "<<margin<<" in DistanceGrid "<<e2.getCollisionModel()->getName()<<intersection->sendl;
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
                if (!singleContact || first || (value < detection->value))
                {
                    detection->point[0] = grid1->meshPts[c1.points[i].index];
                    detection->point[1] = Vector3(p2) - grad * d;
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                    detection->baryCoords[0] = detection->point[0];
                    detection->baryCoords[1] = Vector3(p2);
#endif
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
                        intersection->sout << "Iter"<<iter<<": "<<err1<<" -> "<<err<<" b = "<<b<<" diff = "<<diff<<" d = "<<grid1->interp(c1.initpos(b))<<""<<intersection->sendl;
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                                    detection->baryCoords[0] = Vector3(pinit);
                                    detection->baryCoords[1] = Vector3(p2);
#endif
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
                        intersection->serr << "ERROR: FFD-Rigid collision failed to converge to undeformed point: p1 = "<<p1<<" b = "<<b<<" c000 = "<<c1.corners[0]<<" c100 = "<<c1.corners[1]<<" c010 = "<<c1.corners[2]<<" c110 = "<<c1.corners[3]<<" c001 = "<<c1.corners[4]<<" c101 = "<<c1.corners[5]<<" c011 = "<<c1.corners[6]<<" c111 = "<<c1.corners[7]<<" pinit = "<<c1.initpos(b)<<" pdeform = "<<c1.deform(b)<<" err = "<<err1<<intersection->sendl;
                }
            }
        }
    }
    return nc;
}


bool FFDDistanceGridDiscreteIntersection::testIntersection(FFDDistanceGridCollisionElement&, FFDDistanceGridCollisionElement&)
{
    return true;
}

int FFDDistanceGridDiscreteIntersection::computeIntersection(FFDDistanceGridCollisionElement& e1, FFDDistanceGridCollisionElement& e2, OutputVector* contacts)
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

    const double d0 = e1.getProximity() + e2.getProximity() + (intersection->getContactDistance() == 0.0 ? 0.001 : intersection->getContactDistance());
    //const SReal margin = 0.001f + (SReal)d0;
    const SReal margin = (SReal)((e1.getProximity() + e2.getProximity() + (intersection->getAlarmDistance() == 0.0 ? 0.001 : intersection->getAlarmDistance()))/2);

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
                    intersection->sout << "Iter"<<iter<<": "<<err1<<" -> "<<err<<" b = "<<b<<" diff = "<<diff<<" d = "<<grid2->interp(c2.initpos(b))<<""<<intersection->sendl;
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                                detection->baryCoords[0] = detection->point[0];
                                detection->baryCoords[1] = Vector3(pinit);
#endif
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
                    intersection->serr << "ERROR: FFD-FFD collision failed to converge to undeformed point: p2 = "<<p2<<" b = "<<b<<" c000 = "<<c2.corners[0]<<" c100 = "<<c2.corners[1]<<" c010 = "<<c2.corners[2]<<" c110 = "<<c2.corners[3]<<" c001 = "<<c2.corners[4]<<" c101 = "<<c2.corners[5]<<" c011 = "<<c2.corners[6]<<" c111 = "<<c2.corners[7]<<" pinit = "<<c2.initpos(b)<<" pdeform = "<<c2.deform(b)<<" err = "<<err1<<intersection->sendl;
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
                    intersection->sout << "Iter"<<iter<<": "<<err1<<" -> "<<err<<" b = "<<b<<" diff = "<<diff<<" d = "<<grid1->interp(c1.initpos(b))<<""<<intersection->sendl;
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                                detection->baryCoords[0] = Vector3(pinit);
                                detection->baryCoords[1] = detection->point[1];
#endif
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
                    intersection->serr << "ERROR: FFD-FFD collision failed to converge to undeformed point: p1 = "<<p1<<" b = "<<b<<" c000 = "<<c1.corners[0]<<" c100 = "<<c1.corners[1]<<" c010 = "<<c1.corners[2]<<" c110 = "<<c1.corners[3]<<" c001 = "<<c1.corners[4]<<" c101 = "<<c1.corners[5]<<" c011 = "<<c1.corners[6]<<" c111 = "<<c1.corners[7]<<" pinit = "<<c1.initpos(b)<<" pdeform = "<<c1.deform(b)<<" err = "<<err1<<intersection->sendl;
            }
        }
    }
    return nc;
}

bool FFDDistanceGridDiscreteIntersection::testIntersection(FFDDistanceGridCollisionElement&, Point&)
{
    return true;
}

int FFDDistanceGridDiscreteIntersection::computeIntersection(FFDDistanceGridCollisionElement& e1, Point& e2, OutputVector* contacts)
{

    DistanceGrid* grid1 = e1.getGrid();
    FFDDistanceGridCollisionModel::DeformedCube& c1 = e1.getCollisionModel()->getDeformCube(e1.getIndex());

    const double d0 = e1.getProximity() + e2.getProximity() + intersection->getContactDistance();
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
            intersection->sout << "Iter"<<iter<<": "<<err1<<" -> "<<err<<" b = "<<b<<" diff = "<<diff<<" d = "<<grid1->interp(c1.initpos(b))<<""<<intersection->sendl;
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                    detection->baryCoords[0] = Vector3(pinit);
                    detection->baryCoords[1] = Vector3(0.0,0.0,0.0);
#endif
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
            intersection->serr << "ERROR: FFD-FFD collision failed to converge to undeformed point: p1 = "<<p1<<" b = "<<b<<" c000 = "<<c1.corners[0]<<" c100 = "<<c1.corners[1]<<" c010 = "<<c1.corners[2]<<" c110 = "<<c1.corners[3]<<" c001 = "<<c1.corners[4]<<" c101 = "<<c1.corners[5]<<" c011 = "<<c1.corners[6]<<" c111 = "<<c1.corners[7]<<" pinit = "<<c1.initpos(b)<<" pdeform = "<<c1.deform(b)<<" err = "<<err1<<intersection->sendl;
    }

    return nc;
}

bool FFDDistanceGridDiscreteIntersection::testIntersection(FFDDistanceGridCollisionElement&, Triangle&)
{
    return true;
}

int FFDDistanceGridDiscreteIntersection::computeIntersection(FFDDistanceGridCollisionElement& e1, Triangle& e2, OutputVector* contacts)
{
    const int f2 = e2.flags();
    if (!(f2&TriangleModel::FLAG_POINTS)) return 0; // no points associated with this triangle

    DistanceGrid* grid1 = e1.getGrid();
    FFDDistanceGridCollisionModel::DeformedCube& c1 = e1.getCollisionModel()->getDeformCube(e1.getIndex());

    const double d0 = e1.getProximity() + e2.getProximity() + intersection->getContactDistance();
    const SReal margin = 0.001f + (SReal)d0;

    c1.updateFaces();
    const SReal cubesize = c1.invDP.norm();
    int nc = 0;
    for (unsigned int iP = 0; iP < 3; ++iP)
    {
        if (!(f2&(TriangleModel::FLAG_P1<<iP))) continue;
        Vector3 p2 = e2.p(iP);
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
                intersection->sout << "Iter"<<iter<<": "<<err1<<" -> "<<err<<" b = "<<b<<" diff = "<<diff<<" d = "<<grid1->interp(c1.initpos(b))<<""<<intersection->sendl;
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                        detection->baryCoords[0] = Vector3(pinit);
                        detection->baryCoords[1] = Vector3((iP == 1)?1.0:0.0,(iP == 2)?1.0:0.0,0.0);
#endif
                        detection->normal = Vector3(grad); // normal in global space from p1's surface
                        detection->value = d - d0;
                        detection->elem.first = e1;
                        detection->elem.second = e2;
                        detection->id = e2.getIndex()*3+iP;
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
                intersection->serr << "ERROR: FFD-FFD collision failed to converge to undeformed point: p1 = "<<p1<<" b = "<<b<<" c000 = "<<c1.corners[0]<<" c100 = "<<c1.corners[1]<<" c010 = "<<c1.corners[2]<<" c110 = "<<c1.corners[3]<<" c001 = "<<c1.corners[4]<<" c101 = "<<c1.corners[5]<<" c011 = "<<c1.corners[6]<<" c111 = "<<c1.corners[7]<<" pinit = "<<c1.initpos(b)<<" pdeform = "<<c1.deform(b)<<" err = "<<err1<<intersection->sendl;
        }
    }
    return nc;
}

bool FFDDistanceGridDiscreteIntersection::testIntersection(Ray& /*e1*/, FFDDistanceGridCollisionElement& /*e2*/)
{
    return true;
}

int FFDDistanceGridDiscreteIntersection::computeIntersection(Ray& e2, FFDDistanceGridCollisionElement& e1, OutputVector* contacts)
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
        //SReal err1 = 1000.0f;
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
            //err1 = err;
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                detection->baryCoords[0] = Vector3(rayPos,0,0);
                detection->baryCoords[1] = detection->point[1];
#endif
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

