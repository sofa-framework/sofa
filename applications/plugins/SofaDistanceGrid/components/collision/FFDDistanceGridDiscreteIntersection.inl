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
#ifndef SOFA_COMPONENT_COLLISION_FFDDISTANCEGRIDDISCRETEINTERSECTION_INL
#define SOFA_COMPONENT_COLLISION_FFDDISTANCEGRIDDISCRETEINTERSECTION_INL
#include <SofaDistanceGrid/config.h>

#include <iostream>
#include <algorithm>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/proximity.h>

#include "FFDDistanceGridDiscreteIntersection.h"

namespace sofa
{

namespace component
{

namespace collision
{


template<class T>
bool FFDDistanceGridDiscreteIntersection::testIntersection(FFDDistanceGridCollisionElement&, TSphere<T>&)
{
    return true;
}

template<class T>
int FFDDistanceGridDiscreteIntersection::computeIntersection(FFDDistanceGridCollisionElement& e1, TSphere<T>& e2, OutputVector* contacts)
{

    DistanceGrid* grid1 = e1.getGrid();
    FFDDistanceGridCollisionModel::DeformedCube& c1 = e1.getCollisionModel()->getDeformCube(e1.getIndex());

    const double d0 = e1.getProximity() + e2.getProximity() + intersection->getContactDistance() + e2.r();
    const SReal margin = 0.001f + (SReal)d0;

    c1.updateFaces();
    const SReal cubesize = c1.invDP.norm();
    defaulttype::Vector3 p2 = e2.center();
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
                    sofa::core::collision::DetectionOutput *detection = &*(contacts->end()-1);
                    detection->normal = defaulttype::Vector3(grad); // normal in global space from p1's surface
                    detection->value = d - d0;
                    detection->elem.first = e1;
                    detection->elem.second = e2;
                    detection->id = e2.getIndex();
                    detection->point[0] = defaulttype::Vector3(pinit);
                    detection->point[1] = e2.getContactPointWithSurfacePoint( pinit );
                    return 1;
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

    return 0;
}


} // namespace collision

} // namespace component

} // namespace sofa

#endif
