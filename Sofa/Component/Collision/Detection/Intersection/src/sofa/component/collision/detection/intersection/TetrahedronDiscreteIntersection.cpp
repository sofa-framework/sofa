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
#include <sofa/component/collision/detection/intersection/TetrahedronDiscreteIntersection.h>

#include <sofa/component/collision/detection/intersection/DiscreteIntersection.h>
#include <sofa/core/collision/Intersection.inl>
#include <sofa/core/collision/IntersectorFactory.h>

namespace sofa::component::collision::detection::intersection
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace sofa::component::collision::geometry;

IntersectorCreator<DiscreteIntersection, TetrahedronDiscreteIntersection> TetrahedronDiscreteIntersectors("Ray");

TetrahedronDiscreteIntersection::TetrahedronDiscreteIntersection(DiscreteIntersection* intersection)
{
    intersection->intersectors.add<TetrahedronCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>,       TetrahedronDiscreteIntersection>  (this);
    intersection->intersectors.add<RayCollisionModel, TetrahedronCollisionModel,         TetrahedronDiscreteIntersection>  (this);
}

bool TetrahedronDiscreteIntersection::testIntersection(Tetrahedron&, Point&, const core::collision::Intersection* currentIntersection)
{
    SOFA_UNUSED(currentIntersection);

    return true;
}

int TetrahedronDiscreteIntersection::computeIntersection(Tetrahedron& e1, Point& e2, OutputVector* contacts, const sofa::core::collision::Intersection*)
{
    const Vec3 n = e2.n();
    if (n == Vec3()) return 0; // no normal on point -> either an internal points or normal is not available

    if (e1.getCollisionModel()->getMechanicalState() == e2.getCollisionModel()->getMechanicalState())
    {
        // self-collisions: make sure the point is not one of the vertices of the tetrahedron
        const int i = e2.getIndex();
        if (i == e1.p1Index() || i == e1.p2Index() || i == e1.p3Index() || i == e1.p4Index())
            return 0;
    }

    const Vec3 P = e2.p();
    Vec3 b0 = e1.getBary(P);
    if (b0[0] < 0 || b0[1] < 0 || b0[2] < 0 || (b0[0]+b0[1]+b0[2]) > 1)
        return 1; // out of tetrahedron

    // Find the point on the surface of the tetrahedron in the direction of -n
    Vec3 bdir = e1.getDBary(-n);
    //msg_info() << "b0 = "<<b0<<" \tbdir = "<<bdir<<sendl;
    double l1 = 1.0e10;
    for (int c=0; c<3; ++c)
    {
        if (bdir[c] < -1.0e-10)
        {
            const double l = -b0[c]/bdir[c];
            if (l < l1) l1 = l;
        }
    }
    // 4th plane : bx+by+bz = 1
    {
        const double bd = bdir[0]+bdir[1]+bdir[2];
        if (bd > 1.0e-10)
        {
            const double l = (1-(b0[0]+b0[1]+b0[2]))/bd;
            if (l < l1) l1 = l;
        }
    }
    if (l1 >= 1.0e9) l1 = 0;
    const double l = l1;
    const Vec3 X = P-n*l;

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

bool TetrahedronDiscreteIntersection::testIntersection(Ray&, Tetrahedron&, const sofa::core::collision::Intersection*)
{
    return true;
}

int TetrahedronDiscreteIntersection::computeIntersection(Ray& e1, Tetrahedron& e2, OutputVector* contacts, const sofa::core::collision::Intersection*)
{
    const Vec3 P = e1.origin();
    Vec3 PQ = e1.direction();
    Vec3 b0 = e2.getBary(P);
    Vec3 bdir = e2.getDBary(PQ);
    //msg_info() << "b0 = "<<b0<<" \tbdir = "<<bdir<<sendl;
    double l0 = 0;
    double l1 = e1.l();
    for (int c=0; c<3; ++c)
    {
        if (b0[c] < 0 && bdir[c] < 0) return 0; // no intersection
        //if (b0[c] > 1 && bdir[c] > 0) return 0; // no intersection
        if (bdir[c] < -1.0e-10)
        {
            const double l = -b0[c]/bdir[c];
            if (l < l1) l1 = l;
        }
        else if (bdir[c] > 1.0e-10)
        {
            const double l = -b0[c]/bdir[c];
            if (l > l0) l0 = l;
        }
    }
    // 4th plane : bx+by+bz = 1
    {
        const double bd = bdir[0]+bdir[1]+bdir[2];
        if (bd > 1.0e-10)
        {
            const double l = (1-(b0[0]+b0[1]+b0[2]))/bd;
            if (l < l1) l1 = l;
        }
        else if (bd < -1.0e-10)
        {
            const double l = (1-(b0[0]+b0[1]+b0[2]))/bd;
            if (l > l0) l0 = l;
        }
    }
    if (l0 > l1) return 0; // empty intersection
    const double l = l0; //(l0+l1)/2;
    const Vec3 X = P+PQ*l;

    //msg_info() << "tetra "<<e2.getIndex()<<": b0 = "<<b0<<" \tbdir = "<<bdir<<sendl;
    //msg_info() << "l0 = "<<l0<<" \tl1 = "<<l1<<" \tX = "<<X<<" \tbX = "<<e2.getBary(X)<<" \t?=? "<<(b0+bdir*l)<<sendl;
    //msg_info() << "b1 = "<<e2.getBary(e2.p1())<<" \nb2 = "<<e2.getBary(e2.p2())<<" \nb3 = "<<e2.getBary(e2.p3())<<" \nb4 = "<<e2.getBary(e2.p4())<<sendl;

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

} // namespace sofa::component::collision::detection::intersection
