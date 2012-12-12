/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
    intersectors.add<CapsuleModel,CapsuleModel, DiscreteIntersection> (this);
    intersectors.add<CapsuleModel,SphereModel, DiscreteIntersection> (this);
    IntersectorFactory::getInstance()->addIntersectors(this);
}

/// Return the intersector class handling the given pair of collision models, or NULL if not supported.
ElementIntersector* DiscreteIntersection::findIntersector(core::CollisionModel* object1, core::CollisionModel* object2, bool& swapModels)
{
    return intersectors.get(object1, object2, swapModels);
}

bool DiscreteIntersection::testIntersection(Cube& cube1, Cube& cube2)
{
    //std::cout<<"CUBE/CUBE testIntersection!!!!!!!!!!!!!!!!!!!!"<<std::endl;

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

bool DiscreteIntersection::testIntersection(Sphere& sph1, Sphere& sph2)
{
    //sout<<"Collision between Sphere - Sphere"<<sendl;
    typename Sphere::Coord sph1Pos(sph1.center());
    typename Sphere::Coord sph2Pos(sph2.center());
    typename Sphere::Real radius1 = sph1.r(), radius2 = sph2.r();
    typename Sphere::Coord tmp = sph1Pos - sph2Pos;
    return (tmp.norm2() < (radius1 + radius2) * (radius1 + radius2));
}

int DiscreteIntersection::computeIntersection(Cube&, Cube&, OutputVector*)
{
    return 0; /// \todo
}


bool DiscreteIntersection::testIntersection( Sphere& sph1, Cube& cube)
{
    //std::cout<<"SPHERE/CUBE testIntersection!!!!!!!!!!!!!!!!!!!!"<<std::endl;
    // Values of the "aligned" bounding box
    Vector3 Bmin = cube.minVect();
    Vector3 Bmax = cube.maxVect();
    // Center of sphere
    Vector3 ctr(sph1.center());
    // Square of radius
    double r2 = sph1.r()*sph1.r();
    // Distance
    double dmin = 0;

    for ( int i = 0; i<3; i++)
    {
        if ( ctr[i] < Bmin[i] )      dmin += (ctr[i]-Bmin[i])*(ctr[i]-Bmin[i]);
        else if ( ctr[i] > Bmax[i] ) dmin += (ctr[i]-Bmax[i])*(ctr[i]-Bmax[i]);
    }

    return (dmin <= r2 );
}

bool DiscreteIntersection::testIntersection(Capsule&, Capsule&){
    //TO DO
    return false;
}

bool DiscreteIntersection::testIntersection(Capsule&, Sphere&){
    //TO DO
    return false;
}

int DiscreteIntersection::computeIntersection(Sphere& sph1, Sphere& sph2, OutputVector* contacts)
{
    double r = sph1.r() + sph2.r();
    Vector3 dist = sph2.center() - sph1.center();

    if (dist.norm2() >= r*r)
        return 0;

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    detection->normal = dist;
    double distSph1Sph2 = detection->normal.norm();
    detection->normal /= distSph1Sph2;
    detection->point[0] = sph1.center() + detection->normal * sph1.r();
    detection->point[1] = sph2.center() - detection->normal * sph2.r();

    detection->value = distSph1Sph2 - r;
    detection->elem.first = sph1;
    detection->elem.second = sph2;
    detection->id = (sph1.getCollisionModel()->getSize() > sph2.getCollisionModel()->getSize()) ? sph1.getIndex() : sph2.getIndex();

    return 1;
}

int DiscreteIntersection::computeIntersection(Sphere& /*sph1*/, Cube& /*cube*/, OutputVector* /*contacts*/)
{
    //to do
    return 0;
}


int DiscreteIntersection::computeIntersection(Capsule & e1,Capsule & e2,OutputVector * contacts){
    return CapsuleIntTool::computeIntersection(e1,e2,getAlarmDistance(),getContactDistance(),contacts);
}


int DiscreteIntersection::computeIntersection(Capsule & cap, Sphere & sph,OutputVector* contacts){
    return CapsuleIntTool::computeIntersection(cap,sph,getAlarmDistance(),getContactDistance(),contacts);
}

//int DiscreteIntersection::computeIntersection(Triangle&, Triangle&, OutputVector*)
//{
//	sout<<"Distance correction between Triangle - Triangle"<<sendl;
//	return 0;
//}

} // namespace collision

} // namespace component

namespace core
{
namespace collision
{
template class SOFA_BASE_COLLISION_API IntersectorFactory<component::collision::DiscreteIntersection>;
}
}

} // namespace sofa

