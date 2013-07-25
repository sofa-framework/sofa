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
#ifndef SOFA_COMPONENT_COLLISION_DISCRETEINTERSECTION_INL
#define SOFA_COMPONENT_COLLISION_DISCRETEINTERSECTION_INL
#include <sofa/helper/system/config.h>
#include <sofa/component/collision/DiscreteIntersection.h>
#include <sofa/component/collision/SphereModel.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/Intersection.inl>
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




template <class DataTypes1,class DataTypes2>
bool DiscreteIntersection::testIntersection(TSphere<DataTypes1>& sph1, TSphere<DataTypes2>& sph2)
{
    return BaseIntTool::testIntersection( sph1, sph2, getAlarmDistance() );
}

template <class DataTypes>
bool DiscreteIntersection::testIntersection( TSphere<DataTypes>& sph1, Cube& cube )
{
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


template <class DataTypes1,class DataTypes2>
int DiscreteIntersection::computeIntersection(TSphere<DataTypes1>& sph1, TSphere<DataTypes2>& sph2, OutputVector* contacts)
{
    return BaseIntTool::computeIntersection(sph1,sph2,getAlarmDistance(),getContactDistance(),contacts);
}

template <class DataTypes>
bool DiscreteIntersection::testIntersection(Capsule&, TSphere<DataTypes>&){
    //TO DO
    return false;
}

template <class DataTypes>
int DiscreteIntersection::computeIntersection(Capsule & cap, TSphere<DataTypes> & sph,OutputVector* contacts){
    return CapsuleIntTool::computeIntersection(cap,sph,getAlarmDistance(),getContactDistance(),contacts);
}

template <class DataTypes>
bool DiscreteIntersection::testIntersection(TSphere<DataTypes> &,OBB &){
    return false;
}

template <class DataTypes>
int DiscreteIntersection::computeIntersection(TSphere<DataTypes> & sph, OBB & box,OutputVector* contacts){
    return OBBIntTool::computeIntersection(sph,box,getAlarmDistance(),getContactDistance(),contacts);
}



} // namespace collision

} // namespace component

} // namespace sofa

#endif
