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
#ifndef BASEINTTOOL_H
#define BASEINTTOOL_H
#include "config.h"

#include <sofa/core/collision/Intersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <SofaBaseCollision/CapsuleModel.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaBaseCollision/OBBModel.h>
#include <SofaBaseCollision/IntrCapsuleOBB.h>
#include <SofaBaseCollision/CapsuleIntTool.h>
#include <SofaBaseCollision/OBBIntTool.h>
#include <cmath>

namespace sofa
{
namespace component
{
namespace collision
{

class SOFA_BASE_COLLISION_API BaseIntTool : public CapsuleIntTool,public OBBIntTool{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    template <class Elem1,class Elem2>
    static bool testIntersection(Elem1&,Elem2&,SReal){
        msg_info("BaseIntTool")<<"testIntersection should not be used with theese types";
        return false;
    }

    static bool testIntersection(Cube &cube1, Cube &cube2,SReal alarmDist);


    template <class DataTypes1,class DataTypes2>
    static bool testIntersection(TSphere<DataTypes1>& sph1, TSphere<DataTypes2>& sph2,SReal alarmDist)
    {
        typename TSphere<DataTypes1>::Real r = sph1.r() + sph2.r() + alarmDist;
        return ( sph1.center() - sph2.center() ).norm2() <= r*r;
    }





    template <class DataTypes1,class DataTypes2>
    static int computeIntersection(TSphere<DataTypes1>& sph1, TSphere<DataTypes2>& sph2,SReal alarmDist,SReal contactDist,OutputVector* contacts)
    {
        SReal r = sph1.r() + sph2.r();
        SReal myAlarmDist = alarmDist + r;
        defaulttype::Vector3 dist = sph2.center() - sph1.center();
        SReal norm2 = dist.norm2();

        if (norm2 > myAlarmDist*myAlarmDist)
            return 0;

        contacts->resize(contacts->size()+1);
        DetectionOutput *detection = &*(contacts->end()-1);
        SReal distSph1Sph2 = helper::rsqrt(norm2);
        detection->normal = dist / distSph1Sph2;
        detection->point[0] = sph1.getContactPointByNormal( -detection->normal );
        detection->point[1] = sph2.getContactPointByNormal( detection->normal );

        detection->value = distSph1Sph2 - r - contactDist;
        detection->elem.first = sph1;
        detection->elem.second = sph2;
        detection->id = (sph1.getCollisionModel()->getSize() > sph2.getCollisionModel()->getSize()) ? sph1.getIndex() : sph2.getIndex();

        return 1;
    }



    template <class DataTypes1,class DataTypes2>
    inline static int computeIntersection(TCapsule<DataTypes1> &c1, TCapsule<DataTypes2> &c2, SReal alarmDist, SReal contactDist, OutputVector *contacts){
        return CapsuleIntTool::computeIntersection(c1,c2,alarmDist,contactDist,contacts);
    }

    template <class DataTypes1,class DataTypes2>
    inline static int computeIntersection(TCapsule<DataTypes1> &cap, TSphere<DataTypes2> &sph, SReal alarmDist, SReal contactDist, OutputVector *contacts){
        return CapsuleIntTool::computeIntersection(cap,sph,alarmDist,contactDist,contacts);
    }

    template <class DataTyes>
    inline static int computeIntersection(TCapsule<DataTyes> &cap, OBB & obb, SReal alarmDist, SReal contactDist, OutputVector *contacts){
        return CapsuleIntTool::computeIntersection(cap,obb,alarmDist,contactDist,contacts);
    }

    inline static int computeIntersection(OBB &obb0, OBB &obb1, SReal alarmDist, SReal contactDist, OutputVector *contacts){
        return OBBIntTool::computeIntersection(obb0,obb1,alarmDist,contactDist,contacts);
    }

    template <class DataType>
    inline static int computeIntersection(TSphere<DataType> &sph, OBB &obb, SReal alarmDist, SReal contactDist, OutputVector *contacts){
        return OBBIntTool::computeIntersection(sph,obb,alarmDist,contactDist,contacts);
    }

    inline static int computeIntersection(Cube&, Cube&, SReal, SReal, OutputVector *){
        return 0;
    }


};





}
}
}
#endif // BASEINTTOOL_H
