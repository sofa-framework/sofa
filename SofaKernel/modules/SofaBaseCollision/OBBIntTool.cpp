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
#include <SofaBaseCollision/OBBIntTool.h>

namespace sofa{
namespace component{
namespace collision{


int OBBIntTool::computeIntersection(OBB & box0, OBB & box1,SReal alarmDist,SReal contactDist,OutputVector* contacts){
//    OBB::Real r02 = box0.extent(0)* box0.extent(0) + box0.extent(1)* box0.extent(1) + box0.extent(2)* box0.extent(2);
//    OBB::Real r12 = box1.extent(0)* box1.extent(0) + box1.extent(1)* box1.extent(1) + box1.extent(2)* box1.extent(2);
//    OBB::Real r0 = helper::rsqrt(r02);
//    OBB::Real r1 = helper::rsqrt(r12);
//    if((box0.center() - box1.center()).norm2() > r02 + 2*r0*r1 + r12){
//        return 0;
//    }
    IntrOBBOBB intr(box0,box1);
    //SReal max_time = helper::rsqrt((alarmDist * alarmDist)/((box1.lvelocity() - box0.lvelocity()).norm2()));
    if(/*intr.Find(max_time,box0.lvelocity(),box1.lvelocity())*/intr.Find(alarmDist)){
        defaulttype::Vector3 P0P1(intr.pointOnSecond() - intr.pointOnFirst());
        OBB::Real dist2 = P0P1.norm2();
        if((!intr.colliding()) && dist2 > alarmDist * alarmDist)
            return 0;        

        assert((!intr.colliding()) || (P0P1.cross(intr.separatingAxis()).norm2() < IntrUtil<SReal>::ZERO_TOLERANCE()));

        contacts->resize(contacts->size()+1);
        DetectionOutput *detection = &*(contacts->end()-1);

        if((P0P1.cross(intr.separatingAxis()).norm2() < IntrUtil<SReal>::ZERO_TOLERANCE())){
            detection->normal = intr.separatingAxis();//separatingAxis is normalized while P0P1 is not
        }
        else{
            detection->normal = P0P1;
            detection->normal.normalize();
        }

        detection->point[0] = intr.pointOnFirst();
        detection->point[1] = intr.pointOnSecond();

        if(intr.colliding())
            detection->value = -helper::rsqrt(dist2) - contactDist;
        else
            detection->value = helper::rsqrt(dist2) - contactDist;

        detection->elem.first = box0;
        detection->elem.second = box1;
        detection->id = (box0.getCollisionModel()->getSize() > box1.getCollisionModel()->getSize()) ? box0.getIndex() : box1.getIndex();

        return 1;
    }

    return 0;
}


}
}
}
