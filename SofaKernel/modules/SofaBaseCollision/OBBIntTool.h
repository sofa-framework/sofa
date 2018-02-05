/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef OBBINTTOOL_H
#define OBBINTTOOL_H
#include "config.h"

#include <SofaBaseCollision/OBBModel.h>
#include <SofaBaseCollision/IntrOBBOBB.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <SofaBaseCollision/CapsuleModel.h>
#include <SofaBaseCollision/IntrSphereOBB.h>
namespace sofa{
namespace component{
namespace collision{

class SOFA_BASE_COLLISION_API OBBIntTool{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;
    typedef sofa::core::collision::DetectionOutput DetectionOutput;

    static int computeIntersection(OBB&, OBB&,SReal alarmDist,SReal contactDist,OutputVector* contacts);

    template <class DataTypes>
    static int computeIntersection(TSphere<DataTypes> &sph1, OBB &box,SReal alarmDist,SReal contactDist,OutputVector* contacts);
};

template <class DataTypes>
int OBBIntTool::computeIntersection(TSphere<DataTypes> & sphere,OBB & box,SReal alarmDist,SReal contactDist,OutputVector* contacts){
    TIntrSphereOBB<DataTypes,OBB::DataTypes> intr(sphere,box);
    //double max_time = helper::rsqrt((alarmDist * alarmDist)/((box1.lvelocity() - box0.lvelocity()).norm2()));
    if(/*intr.Find(max_time,box0.lvelocity(),box1.lvelocity())*/intr.Find()){
        OBB::Real dist = intr.distance();
        if((!intr.colliding()) && dist > alarmDist)
            return 0;

        contacts->resize(contacts->size()+1);
        DetectionOutput *detection = &*(contacts->end()-1);

        detection->normal = intr.separatingAxis();
        detection->point[0] = sphere.getContactPointWithSurfacePoint( intr.pointOnFirst() );
        detection->point[1] = intr.pointOnSecond();

        if(intr.colliding())
            detection->value = -dist - contactDist;
        else
            detection->value = dist - contactDist;

        detection->elem.first = sphere;
        detection->elem.second = box;
        //detection->id = (box.getCollisionModel()->getSize() > sphere.getCollisionModel()->getSize()) ? box.getIndex() : sphere.getIndex();
        detection->id = sphere.getIndex();


        return 1;
    }

    return 0;
}


}
}
}
#endif // OBBINTTOOL_H
