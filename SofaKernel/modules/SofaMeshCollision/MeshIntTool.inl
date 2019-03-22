/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "MeshIntTool.h"
namespace sofa
{
namespace component
{
namespace collision
{


template <class DataTypes>
int MeshIntTool::computeIntersection(TCapsule<DataTypes> & cap, Point & pnt,SReal alarmDist,SReal contactDist,OutputVector* contacts)
{
    DetectionOutput detection;
    if(doCapPointInt(cap,pnt.p(),alarmDist,contactDist, detection))
    {
        detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, pnt);
        contacts->addContact(&detection);
        return 1;
    }

    return 0;
}

template <class DataTypes>
int MeshIntTool::doCapPointInt(TCapsule<DataTypes>& cap, const defaulttype::Vector3& q,SReal alarmDist,SReal contactDist, DetectionOutput& detection){
    const defaulttype::Vector3 p1 = cap.point1();
    const defaulttype::Vector3 p2 = cap.point2();
    const defaulttype::Vector3 AB = p2-p1;
    const defaulttype::Vector3 AQ = q -p1;
    SReal A;
    SReal b;
    A = AB*AB;
    b = AQ*AB;
    SReal cap_rad = cap.radius();

    SReal alpha = 0.5;

    alpha = b/A;//projection of the point on the capsule segment such as the projected point P = p1 + AB * alpha
    if (alpha < 0.0) alpha = 0.0;//if the projection is out the segment, we associate it to a segment apex
    else if (alpha > 1.0) alpha = 1.0;

    defaulttype::Vector3 p,pq;
    p = p1 + AB * alpha;
    pq = q-p;

    SReal enough_to_touch = alarmDist + cap_rad;
    if (pq.norm2() >= enough_to_touch * enough_to_touch)
        return 0;

    //const SReal contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();

    detection.point[0]=p;
    detection.point[1]=q;
    detection.normal = pq;

    detection.value = detection.normal.norm();
    detection.normal /= detection.value;

    detection.value -= (contactDist + cap_rad);

    return 1;
}

template <class DataTypes>
int MeshIntTool::computeIntersection(TCapsule<DataTypes> & cap, Line & lin,SReal alarmDist,SReal contactDist,OutputVector* contacts)
{
    SReal cap_rad = cap.radius();
    const defaulttype::Vector3 p1 = cap.point1();
    const defaulttype::Vector3 p2 = cap.point2();
    const defaulttype::Vector3 q1 = lin.p1();
    const defaulttype::Vector3 q2 = lin.p2();
    DetectionOutput detection;

    if(doCapLineInt(p1,p2,cap_rad,q1,q2,alarmDist,contactDist, detection)){
        //OutputVector::iterator detection = contacts->end()-1;
        //detection.id = cap.getCollisionModel()->getSize() > lin.getCollisionModel()->getSize() ? cap.getIndex() : lin.getIndex();
        //detection.id = cap.getIndex();
        //detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, lin);
        detection.id = cap.getIndex();
        detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, lin);
        contacts->addContact(&detection);
        return 1;
    }

    return 0;
}

template <class DataTypes>
int MeshIntTool::doCapLineInt(TCapsule<DataTypes> & cap,const defaulttype::Vector3 & q1,const defaulttype::Vector3 & q2 ,SReal alarmDist,SReal contactDist, DetectionOutput& detection, bool ignore_p1, bool ignore_p2)
{
    SReal cap_rad = cap.radius();
    const defaulttype::Vector3 p1 = cap.point1();
    const defaulttype::Vector3 p2 = cap.point2();

    return doCapLineInt(p1,p2,cap_rad,q1,q2,alarmDist,contactDist, detection,ignore_p1,ignore_p2);
}


template <class DataTypes>
int MeshIntTool::computeIntersection(TCapsule<DataTypes>& cap, Triangle& tri,SReal alarmDist,SReal contactDist,OutputVector* contacts){
    const int tri_flg = tri.flags();

    int id = cap.getIndex();
    int n1 = 0;
    int n2 = 0;

    const defaulttype::Vector3 cap_p1 = cap.point1();
    const defaulttype::Vector3 cap_p2 = cap.point2();
    SReal cap_rad = cap.radius();
    SReal dist2 = (alarmDist + cap_rad) * (alarmDist + cap_rad);

    const defaulttype::Vector3 tri_p1 = tri.p1();
    const defaulttype::Vector3 tri_p2 = tri.p2();
    const defaulttype::Vector3 tri_p3 = tri.p3();

    SReal substract_dist = contactDist + cap_rad;
    DetectionOutput detection1, detection2;
    n1 = doIntersectionTrianglePoint(dist2,tri_flg,tri_p1,tri_p2,tri_p3,cap_p1,detection1,true);
    n2 = doIntersectionTrianglePoint(dist2,tri_flg,tri_p1,tri_p2,tri_p3,cap_p2,detection2,true);

    if(n1 > 0 && n2 > 0){
        //OutputVector::iterator detection1 = contacts->end() - 2;
        //OutputVector::iterator detection2 = contacts->end() - 1;

        if(detection1.value > detection2.value - 1e-15 && detection1.value < detection2.value + 1e-15){
            detection1.point[0] = (detection1.point[0] + detection2.point[0])/2.0;
            detection1.point[1] = (detection1.point[1] + detection2.point[1])/2.0;
            detection1.normal = (detection1.normal + detection2.normal)/2.0;
            detection1.value = (detection1.value + detection2.value)/2.0 - substract_dist;
            detection1.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);

            //contacts->pop_back();
            contacts->addContact(&detection1);
            //n = 1;
        }
        else{

            //for (OutputVector::iterator detection = contacts->end() - n; detection != contacts->end(); ++detection) {
            //    detection.value -= substract_dist;
            //    detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);
            //    detection.id = id;
            //}

            detection1.value -= substract_dist;
            detection1.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);
            detection1.id = id;
            contacts->addContact(&detection1);

            detection2.value -= substract_dist;
            detection2.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);
            detection2.id = id;
            contacts->addContact(&detection2);
        }
    }
    else{
        //for(OutputVector::iterator detection = contacts->end() - n; detection != contacts->end() ; ++detection){
        //    detection.value -= substract_dist;
        //    detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);
        //    detection.id = id;
        //}
        detection1.value -= substract_dist;
        detection1.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);
        detection1.id = id;
        contacts->addContact(&detection1);

        detection2.value -= substract_dist;
        detection2.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);
        detection2.id = id;
        contacts->addContact(&detection2);
    }

    //int old_n = n;
    int n = 0;
    DetectionOutput detection;

    if (tri_flg&TriangleModel::FLAG_E12)
    {
        if (doCapLineInt(cap_p1, cap_p2, cap_rad, tri_p1, tri_p2, alarmDist, contactDist, detection, !(tri_flg&TriangleModel::FLAG_P1), !(tri_flg&TriangleModel::FLAG_P2)))
        {            
            detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);
            detection.id = id;
            contacts->addContact(&detection);
            ++n;
        }
    }       
    if (tri_flg&TriangleModel::FLAG_E23)
    {
        if (doCapLineInt(cap_p1, cap_p2, cap_rad, tri_p2, tri_p3, alarmDist, contactDist, detection, !(tri_flg&TriangleModel::FLAG_P2), !(tri_flg&TriangleModel::FLAG_P3)) )
        {
            detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);
            detection.id = id;
            contacts->addContact(&detection);
            ++n;
        }
    }        
    if (tri_flg&TriangleModel::FLAG_E31)
    {
        if(doCapLineInt(cap_p1, cap_p2, cap_rad, tri_p3, tri_p1, alarmDist, contactDist, detection, !(tri_flg&TriangleModel::FLAG_P3), !(tri_flg&TriangleModel::FLAG_P1)) )
        {
            detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);
            detection.id = id;
            contacts->addContact(&detection);
            ++n;
        }
    }        

    //for(OutputVector::iterator detection = contacts->end()-n ; detection != contacts->end() ; ++detection){
    //    detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);
    //    detection.id = id;
    //}

    return n + n1 + n2;
}

















}}}
