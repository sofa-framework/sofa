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
#ifndef SOFA_COMPONENT_COLLISION_LOCALMINDISTANCE_INL
#define SOFA_COMPONENT_COLLISION_LOCALMINDISTANCE_INL

#include <sofa/helper/system/config.h>
#include <SofaConstraint/LocalMinDistance.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/proximity.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/collision/Intersection.inl>
#include <iostream>
#include <algorithm>
#include <sofa/helper/gl/template.h>

#include <sofa/simulation/Node.h>

#include <iostream>
#include <algorithm>


#define DYNAMIC_CONE_ANGLE_COMPUTATION

namespace sofa
{

namespace component
{

namespace collision
{

typedef BaseMeshTopology::PointID			PointID;

#ifdef SOFA_DEV

//Copy of Line computation. TODO_Spline : finding adaptive and optimized computation for Spline
template<int FLAG>
bool LocalMinDistance::testValidity(CubicBezierCurve<FLAG>& spline, const Vector3& /*PQ*/)
{
    return spline.isActivated();
}


template<int FLAG>
bool LocalMinDistance::testIntersection(CubicBezierCurve<FLAG>& e2, Point& e1)
{
    if(!e1.activated(e2.getCollisionModel()) || !e2.activated(e1.getCollisionModel()) || !e2.isActivated())
        return false;

    bool isCollided=false;
    Vector3 AB;
    Vector3 AP;
    Vector3 P,Q,PQ;
    double alpha;

    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e2.r();

    double A;
    double b;

    if(e2.isStraightLine() )//In the case where e2 is tagged as a straight line, computation behaves exactly as a lineModel
    {
        AB = e2[3]-e2[0];
        AP = e1.p()-e2[0];
        A = AB*AB;
        b = AP*AB;

        alpha = 0.5;

        //if (A < -0.000001 || A > 0.000001)
        {
            alpha = b/A;
            if (alpha < 0.000001 || alpha > 0.999999)
                return false;
        }

        P = e1.p();
        Q = e2[0] + AB * alpha;
        PQ = Q-P;

        if (PQ.norm2() < alarmDist*alarmDist)
        {
            // filter for LMD

            if (!useLMDFilters.getValue())
            {
                if (!testValidity(e1, PQ))
                    return false;

                Vector3 QP = -PQ;
                return testValidity(e2, QP);
            }
            else
            {
                return true;
            }// end filter
        }
        else
            return false;
    }
    else if( e2.isCubicBezier () )
    {
        //In the case where e2 is tagged as a cubic curve, computation behaves discretly as 3 Lines
        //Todo implementation for continuous curve
        for(int i=0; i<3; i++)
        {
            AB = e2[i+1]-e2[i];
            AP = e1.p()-e2[i];
            A = AB*AB;
            b = AP*AB;
            double _alpha = b/A;
            if (_alpha < 0.000001 || _alpha > 0.999999)
                continue;

            alpha = _alpha;
            P = e1.p();
            Q = e2[i] + AB * alpha;
            PQ = Q-P;

            if (PQ.norm2() < alarmDist*alarmDist)
            {
                // filter for LMD

                if (!useLMDFilters.getValue())
                {
                    if (!testValidity(e1, PQ))
                        continue;

                    Vector3 QP = -PQ;
                    if (!testValidity(e2, QP))
                        continue;
                    else
                        isCollided =true;
                }
                else
                {
                    isCollided = true;
                }// end filter
            }
            else
                continue;
        }
    }
    return isCollided;
}

template<int FLAG>
int LocalMinDistance::computeIntersection(CubicBezierCurve<FLAG>& e2, Point& e1, OutputVector* contacts)
{
    if(!e1.activated(e2.getCollisionModel()) || !e2.activated(e1.getCollisionModel()))
        return 0;

    Vector3 AB;
    Vector3 AP;
    Vector3 P, Q, PQ, QP;
    double alpha;
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e2.r();
    double A;
    double b;

    if(e2.isStraightLine() )//In the case where e2 is tagged as a straight line, computation behaves exactly as a lineModel
    {
        AB = e2[3]-e2[0];
        AP = e1.p()-e2[0];
        A = AB*AB;
        b = AP*AB;

        alpha = 0.5;

        //if (A < -0.000001 || A > 0.000001)
        {
            alpha = b/A;
            if (alpha < 0.000001 || alpha > 0.999999)
                return 0;
        }

        P = e1.p();
        Q = e2[0] + AB * alpha;
        PQ = Q - P;
        QP = -PQ;

        if (PQ.norm2() >= alarmDist*alarmDist)
            return 0;

        if (!useLMDFilters.getValue())
        {
            if (!testValidity(e1, PQ))
                return 0;

            if (!testValidity(e2, QP))
                return 0;
        }
        else
        {

        }// end filter
        contacts->resize(contacts->size()+1);
        DetectionOutput *detection = &*(contacts->end()-1);

        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();

        //detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
        detection->elem.first = e2;
        detection->elem.second = e1;
        detection->id = e1.getIndex();
        detection->point[0]=Q;
        detection->point[1]=P;
        detection->normal=QP;
        detection->value = detection->normal.norm();
        detection->normal /= detection->value;
        detection->value -= contactDist;
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
        //alpha here is the barycentric coordinate of the local spline (littler)
        //need to be transformed to the barycentric coordinate of the globale spline on the Edge
        double t=(double)e2.getGlobal_t_(alpha);// alpha*=(e2.t0() - e2.t1());alpha+=e2.t0();
        detection->baryCoords[0][0]=t;

        typename CubicBezierCurve<FLAG>::Quat qt = e2.quat(alpha);
        Vector3 e_X = qt.rotate(Vector3(0.,1.,0.));
        Vector3 e_Y = qt.rotate(Vector3(0.,0.,1.));
        double projectX = dot(QP,e_X);
        double projectY = dot(QP,e_Y);

        detection->baryCoords[0][1]=projectX;
        detection->baryCoords[0][2]=projectY;
#endif
    }
    else if ( e2.isCubicBezier () )
    {
        //In the case where e2 is tagged as a cubic curve, computation behaves discretly as 3 Lines
        //Todo implementation for continuous curve
        for(int i=0; i<3; i++)
        {
            AB = e2[i+1]-e2[i];
            AP = e1.p()-e2[i];
            A = AB*AB;
            b = AP*AB;

            double _alpha = 0.5;
            _alpha = b/A;
            if (_alpha < 0.000001 || _alpha > 0.999999)
                continue;
            alpha = _alpha;

            P = e1.p();
            Q = e2[i] + AB * alpha;
            PQ = Q - P;
            QP = -PQ;

            if (PQ.norm2() >= alarmDist*alarmDist)
                return 0;

            if (!useLMDFilters.getValue())
            {
                if (!testValidity(e1, PQ))
                    return 0;

                if (!testValidity(e2, QP))
                    return 0;
            }
            else
            {

            }// end filter
            contacts->resize(contacts->size()+1);
            DetectionOutput *detection = &*(contacts->end()-1);

            const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();

            //detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
            detection->elem.first = e2;
            detection->elem.second = e1;
            detection->id = e1.getIndex();
            detection->point[0]=Q;
            detection->point[1]=P;
            detection->normal=QP;
            detection->value = detection->normal.norm();
            detection->normal /= detection->value;
            detection->value -= contactDist;
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
            //alpha here is the barycentric coordinate of the local spline (littler)
            //need to be transformed to the barycentric coordinate of the globale spline on the Edge
            double t=(double)e2.getGlobal_t_(alpha);// alpha*=(e2.t0() - e2.t1());alpha+=e2.t0();
            detection->baryCoords[0][0]=t;

            typename CubicBezierCurve<FLAG>::Quat qt = e2.quat(alpha);
            Vector3 e_X = qt.rotate(Vector3(0.,1.,0.));
            Vector3 e_Y = qt.rotate(Vector3(0.,0.,1.));
            double projectX = dot(QP,e_X);
            double projectY = dot(QP,e_Y);

            detection->baryCoords[0][1]=projectX;
            detection->baryCoords[0][2]=projectY;
#endif
            std::cout<<contacts->size()<<" contacts.size() LocalMinDistance::1607  "<<"e1.getIndex() " <<e1.getIndex() <<"  e2.getIndex()" <<e2.getIndex()
                    << " t0:"<< e2.t0()<<"   alpha:"<<alpha <<"  t1:" << e2.t1()<<"  t:"<<t
                    <<"     P : " <<P <<"   Q : " <<Q <<std::endl;//////////////////////////////////
        }
    }


    return 1;
}

/////////////////////////////////////////////////////////////////
template<int FLAG>
bool LocalMinDistance::testIntersection(CubicBezierCurve<FLAG>& e2, Sphere& e1)
{
    if(!e2.activated(e1.getCollisionModel()) || !e2.isActivated())
        return false;

    bool isCollided=false;
    Vector3 x32;
    Vector3 x31;
    Vector3 P,Q,PQ;
    double alpha;

    const double alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();
    double A;
    double b;

    if(e2.isStraightLine() )//In the case where e2 is tagged as a straight line, computation behaves exactly as a lineModel
    {

        x32 = e2[0]-e2[3];
        x31 = e1.center()-e2[3];
        A = x32*x32;
        b = x32*x31;

        alpha = 0.5;
        //if (A < -0.000001 || A > 0.000001)
        {
            alpha = b/A;
            if (alpha < 0.000001 || alpha > 0.999999)
                return false;
        }


        P = e1.center();
        Q = e2[0] - x32 * alpha;
        PQ = Q-P;

        if (PQ.norm2() < alarmDist*alarmDist)
        {
            return true;
        }
        else
            return false;

    }
    else if( e2.isCubicBezier () )
    {
        //In the case where e2 is tagged as a cubic curve, computation behaves discretly as 3 Lines
        //Todo implementation for continuous curve
        for(int i=0; i<3; i++)
        {
            x32 = e2[i]-e2[i+1];
            x31 = e1.center()-e2[i+1];
            A = x32*x32;
            b = x32*x31;

            alpha = 0.5;
            alpha = b/A;
            if (alpha < 0.000001 || alpha > 0.999999)
                continue;

            P = e1.center();
            Q = e2[i] - x32 * alpha;
            PQ = Q-P;

            if (PQ.norm2() < alarmDist*alarmDist)
            {
                isCollided = true;
            }
            else
                continue;
        }
    }
    return isCollided;
}
template<int FLAG>
int LocalMinDistance::computeIntersection(CubicBezierCurve<FLAG>& e2, Sphere& e1, OutputVector* contacts)
{

    if(!e2.activated(e1.getCollisionModel()) || !e2.isActivated() )
        return 0;

    Vector3 x32;
    Vector3 x31;
    Vector3 P, Q, PQ, QP;
    double alpha;
    const double alarmDist = getAlarmDistance() + e1.r() + e2.r() + e1.getProximity() + e2.getProximity();

    double A;
    double b;


    if(e2.isStraightLine() )//In the case where e2 is tagged as a straight line, computation behaves exactly as a lineModel
    {

        x32 = e2[0]-e2[3];
        x31 = e1.center()-e2[3];
        A = x32*x32;
        b = x32*x31;

        alpha = 0.5;

        //if (A < -0.000001 || A > 0.000001)
        {
            alpha = b/A;
            if (alpha < 0.000001 || alpha > 0.999999)
                return 0;
        }

        P = e1.center();
        Q = e2[0] - x32 * alpha;
        QP = P-Q;
        PQ = Q-P;

        if (QP.norm2() >= alarmDist*alarmDist)
            return 0;

        const double contactDist = getContactDistance() + e1.r() + e1.getProximity() + e2.getProximity();

        contacts->resize(contacts->size()+1);
        DetectionOutput *detection = &*(contacts->end()-1);
        //detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
        detection->elem.first = e2;
        detection->elem.second = e1;
        detection->id = e1.getIndex();
        detection->point[0]=Q;
        detection->point[1]=P;
        detection->normal=QP;
        detection->value = detection->normal.norm();
        detection->normal /= detection->value;
        detection->value -= contactDist;

#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
        //alpha here is the barycentric coordinate of the local spline (littler)
        //need to be transformed to the barycentric coordinate of the globale spline on the Edge
        double t=(double)e2.getGlobal_t_(alpha);// alpha*=(e2.t0() - e2.t1());alpha+=e2.t0();
        detection->baryCoords[0][0]=t;

        typename CubicBezierCurve<FLAG>::Quat qt = e2.quat(alpha);
        Vector3 e_X = qt.rotate(Vector3(0.,1.,0.));
        Vector3 e_Y = qt.rotate(Vector3(0.,0.,1.));
        double projectX = dot(QP,e_X);
        double projectY = dot(QP,e_Y);

        detection->baryCoords[0][1]=projectX;
        detection->baryCoords[0][2]=projectY;
#endif
    }
    else if( e2.isCubicBezier () )
    {
        //In the case where e2 is tagged as a cubic curve, computation behaves discretly as 3 Lines
        //Todo implementation for continuous curve
        for(int i=0; i<3; i++)
        {

            x32 = e2[i]-e2[i+1];
            x31 = e1.center()-e2[i+1];
            A = x32*x32;
            b = x32*x31;

            alpha = 0.5;

            //if (A < -0.000001 || A > 0.000001)
            {
                alpha = b/A;
                if (alpha < 0.000001 || alpha > 0.999999)
                    continue;
            }

            P = e1.center();
            Q = e2[i] - x32 * alpha;
            QP = P-Q;
            PQ = Q-P;

            if (QP.norm2() >= alarmDist*alarmDist)
                continue;

            const double contactDist = getContactDistance() +e2.r() +e1.r() + e1.getProximity() + e2.getProximity();

            contacts->resize(contacts->size()+1);
            DetectionOutput *detection = &*(contacts->end()-1);
            //detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
            detection->elem.first = e2;
            detection->elem.second = e1;
            detection->id = e1.getIndex();
            detection->point[0]=Q;
            detection->point[1]=P;
            detection->normal=QP;
            detection->value = detection->normal.norm();
            detection->normal /= detection->value;
            detection->value -= contactDist;

#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
            //alpha here is the barycentric coordinate of the local spline (littler)
            //need to be transformed to the barycentric coordinate of the globale spline on the Edge
            double t=(double)e2.getGlobal_t_(alpha);// alpha*=(e2.t0() - e2.t1());alpha+=e2.t0();
            detection->baryCoords[0][0]=t;

            typename CubicBezierCurve<FLAG>::Quat qt = e2.quat(alpha);
            Vector3 e_X = qt.rotate(Vector3(0.,1.,0.));
            Vector3 e_Y = qt.rotate(Vector3(0.,0.,1.));
            double projectX = dot(QP,e_X);
            double projectY = dot(QP,e_Y);

            detection->baryCoords[0][1]=projectX;
            detection->baryCoords[0][2]=projectY;
#endif
            std::cout<<contacts->size()<<" contacts.size() LocalMinDistance::1706  "<<"e1.getIndex() " <<e1.getIndex() <<"  e2.getIndex()" <<e2.getIndex()
                    << " t0:"<< e2.t0()<<"   alpha:"<<alpha <<"  t1:" << e2.t1()<<"  t:"<<t
                    <<"     P : " <<P <<"   Q : " <<Q <<std::endl;//////////////////////////////////

        }
    }





    return 1;
}

#endif // SOFA_DEV




} // namespace collision

} // namespace component

} // namespace sofa


#endif /* SOFA_COMPONENT_COLLISION_LOCALMINDISTANCE_INL */
