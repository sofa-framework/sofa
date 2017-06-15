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
#ifndef SOFA_COMPONENT_COLLISION_LMDNEWPROXIMITYINTERSECTION_INL
#define SOFA_COMPONENT_COLLISION_LMDNEWPROXIMITYINTERSECTION_INL

#include <sofa/helper/system/config.h>
#include <SofaConstraint/LMDNewProximityIntersection.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/proximity.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/collision/Intersection.inl>
#include <iostream>
#include <algorithm>


namespace sofa
{

namespace component
{

namespace collision
{

template< class TFilter1, class TFilter2 >
inline int LMDNewProximityIntersection::doIntersectionLineLine(double dist2, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& q1, const defaulttype::Vector3& q2, OutputVector* contacts, int id, int indexLine1, int indexLine2,  TFilter1 &f1, TFilter2 &f2)
//inline int LMDNewProximityIntersection::doIntersectionLineLine(double dist2, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& q1, const defaulttype::Vector3& q2, OutputVector* contacts, int id)
{

//	std::cout<<"doIntersectionLine "<<indexLine1 <<" and Line "<<indexLine2 <<" is called" <<std::endl;

    bool debug=false;
    if(indexLine1==-1 || indexLine2==-1)
        debug=true;


    const defaulttype::Vector3 AB = p2-p1;
    const defaulttype::Vector3 CD = q2-q1;
    const defaulttype::Vector3 AC = q1-p1;
    sofa::defaulttype::Matrix2 A;
    sofa::defaulttype::Vector2 b;
    A[0][0] = AB*AB;
    A[1][1] = CD*CD;
    A[0][1] = A[1][0] = -CD*AB;
    b[0] = AB*AC;
    b[1] = -CD*AC;
    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

    if (det < -0.000000000001 || det > 0.000000000001)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;

        if (alpha < 0.000001 || alpha > 0.999999 || beta < 0.000001 || beta > 0.999999 )
        {
            if(debug)
                std::cout<<"rejected because of alpha= "<<alpha<<"  or beta= "<<beta<<std::endl;
            return 0;
        }
    }

    defaulttype::Vector3 p,q,pq;
    p = p1 + AB * alpha;
    q = q1 + CD * beta;
    pq = q-p;
    if (pq.norm2() >= dist2)
    {
        if(debug)
            std::cout<<"pq.norm2()= "<<pq.norm2()<<" >= dist2 = "<<dist2<<std::endl;
        return 0;
    }

    if (!f1.validLine(indexLine1, pq))
    {
        if(debug)
            std::cout<<"rejected because of validLine= "<<indexLine1<<" with pq= "<<pq<<std::endl;
        return 0;
    }

    defaulttype::Vector3 qp = p-q;
    if (!f2.validLine(indexLine2, qp))
    {
        if(debug)
            std::cout<<"rejected because of validLine= "<<indexLine2<<" with qp= "<<pq<<std::endl;
        return 0;
    }

    //const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
    contacts->resize(contacts->size()+1);
    sofa::core::collision::DetectionOutput *detection = &*(contacts->end()-1);
    //detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->id = id;
    detection->point[0]=p;
    detection->point[1]=q;
    detection->normal=pq;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    //detection->value -= contactDist;

    if(debug)
        std::cout<<" --------------------------------- ACCEPTED ! --------------------------------"<<pq<<std::endl;
    return 1;
}

template< class TFilter1, class TFilter2 >
inline int LMDNewProximityIntersection::doIntersectionLinePoint(double dist2, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& q, OutputVector* contacts, int id, int indexLine1, int indexPoint2, TFilter1 &f1, TFilter2 &f2, bool swapElems)
//inline int LMDNewProximityIntersection::doIntersectionLinePoint(double dist2, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& q, OutputVector* contacts, int id, bool swapElems)
{
    std::cout<<"doIntersectionLinePoint is called"<<std::endl;
    const defaulttype::Vector3 AB = p2-p1;
    const defaulttype::Vector3 AQ = q -p1;
    double A;
    double b;
    A = AB*AB;
    b = AQ*AB;

    double alpha = 0.5;

    //if (A < -0.000001 || A > 0.000001)
    {
        alpha = b/A;
        //if (alpha < 0.000001 || alpha > 0.999999)
        //        return 0;
        if (alpha < 0.0) alpha = 0.0;
        else if (alpha > 1.0) alpha = 1.0;
    }

    defaulttype::Vector3 p,pq, qp;
    p = p1 + AB * alpha;
    pq = q-p;
    qp = p-q;
    if (pq.norm2() >= dist2)
        return 0;

    if (!f1.validLine(indexLine1, pq))
        return 0;

    if (!f2.validPoint(indexPoint2, qp))
        return 0;



    //const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
    contacts->resize(contacts->size()+1);
    sofa::core::collision::DetectionOutput *detection = &*(contacts->end()-1);

    //detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
    detection->id = id;
    if (swapElems)
    {
        detection->point[0]=q;
        detection->point[1]=p;
        detection->normal = -pq;
    }
    else
    {
        detection->point[0]=p;
        detection->point[1]=q;
        detection->normal = pq;
    }
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    //detection->value -= contactDist;
    return 1;
}

template< class TFilter1, class TFilter2 >
inline int LMDNewProximityIntersection::doIntersectionPointPoint(double dist2, const defaulttype::Vector3& p, const defaulttype::Vector3& q, OutputVector* contacts, int id, int indexPoint1, int indexPoint2, TFilter1 &f1, TFilter2 &f2)
//inline int LMDNewProximityIntersection::doIntersectionPointPoint(double dist2, const defaulttype::Vector3& p, const defaulttype::Vector3& q, OutputVector* contacts, int id)
{
    defaulttype::Vector3 pq;
    pq = q-p;
    if (pq.norm2() >= dist2)
        return 0;

    if (!f1.validPoint(indexPoint1, pq))
        return 0;

    defaulttype::Vector3 qp = p-q;
    if (!f2.validPoint(indexPoint2, qp))
        return 0;

    //const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
    contacts->resize(contacts->size()+1);
    sofa::core::collision::DetectionOutput *detection = &*(contacts->end()-1);
    //detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->id = id;
    detection->point[0]=p;
    detection->point[1]=q;
    detection->normal=pq;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    //detection->value -= contactDist;
    return 1;
}

template< class TFilter1, class TFilter2 >
inline int LMDNewProximityIntersection::doIntersectionTrianglePoint(double dist2, int flags, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& p3, const defaulttype::Vector3& /*n*/, const defaulttype::Vector3& q, OutputVector* contacts, int id,  Triangle &e1, unsigned int *edgesIndices, int indexPoint2, TFilter1 &f1, TFilter2 &f2, bool swapElems)
//inline int LMDNewProximityIntersection::doIntersectionTrianglePoint(double dist2, int flags, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& p3, const defaulttype::Vector3& /*n*/, const defaulttype::Vector3& q, OutputVector* contacts, int id, bool swapElems)
{
    const defaulttype::Vector3 AB = p2-p1;
    const defaulttype::Vector3 AC = p3-p1;
    const defaulttype::Vector3 AQ = q -p1;
    sofa::defaulttype::Matrix2 A;
    sofa::defaulttype::Vector2 b;
    A[0][0] = AB*AB;
    A[1][1] = AC*AC;
    A[0][1] = A[1][0] = AB*AC;
    b[0] = AQ*AB;
    b[1] = AQ*AC;
    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

    if(det==0.0)
    {
        msg_warning("LMDNewProximityIntersection")<<"(doIntersectionTrianglePoint) point is just on the triangle or the triangle do not exists: computation impossible";
        return 0;
    }


    alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
    beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
    defaulttype::Vector3 pq;
    defaulttype::Vector3 p;
    //if (alpha < 0.000001 ||
    //    beta  < 0.000001 ||
    //    alpha + beta  > 0.999999)
    //        return 0;
    if (alpha < 0.000001 || beta < 0.000001 || alpha + beta > 0.999999)
    {
        // nearest point is on an edge or corner
        // barycentric coordinate on AB
        double pAB = b[0] / A[0][0]; // AQ*AB / AB*AB
        // barycentric coordinate on AC
        double pAC = b[1] / A[1][1]; // AQ*AB / AB*AB
        if (pAB < 0.000001 && pAC < 0.0000001)
        {
            ///////////////////////
            // closest point is A
            ///////////////////////
            if (!(flags&TriangleModel::FLAG_P1)) return 0; // this corner is not considered
            alpha = 0.0;
            beta = 0.0;
            //p = p1 + AB * alpha + AC * beta;
            pq = q-p1;
            if (pq.norm2() >= dist2)
                return 0;

            if (!f1.validPoint(e1.p1Index(), pq))
                return 0;


        }
        else if (pAB < 0.999999 && beta < 0.000001)
        {
            ///////////////////////////
            // closest point is on AB : convention edgesIndices 0
            ///////////////////////////
            if (!(flags&TriangleModel::FLAG_E12)) return 0; // this edge is not considered
            alpha = pAB;
            beta = 0.0;
            pq = q-p1 - AB*alpha;// p= p1 + AB * alpha + AC * beta;
            if (pq.norm2() >= dist2)
                return 0;

            if (!f1.validLine(edgesIndices[0], pq))
                return 0;
        }
        else if (pAC < 0.999999 && alpha < 0.000001)
        {
            ///////////////////////////
            // closest point is on AC: convention edgesIndices 2
            ///////////////////////////
            if (!(flags&TriangleModel::FLAG_E31)) return 0; // this edge is not considered
            alpha = 0.0;
            beta = pAC;
            pq = q-p1 - AC*beta;// p= p1 + AB * alpha + AC * beta;
            if (pq.norm2() >= dist2)
                return 0;

            if (!f1.validLine(edgesIndices[2], pq))
                return 0;
        }
        else
        {
            // barycentric coordinate on BC
            // BQ*BC / BC*BC = (AQ-AB)*(AC-AB) / (AC-AB)*(AC-AB) = (AQ*AC-AQ*AB + AB*AB-AB*AC) / (AB*AB+AC*AC-2AB*AC)
            double pBC = (b[1] - b[0] + A[0][0] - A[0][1]) / (A[0][0] + A[1][1] - 2*A[0][1]); // BQ*BC / BC*BC
            if (pBC < 0.000001)
            {
                //////////////////////
                // closest point is B
                //////////////////////
                if (!(flags&TriangleModel::FLAG_P2)) return 0; // this point is not considered
                alpha = 1.0;
                beta = 0.0;
                pq = q-p2;
                if (pq.norm2() >= dist2)
                    return 0;

                if (!f1.validPoint(e1.p2Index(), pq))
                    return 0;
            }
            else if (pBC > 0.999999)
            {
                // closest point is C
                if (!(flags&TriangleModel::FLAG_P3)) return 0; // this point is not considered
                alpha = 0.0;
                beta = 1.0;
                pq = q-p3;
                if (pq.norm2() >= dist2)
                    return 0;

                if (!f1.validPoint(e1.p3Index(), pq))
                    return 0;
            }
            else
            {
                ///////////////////////////
                // closest point is on BC: convention edgesIndices 1
                ///////////////////////////
                if (!(flags&TriangleModel::FLAG_E23)) return 0; // this edge is not considered
                alpha = 1.0-pBC;
                beta = pBC;
                pq = q-p1 - AB * alpha - AC * beta;
                if (pq.norm2() >= dist2)
                    return 0;

                if (!f1.validLine(edgesIndices[1], pq))
                    return 0;
            }
        }
    }
    else
    {
        // nearest point is on the triangle

        p = p1 + AB * alpha + AC * beta;
        pq = q-p;
        if (pq.norm2() >= dist2)
            return 0;

        if (!f1.validTriangle(e1.getIndex(), pq))
            return 0;
    }

    p = p1 + AB * alpha + AC * beta;
    defaulttype::Vector3 qp = p-q;
    if (!f2.validPoint(indexPoint2, qp))
        return 0;

    //const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
    contacts->resize(contacts->size()+1);
    sofa::core::collision::DetectionOutput *detection = &*(contacts->end()-1);
    //detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->id = id;
    if (swapElems)
    {
        detection->point[0]=q;
        detection->point[1]=p;
        detection->normal = -pq;
    }
    else
    {
        detection->point[0]=p;
        detection->point[1]=q;
        detection->normal = pq;
    }
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;

    //printf("\n normale : x = %f , y = %f, z = %f",detection->normal.x(),detection->normal.y(),detection->normal.z());
    //if (e2.getCollisionModel()->isStatic() && detection->normal * e2.n() < -0.95)
    //{ // The elements are interpenetrating
    //	detection->normal = -detection->normal;
    //	detection->value = -detection->value;
    //}
    //detection->value -= contactDist;
    return 1;
}

template<class T>
bool LMDNewProximityIntersection::testIntersection(TSphere<T>& e1, Point& e2)
{
    OutputVector contacts;
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e1.r();

    EmptyFilter emptyFilter;

    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.p(), &contacts, -1, e1.getIndex(), e2.getIndex(), emptyFilter, emptyFilter);

    return (n > 0);
}

template<class T>
int LMDNewProximityIntersection::computeIntersection(TSphere<T>& e1, Point& e2, OutputVector* contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e1.r();

    EmptyFilter emptyFilter;

    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.p(), contacts, (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex(), e1.getIndex(), e2.getIndex(), emptyFilter, emptyFilter);

    if (n > 0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity() + e1.r();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}

template<class T1, class T2>
bool LMDNewProximityIntersection::testIntersection(TSphere<T1>& e1, TSphere<T2>& e2)
{
    OutputVector contacts;
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
    EmptyFilter emptyFilter;

    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.center(), &contacts, -1, e1.getIndex(), e2.getIndex(), emptyFilter, emptyFilter);

    return (n > 0);
}

template<class T1,class T2>
int LMDNewProximityIntersection::computeIntersection(TSphere<T1>& e1, TSphere<T2>& e2, OutputVector* contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
    EmptyFilter emptyFilter;

    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.center(), contacts, (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex(), e1.getIndex(), e2.getIndex(), emptyFilter, emptyFilter);

    if (n > 0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}

template<class T>
bool LMDNewProximityIntersection::testIntersection(Line&, TSphere<T>&)
{
    serr << "Unnecessary call to NewProximityIntersection::testIntersection(Line,Sphere)."<<sendl;
    return true;
}

template<class T>
int LMDNewProximityIntersection::computeIntersection(Line& e1, TSphere<T>& e2, OutputVector* contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e2.r();
    EmptyFilter emptyFilter;
    int id= e2.getIndex();

    int n = doIntersectionLinePoint(alarmDist*alarmDist, e1.p1(),e1.p2(), e2.center(), contacts, id, e1.getIndex(), e2.getIndex(), emptyFilter, emptyFilter);

    if (n > 0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity() + e2.r();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}

template<class T>
bool LMDNewProximityIntersection::testIntersection(Triangle&, TSphere<T>&)
{
    serr << "Unnecessary call to NewProximityIntersection::testIntersection(Triangle,Sphere)."<<sendl;
    return true;
}

template<class T>
int LMDNewProximityIntersection::computeIntersection(Triangle& e1, TSphere<T>& e2, OutputVector* contacts)
{

// index of lines:
    const sofa::helper::fixed_array<unsigned int,3>& edgesInTriangle1 = e1.getCollisionModel()->getTopology()->getEdgesInTriangle(e1.getIndex());
    unsigned int E1edge1verif, E1edge2verif, E1edge3verif;
    E1edge1verif=0; E1edge2verif=0; E1edge3verif=0;



    // verify the edge ordering //
    sofa::core::topology::BaseMeshTopology::Edge edge[3];
    //std::cout<<"E1 & E2 verif: ";
    for (int i=0; i<3; i++)
    {
        // Verify for E1: convention: Edge1 = P1 P2    Edge2 = P2 P3    Edge3 = P3 P1
        edge[i] = e1.getCollisionModel()->getTopology()->getEdge(edgesInTriangle1[i]);
        if(((int)edge[i][0]==e1.p1Index() && (int)edge[i][1]==e1.p2Index()) || ((int)edge[i][0]==e1.p2Index() && (int)edge[i][1]==e1.p1Index()))
        {
            E1edge1verif = edgesInTriangle1[i]; /*std::cout<<"- e1 1: "<<i ;*/
        }
        if(((int)edge[i][0]==e1.p2Index() && (int)edge[i][1]==e1.p3Index()) || ((int)edge[i][0]==e1.p3Index() && (int)edge[i][1]==e1.p2Index()))
        {
            E1edge2verif = edgesInTriangle1[i]; /*std::cout<<"- e1 2: "<<i ;*/
        }
        if(((int)edge[i][0]==e1.p1Index() && (int)edge[i][1]==e1.p3Index()) || ((int)edge[i][0]==e1.p3Index() && (int)edge[i][1]==e1.p1Index()))
        {
            E1edge3verif = edgesInTriangle1[i]; /*std::cout<<"- e1 3: "<<i ;*/
        }
    }

    unsigned int e1_edgesIndex[3];
    e1_edgesIndex[0]=E1edge1verif; e1_edgesIndex[1]=E1edge2verif; e1_edgesIndex[2]=E1edge3verif;


    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e2.r();
    const double dist2 = alarmDist*alarmDist;
    EmptyFilter emptyFilter;

    int id= e2.getIndex();

    int n = doIntersectionTrianglePoint(dist2, e1.flags(),e1.p1(),e1.p2(),e1.p3(),e1.n(), e2.center(), contacts, id, e1, e1_edgesIndex, e2.getIndex(), emptyFilter, emptyFilter);

    if (n > 0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity() + e2.r();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_COLLISION_LMDNEWPROXIMITYINTERSECTION_INL
