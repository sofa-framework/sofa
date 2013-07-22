#ifndef MESHINTTOOL_H
#define MESHINTTOOL_H
#include <sofa/core/collision/Intersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <sofa/component/collision/OBBModel.h>
#include <sofa/component/collision/CapsuleModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/IntrTriangleOBB.h>
#include <sofa/component/collision/SphereModel.h>

namespace sofa
{
namespace component
{
namespace collision
{


class SOFA_MESH_COLLISION_API MeshIntTool
{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;
    typedef sofa::core::collision::DetectionOutput DetectionOutput;

    static int computeIntersection(Capsule& cap, Point& pnt,double alarmDist,double contactDist,OutputVector* contacts);
    ////!\ CAUTION : uninitialized fields detection->elem and detection->id
    static int doCapPointInt(Capsule& cap, const Vector3& q,double alarmDist,double contactDist,OutputVector* contacts);

    static int computeIntersection(Capsule& cap, Line& lin,double alarmDist,double contactDist,OutputVector* contacts);

    ////!\ CAUTION : uninitialized fields detection->elem and detection->id
    static int doCapLineInt(Capsule& cap,const Vector3 & q1,const Vector3 & q2,double alarmDist,double contactDist,OutputVector* contacts,bool ignore_p1 = false,bool ignore_p2 = false);

    ////!\ CAUTION : uninitialized fields detection->elem and detection->id and detection->value
    static int doCapLineInt(const Vector3 & p1,const Vector3 & p2,double cap_rad,
                         const Vector3 & q1, const Vector3 & q2,double alarmDist,double contactDist,OutputVector* contacts,bool ignore_p1 = false,bool ignore_p2 = false);

    ////!\ CAUTION : uninitialized fields detection->elem and detection->id and detection->value, you have to substract contactDist, because
    ///this function can be used also as doIntersectionTriangleSphere where the contactDist = getContactDist() + sphere_radius
    static int doIntersectionTrianglePoint(double dist2, int flags, const Vector3& p1, const Vector3& p2, const Vector3& p3,const Vector3& q, OutputVector* contacts,bool swapElems = false);

    static int computeIntersection(Capsule& cap, Triangle& tri,double alarmDist,double contactDist,OutputVector* contacts);

    static int computeIntersection(Triangle& tri,OBB & obb,double alarmDist,double contactDist,OutputVector* contacts);

    static int computeIntersection(Triangle& tri,int flags,OBB & obb,double alarmDist,double contactDist,OutputVector* contacts);

    //SPHERE - POINT
    template <class DataTypes>
    static int computeIntersection(TSphere<DataTypes> & sph, Point& pt,double alarmDist,double contactDist, OutputVector* contacts);

    template <class TReal>
    static int computeIntersection(TSphere<StdVectorTypes<Vec<3,TReal>,Vec<3,TReal>,TReal> > & sph, Point& pt,double alarmDist,double contactDist, OutputVector* contacts);
    ///

    //LINE - SPHERE
    template <class DataTypes>
    static int computeIntersection(Line& e2, TSphere<DataTypes>& e1,double alarmDist,double contactDist, OutputVector* contacts);

    template <class TReal>
    static int computeIntersection(Line& e2, TSphere<StdVectorTypes<Vec<3,TReal>,Vec<3,TReal>,TReal> >& e1,double alarmDist,double contactDist, OutputVector* contacts);
    ///

    //TRIANGLE - SPHERE
    template <class DataTypes>
    static int computeIntersection(Triangle& tri, TSphere<DataTypes>& sph,double alarmDist,double contactDist, OutputVector* contacts);

    template <class TReal>
    static int computeIntersection(Triangle& tri, TSphere<StdVectorTypes<Vec<3,TReal>,Vec<3,TReal>,TReal> >& sph,double alarmDist,double contactDist, OutputVector* contacts);
    ///

    //flags are the flags of the Triangle and p1 p2 p3 its vertices, to_be_projected is the point to be projected on the triangle, i.e.
    //after this method, it will probably be different
    static int projectPointOnTriangle(int flags, const Vector3& p1, const Vector3& p2, const Vector3& p3,Vector3& to_be_projected);
};

inline int MeshIntTool::computeIntersection(Triangle& tri,OBB & obb,double alarmDist,double contactDist,OutputVector* contacts){
    return computeIntersection(tri,tri.flags(),obb,alarmDist,contactDist,contacts);
}

template <class DataTypes>
int MeshIntTool::computeIntersection(TSphere<DataTypes> & e1, Point& e2,double alarmDist,double contactDist, OutputVector* contacts){
    const double myAlarmDist = alarmDist + e1.r();

    Vector3 P,Q,PQ;
    P = e1.center();
    Q = e2.p();
    PQ = Q-P;
    if (PQ.norm2() >= myAlarmDist*myAlarmDist)
        return 0;

    const double myContactDist = contactDist + e1.r();

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->id = (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex();
    detection->point[1]=Q;
    detection->normal=PQ;
    detection->value = detection->normal.norm();
    if(detection->value>1e-15)
    {
        detection->normal /= detection->value;
    }
    else
    {
        //intersection->serr<<"WARNING: null distance between contact detected"<<intersection->sendl;
        detection->normal= Vector3(1,0,0);
    }
    detection->point[0] = e1.getContactPoint( detection->normal );

    detection->value -= myContactDist;
    return 1;
}


template <class DataTypes>
int MeshIntTool::computeIntersection(Line& e2, TSphere<DataTypes>& e1,double alarmDist,double contactDist, OutputVector* contacts){
    const double myAlarmDist = alarmDist + e1.r();

    const Vector3 x32 = e2.p1()-e2.p2();
    const Vector3 x31 = e1.center()-e2.p2();
    double A;
    double b;
    A = x32*x32;
    b = x32*x31;

    double alpha = 0.5;
    Vector3 Q;

    if(alpha <= 0){
        Q = e2.p1();
    }
    else if(alpha >= 1){
        Q = e2.p2();
    }
    else{
        Q = e2.p1() - x32 * alpha;
    }

    Vector3 P = e1.center();
    Vector3 QP = P-Q;
    //Vector3 PQ = Q-P;

    if (QP.norm2() >= myAlarmDist*myAlarmDist)
        return 0;

    const double myContactDist = contactDist + e1.r();

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
    detection->id = e1.getIndex();
    detection->point[0]=Q;
    detection->normal=QP;
    detection->value = detection->normal.norm();
    if(detection->value>1e-15)
    {
        detection->normal /= detection->value;
    }
    else
    {
        //intersection->serr<<"WARNING: null distance between contact detected"<<intersection->sendl;
        detection->normal= Vector3(1,0,0);
    }
    detection->point[1]=e1.getContactPoint( detection->normal );
    detection->value -= myContactDist;
    return 1;
}



template <class DataTypes>
int MeshIntTool::computeIntersection(Triangle& tri, TSphere<DataTypes>& sph,double alarmDist,double contactDist, OutputVector* contacts){
    const Vector3 sph_center = sph.p();
    Vector3 proj_p = sph_center;
    if(projectPointOnTriangle(tri.flags(),tri.p1(),tri.p2(),tri.p3(),proj_p)){

        Vector3 proj_p_sph_center = sph_center - proj_p;
        double myAlarmDist = alarmDist + sph.r();
        if(proj_p_sph_center.norm2() >= myAlarmDist*myAlarmDist)
            return 0;

        //const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
        contacts->resize(contacts->size()+1);
        DetectionOutput *detection = &*(contacts->end()-1);
        detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(tri, sph);
        detection->id = sph.getIndex();
        detection->point[0]=proj_p;
        detection->normal = proj_p_sph_center;
        detection->value = detection->normal.norm();
        detection->normal /= detection->value;
        detection->point[1] = sph.getContactPoint( detection->normal );
        detection->value -= (contactDist + sph.r());
    }
    else{
        return 0;
    }
}


}
}
}
#endif
