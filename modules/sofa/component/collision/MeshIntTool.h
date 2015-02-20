#ifndef MESHINTTOOL_H
#define MESHINTTOOL_H
#include <sofa/core/collision/Intersection.h>
#include <sofa/helper/FnDispatcher.h>

#ifndef SOFA_FLAG_SOFAPRO
#include <sofa/component/collision/OBBModel.h>
#include <sofa/component/collision/CapsuleModel.h>
#include <sofa/component/collision/RigidCapsuleModel.h>
#include <sofa/component/collision/IntrTriangleOBB.h>
#endif // SOFA_FLAG_SOFAPRO

#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/LineModel.h>
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

#ifndef SOFA_FLAG_SOFAPRO
    template <class DataTypes>
    static int computeIntersection(TCapsule<DataTypes>& cap, Point& pnt,SReal alarmDist,SReal contactDist,OutputVector* contacts);
    ////!\ CAUTION : uninitialized fields detection->elem and detection->id
    template <class DataTypes>
    static int doCapPointInt(TCapsule<DataTypes>& cap, const Vector3& q,SReal alarmDist,SReal contactDist,OutputVector* contacts);

    template <class DataTypes>
    static int computeIntersection(TCapsule<DataTypes>& cap, Line& lin,SReal alarmDist,SReal contactDist,OutputVector* contacts);

    ////!\ CAUTION : uninitialized fields detection->elem and detection->id
    template <class DataTypes>
    static int doCapLineInt(TCapsule<DataTypes>& cap,const Vector3 & q1,const Vector3 & q2,SReal alarmDist,SReal contactDist,OutputVector* contacts,bool ignore_p1 = false,bool ignore_p2 = false);

#endif // SOFA_FLAG_SOFAPRO

    ////!\ CAUTION : uninitialized fields detection->elem and detection->id and detection->value
    static int doCapLineInt(const Vector3 & p1,const Vector3 & p2,SReal cap_rad,
                         const Vector3 & q1, const Vector3 & q2,SReal alarmDist,SReal contactDist,OutputVector* contacts,bool ignore_p1 = false,bool ignore_p2 = false);

    ////!\ CAUTION : uninitialized fields detection->elem and detection->id and detection->value, you have to substract contactDist, because
    ///this function can be used also as doIntersectionTriangleSphere where the contactDist = getContactDist() + sphere_radius
    static int doIntersectionTrianglePoint(SReal dist2, int flags, const Vector3& p1, const Vector3& p2, const Vector3& p3,const Vector3& q, OutputVector* contacts,bool swapElems = false);

#ifndef SOFA_FLAG_SOFAPRO
    template <class DataTypes>
    static int computeIntersection(TCapsule<DataTypes>& cap, Triangle& tri,SReal alarmDist,SReal contactDist,OutputVector* contacts);

    static int computeIntersection(Triangle& tri,OBB & obb,SReal alarmDist,SReal contactDist,OutputVector* contacts);

    static int computeIntersection(Triangle& tri,int flags,OBB & obb,SReal alarmDist,SReal contactDist,OutputVector* contacts);
#endif // SOFA_FLAG_SOFAPRO

    //SPHERE - POINT
    template <class DataTypes>
    static int computeIntersection(TSphere<DataTypes> & sph, Point& pt,typename DataTypes::Real alarmDist,typename DataTypes::Real contactDist, OutputVector* contacts);

    template <class TReal>
    static int computeIntersection(TSphere<StdVectorTypes<Vec<3,TReal>,Vec<3,TReal>,TReal> > & sph, Point& pt,TReal alarmDist,TReal contactDist, OutputVector* contacts);
    ///

    //LINE - SPHERE
    template <class DataTypes>
    static int computeIntersection(Line& e2, TSphere<DataTypes>& e1,typename DataTypes::Real alarmDist,typename DataTypes::Real contactDist, OutputVector* contacts);

    template <class TReal>
    static int computeIntersection(Line& e2, TSphere<StdVectorTypes<Vec<3,TReal>,Vec<3,TReal>,TReal> >& e1,TReal alarmDist,TReal contactDist, OutputVector* contacts);
    ///

    //TRIANGLE - SPHERE
    template <class DataTypes>
    static int computeIntersection(Triangle& tri, TSphere<DataTypes>& sph,typename DataTypes::Real alarmDist,typename DataTypes::Real contactDist, OutputVector* contacts);

    template <class TReal>
    static int computeIntersection(Triangle& tri, TSphere<StdVectorTypes<Vec<3,TReal>,Vec<3,TReal>,TReal> >& sph,TReal alarmDist,TReal contactDist, OutputVector* contacts);
    ///

    //flags are the flags of the Triangle and p1 p2 p3 its vertices, to_be_projected is the point to be projected on the triangle, i.e.
    //after this method, it will probably be different
    static int projectPointOnTriangle(int flags, const Vector3& p1, const Vector3& p2, const Vector3& p3,Vector3& to_be_projected);

    //returns barycentric coords in alpha and beta so that to_be_projected = (1 - alpha - beta) * p1 + alpha * p2 + beta * p3
    static void triangleBaryCoords(const Vector3& to_be_projected,const Vector3& p1, const Vector3& p2, const Vector3& p3,SReal & alpha,SReal & beta);
};

#ifndef SOFA_FLAG_SOFAPRO
inline int MeshIntTool::computeIntersection(Triangle& tri,OBB & obb,SReal alarmDist,SReal contactDist,OutputVector* contacts){
    return computeIntersection(tri,tri.flags(),obb,alarmDist,contactDist,contacts);
}
#endif // SOFA_FLAG_SOFAPRO

template <class DataTypes>
int MeshIntTool::computeIntersection(TSphere<DataTypes> & e1, Point& e2,typename DataTypes::Real alarmDist,typename DataTypes::Real contactDist, OutputVector* contacts){
    const typename DataTypes::Real myAlarmDist = alarmDist + e1.r();

    typename DataTypes::Coord P,Q,PQ;
    P = e1.center();
    Q = e2.p();
    PQ = Q-P;
    if (PQ.norm2() >= myAlarmDist*myAlarmDist)
        return 0;

    const typename DataTypes::Real  myContactDist = contactDist + e1.r();

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
        detection->normal= typename DataTypes::Coord(1,0,0);
    }
    detection->point[0] = e1.getContactPointByNormal( -detection->normal );

    detection->value -= myContactDist;
    return 1;
}


template <class DataTypes>
int MeshIntTool::computeIntersection(Line& e2, TSphere<DataTypes>& e1,typename DataTypes::Real alarmDist,typename DataTypes::Real contactDist, OutputVector* contacts){
    const typename DataTypes::Real myAlarmDist = alarmDist + e1.r();

    const typename DataTypes::Coord x32 = e2.p1()-e2.p2();
    const typename DataTypes::Coord x31 = e1.center()-e2.p2();
    typename DataTypes::Real A;
    typename DataTypes::Real b;
    A = x32*x32;
    b = x32*x31;

    typename DataTypes::Real alpha = 0.5;
    typename DataTypes::Coord Q;

    if(alpha <= 0){
        Q = e2.p1();
    }
    else if(alpha >= 1){
        Q = e2.p2();
    }
    else{
        Q = e2.p1() - x32 * alpha;
    }

    typename DataTypes::Coord P = e1.center();
    typename DataTypes::Coord QP = P-Q;
    //typename DataTypes::Coord PQ = Q-P;

    if (QP.norm2() >= myAlarmDist*myAlarmDist)
        return 0;

    const typename DataTypes::Real myContactDist = contactDist + e1.r();

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
        detection->normal= typename DataTypes::Coord(1,0,0);
    }
    detection->point[1]=e1.getContactPointByNormal( detection->normal );
    detection->value -= myContactDist;
    return 1;
}



template <class DataTypes>
int MeshIntTool::computeIntersection(Triangle& tri, TSphere<DataTypes>& sph,typename DataTypes::Real alarmDist,typename DataTypes::Real contactDist, OutputVector* contacts){
    const typename DataTypes::Coord sph_center = sph.p();
    typename DataTypes::Coord proj_p = sph_center;
    if(projectPointOnTriangle(tri.flags(),tri.p1(),tri.p2(),tri.p3(),proj_p)){

        typename DataTypes::Coord proj_p_sph_center = sph_center - proj_p;
        typename DataTypes::Real myAlarmDist = alarmDist + sph.r();
        if(proj_p_sph_center.norm2() >= myAlarmDist*myAlarmDist)
            return 0;

        contacts->resize(contacts->size()+1);
        DetectionOutput *detection = &*(contacts->end()-1);
        detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(tri, sph);
        detection->id = sph.getIndex();
        detection->point[0]=proj_p;
        detection->normal = proj_p_sph_center;
        detection->value = detection->normal.norm();
        detection->normal /= detection->value;
        detection->point[1] = sph.getContactPointByNormal( detection->normal );
        detection->value -= (contactDist + sph.r());
    }
    else{
        return 0;
    }
}

#ifndef SOFA_FLAG_SOFAPRO
#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_MESH_COLLISION)
extern template SOFA_MESH_COLLISION_API int MeshIntTool::computeIntersection(TCapsule<Vec3Types>& cap, Point& pnt,SReal alarmDist,SReal contactDist,OutputVector* contacts);
extern template SOFA_MESH_COLLISION_API int MeshIntTool::doCapPointInt(TCapsule<Vec3Types>& cap, const Vector3& q,SReal alarmDist,SReal contactDist,OutputVector* contacts);
extern template SOFA_MESH_COLLISION_API int MeshIntTool::computeIntersection(TCapsule<Vec3Types>& cap, Line& lin,SReal alarmDist,SReal contactDist,OutputVector* contacts);
extern template SOFA_MESH_COLLISION_API int MeshIntTool::doCapLineInt(TCapsule<Vec3Types>& cap,const Vector3 & q1,const Vector3 & q2,SReal alarmDist,SReal contactDist,OutputVector* contacts,bool ignore_p1,bool ignore_p2);
extern template SOFA_MESH_COLLISION_API int MeshIntTool::computeIntersection(TCapsule<Vec3Types>& cap, Triangle& tri,SReal alarmDist,SReal contactDist,OutputVector* contacts);

extern template SOFA_MESH_COLLISION_API int MeshIntTool::computeIntersection(TCapsule<Rigid3Types>& cap, Point& pnt,SReal alarmDist,SReal contactDist,OutputVector* contacts);
extern template SOFA_MESH_COLLISION_API int MeshIntTool::doCapPointInt(TCapsule<Rigid3Types>& cap, const Vector3& q,SReal alarmDist,SReal contactDist,OutputVector* contacts);
extern template SOFA_MESH_COLLISION_API int MeshIntTool::computeIntersection(TCapsule<Rigid3Types>& cap, Line& lin,SReal alarmDist,SReal contactDist,OutputVector* contacts);
extern template SOFA_MESH_COLLISION_API int MeshIntTool::doCapLineInt(TCapsule<Rigid3Types>& cap,const Vector3 & q1,const Vector3 & q2,SReal alarmDist,SReal contactDist,OutputVector* contacts,bool ignore_p1,bool ignore_p2);
extern template SOFA_MESH_COLLISION_API int MeshIntTool::computeIntersection(TCapsule<Rigid3Types>& cap, Triangle& tri,SReal alarmDist,SReal contactDist,OutputVector* contacts);
#endif
#endif // SOFA_FLAG_SOFAPRO

}
}
}
#endif
