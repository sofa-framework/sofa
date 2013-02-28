// File modified from GeometricTools
// http://www.geometrictools.com/


#ifndef WM5INTRUTILITY3_H
#define WM5INTRUTILITY3_H
#include <sofa/defaulttype/Vec.h>

namespace sofa{
namespace component{
namespace collision{

using namespace sofa::defaulttype;


template <typename Real>
struct IntrUtil{
public:
    inline static Real ZERO_TOLERANCE(){return 1e-6;}
    inline static Real SQ_ZERO_TOLERANCE(){return ZERO_TOLERANCE() * ZERO_TOLERANCE();}

    inline static void normalize(Vec<3,Real> & vec){
        Real n2 = vec.norm2();

        if(n2 < 1- SQ_ZERO_TOLERANCE() || n2 > 1 + SQ_ZERO_TOLERANCE())
            vec.normalize();
    }

    inline static bool normalized(const Vec<3,Real> & vec){
        Real n2 = vec.norm2();

        return n2 < 1 - SQ_ZERO_TOLERANCE() || n2 > 1 + SQ_ZERO_TOLERANCE();
    }
};

}
}
}

#include <sofa/component/collision/OBBModel.h>

namespace sofa{
namespace component{
namespace collision{

using namespace sofa::defaulttype;

template <class TReal>
struct MyBox{
    Vec<3,TReal> Extent;
    Vec<3,TReal> Axis[3];
    Vec<3,TReal> Center;

    void showVertices()const;
};

//----------------------------------------------------------------------------
template <typename Real>
class IntrConfiguration
{
public:
    // ContactSide (order of the intervals of projection).
    enum
    {
        LEFT,
        RIGHT,
        NONE
    };

    // VertexProjectionMap (how the vertices are projected to the minimum
    // and maximum points of the interval).
    enum
    {
        m2, m11,             // segments
        m3, m21, m12, m111,  // triangles
        m44, m2_2, m1_1      // boxes
    };

    // The VertexProjectionMap value for the configuration.
    int mMap;

    // The order of the vertices.
    int mIndex[8];

    // Projection interval.
    Real mMin, mMax;

    IntrConfiguration & operator=(const IntrConfiguration & other);
};
//----------------------------------------------------------------------------

template <typename Real>
class CapIntrConfiguration : public IntrConfiguration<Real>{
public:
    bool have_naxis;
    Vec<3,Real> axis;

    CapIntrConfiguration();

    Vec<3,Real> leftContactPoint(const Vec<3,Real> * seg,Real radius)const;
    Vec<3,Real> rightContactPoint(const Vec<3,Real> * seg,Real radius)const;

    void leftSegment(const Vec<3,Real> * seg,Real radius,Vec<3,Real> * lseg)const;
    void rightSegment(const Vec<3,Real> * seg,Real radius,Vec<3,Real> * lseg)const;

    CapIntrConfiguration & operator=(const CapIntrConfiguration & other);
};

//----------------------------------------------------------------------------
/**
*The axis must be normalized when testing a capsule !.
*TDataTypes is the data type of the OBB.
*/
template <class TDataTypes>
class IntrAxis
{
public:
    typedef typename TDataTypes::Real Real;
    typedef TOBB<TDataTypes> Box;
    typedef typename TOBB<TDataTypes>::Coord Coord;
    typedef IntrConfiguration<Real> IntrConf;

    // Find-query for intersection of projected intervals.  The velocity
    // input is the difference objectVelocity1 - objectVelocity0.  The
    // first and last times of contact are computed, as is information about
    // the contact configuration and the ordering of the projections (the
    // contact side).
//    static bool Find (const Coord& axis,
//        const Vec<3,Real> segment[2], const Triangle3<Real>& triangle,
//        const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast,
//        int& side, IntrConfiguration<Real>& segCfgFinal,
//        IntrConfiguration<Real>& triCfgFinal);

    static bool Find (const Coord& axis,
        const Vec<3,Real> segment[2],Real radius, const Box& box,
        const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast,
        int& side, CapIntrConfiguration<Real> &segCfgFinal,
        IntrConfiguration<Real>& boxCfgFinal,bool & config_modified);   

//    static bool Find (const Coord& axis,
//        const Triangle3<Real>& triangle, const Box& box,
//        const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast,
//        int& side, IntrConfiguration<Real>& triCfgFinal,
//        IntrConfiguration<Real>& boxCfgFinal);

    // if axis is found as the final separating axis then final_axis is updated and
    // become equal axis after this method
    static bool Find (const Coord& axis,
        const Box& box0, const Box& box1,
        const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast,
        int& side, IntrConfiguration<Real>& box0CfgFinal,
        IntrConfiguration<Real>& box1CfgFinal,bool & config_modified);

    static void FindStatic (const Coord& axis,
        const Box& box0, const Box& box1,
        Real dmax,Real& dfirst,
        int& side, IntrConfiguration<Real>& box0CfgFinal,
        IntrConfiguration<Real>& box1CfgFinal,bool & config_modified);

    static void FindStatic (const Coord& axis,
        const Vec<3,Real> segment[2],Real radius, const Box& box,
        Real dmax, Real& dfirst,
        int& side, CapIntrConfiguration<Real> &segCfgFinal,
        IntrConfiguration<Real>& boxCfgFinal,bool & config_modified);

    // Projections.
    static void GetProjection (const Coord& axis,
        const Coord segment[], Real& imin, Real& imax);

//    static void GetProjection (const Coord& axis,
//        const Triangle3<Real>& triangle, Real& imin, Real& imax);

    static void GetProjection (const Coord& axis,
        const Box& box, Real& imin, Real& imax);

    // Configurations.
    static void GetConfiguration (const Coord& axis,
        const Vec<3,Real> segment[2], IntrConfiguration<Real>& cfg);

    /**
    *The axis must be normalized when testing a capsule !.
    */
    static void GetConfiguration (const Coord& axis,
        const Vec<3,Real> segment[2], Real radius,CapIntrConfiguration<Real>& cfg);
    
//    static void GetConfiguration (const Coord& axis,
//        const Triangle3<Real>& triangle, IntrConfiguration<Real>& cfg);

    static void GetConfiguration (const Coord& axis,
        const Box& box, IntrConfiguration<Real>& cfg);

    // Low-level test-query for projections.
    static bool Test (const Coord& axis,
        const Vec<3,Real>& velocity, Real min0, Real max0, Real min1,
        Real max1, Real tmax, Real& tfirst, Real& tlast);

    // Low-level find-query for projections.
    // if axis is found as the final separating axis then final_axis is updated and
    // become equal axis after this method
    template <class Config0,class Config1>
    static bool Find (const Coord& axis,
        const Vec<3,Real>& velocity,
        const Config0& cfg0Start,
        const Config1& cfg1Start, Real tmax, int& side,
        Config0& cfg0Final,
        Config1& cfg1Final, Real& tfirst, Real& tlast,bool & config_modified);

    template <class Config0,class Config1>
    static void FindStatic (const Config0& cfg0Start,
        const Config1& cfg1Start,int& side,
        Config0& cfg0Final,
        Config1& cfg1Final, Real dmax,Real& dfirst,bool & config_modified);
};
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
/**
  *TDataTypes is the OBB type.
  */
template <class TDataTypes>
class  FindContactSet
{
public:
    typedef typename TDataTypes::Real Real;
    typedef TOBB<TDataTypes> Box;

//    FindContactSet (const Vec<3,Real> segment[2],
//        const Triangle3<Real>& triangle, int side,
//        const IntrConfiguration<Real>& segCfg,
//        const IntrConfiguration<Real>& triCfg,
//        const Vec<3,Real>& segVelocity, const Vec<3,Real>& triVelocity,
//        Real tfirst, int& quantity, Vec<3,Real>* P);

    FindContactSet (const Vec<3,Real> segment[2], const Box& box,
        int side, const IntrConfiguration<Real>& segCfg,
        const IntrConfiguration<Real>& boxCfg,
        const Vec<3,Real>& segVelocity, const Vec<3,Real>& boxVelocity,
        Real tfirst, int& quantity, Vec<3,Real>* P);

    FindContactSet (const Vec<3,Real> segment[2], Real radius,const Box& box,
        int side, CapIntrConfiguration<Real> &segCfg,
        const IntrConfiguration<Real>& boxCfg,
        const Vec<3,Real>& segVelocity, const Vec<3,Real>& boxVelocity,
        Real & tfirst, int& quantity, Vec<3,Real>* P);

    FindContactSet (const Vec<3,Real> segment[2], Real radius,const Box& box,const Vec<3,Real> & axis,
        int side, CapIntrConfiguration<Real> &capCfg,
        const IntrConfiguration<Real>& boxCfg,
        Real tfirst, Vec<3,Real> & pt_on_capsule,Vec<3,Real> & pt_on_box);

//    FindContactSet (const Triangle3<Real>& triangle,
//        const Box& box, int side,
//        const IntrConfiguration<Real>& triCfg,
//        const IntrConfiguration<Real>& boxCfg,
//        const Vec<3,Real>& triVelocity, const Vec<3,Real>& boxVelocity,
//        Real tfirst, int& quantity, Vec<3,Real>* P);

    FindContactSet (const Box& box0, const Box& box1,
        int side, const IntrConfiguration<Real>& box0Cfg,
        const IntrConfiguration<Real>& box1Cfg,
        const Vec<3,Real>& box0Velocity,
        const Vec<3,Real>& box1Velocity, Real tfirst, int& quantity,
        Vec<3,Real>* P);

    FindContactSet (const Box& box0, const Box& box1,const Vec<3,Real> & axis,
        int side, const IntrConfiguration<Real>& box0Cfg,
        const IntrConfiguration<Real>& box1Cfg,
        Real tfirst,
        Vec<3,Real> & pt_on_first,Vec<3,Real> & pt_on_second);

    static void moveOnBox(const Box & box,Vec<3,Real> & point);

private:
    // These functions are called when it is known that the features are
    // intersecting.  Consequently, they are specialized versions of the
    // object-object intersection algorithms.

    static void ColinearSegments (const Vec<3,Real> segment0[2],
        const Vec<3,Real> segment1[2], int& quantity, Vec<3,Real>* P);

    static void SegmentThroughPlane (const Vec<3,Real> segment[2],
        const Vec<3,Real>& planeOrigin, const Vec<3,Real>& planeNormal,
        int& quantity, Vec<3,Real>* P);

    static void SegmentSegment (const Vec<3,Real> segment0[2],
        const Vec<3,Real> segment1[2], int& quantity, Vec<3,Real>* P);

    static void ColinearSegmentTriangle (const Vec<3,Real> segment[2],
        const Vec<3,Real> triangle[3], int& quantity, Vec<3,Real>* P);

    static void CoplanarSegmentRectangle (const Vec<3,Real> segment[2],
        const Vec<3,Real> rectangle[4], int& quantity, Vec<3,Real>* P);

    static void CoplanarTriangleRectangle (const Vec<3,Real> triangle[3],
        const Vec<3,Real> rectangle[4], int& quantity, Vec<3,Real>* P);

    static void CoplanarRectangleRectangle (
        const Vec<3,Real> rectangle0[4],
        const Vec<3,Real> rectangle1[4], int& quantity, Vec<3,Real>* P);

    static void FindContactPoint (const Vec<3,Real> & segP0, Real radius,CapIntrConfiguration<Real> &capCfg,const MyBox<Real> &boxFinal,
        const Vec<3,Real>& boxVelocity,const Vec<3,Real> & capVelocity,
        Real & tfirst, Vec<3,Real>* P);

    static void FindContactConfig(const Vec<3,Real> & axis,const Vec<3,Real> & segP0, Real radius,const Box & box,CapIntrConfiguration<Real> &capCfg,int side,
        Vec<3, Real> & pt_on_capsule,Vec<3, Real> & pt_on_box);

    static void projectIntPoints(const Vec<3, Real> & velocity,Real contactTime,const Vec<3,Real> * points,int n,Vec<3,Real> & proj_pt);

    static void projectPointOnCapsuleAndFindCapNormal(const Vec<3,Real> & pt,const Vec<3,Real> segment[2],Real radius,CapIntrConfiguration<Real> & capCfg,Vec<3,Real> & pt_on_capsule);

    static void segNearestPoints(const Vec<3,Real> * p, const Vec<3,Real> * q,Vec<3,Real> & P,Vec<3,Real> & Q);

    static void facesNearestPoints(const Vec<3,Real> first_face[4],const Vec<3,Real> second_face[4],Vec<3,Real> & pt_on_first,Vec<3,Real> & pt_on_second);

    static void faceSegNearestPoints(const Vec<3,Real> face[4],const Vec<3,Real> seg[2],Vec<3,Real> & pt_on_face,Vec<3,Real> & pt_on_seg);    
};
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Miscellaneous support.
//----------------------------------------------------------------------------
// The input and output polygons are stored in P.  The size of P is
// assumed to be large enough to store the clipped convex polygon vertices.
// For now the maximum array size is 8 to support the current intersection
// algorithms.
template <typename Real>
void ClipConvexPolygonAgainstPlane (const Vec<3,Real>& normal,
    Real bonstant, int& quantity, Vec<3,Real>* P);

// Translates an index into the box back into real coordinates.
template <typename TReal>
Vec<3,TReal> GetPointFromIndex (int index, const MyBox<TReal>& box);

template <typename TDataTypes>
Vec<3,typename TDataTypes::Real> getPointFromIndex (int index, const TOBB<TDataTypes>& box);
//----------------------------------------------------------------------------

template <typename Real>
void projectIntPoints(const Vec<3, Real> & velocity0, const Vec<3, Real> & velocity1,Real contactTime,const Vec<3,Real> * points,int n,Vec<3,Real> & pt_on_first,Vec<3,Real> & pt_on_second);

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_BASE_COLLISION)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_COLLISION_API FindContactSet<defaulttype::Rigid3dTypes>;
extern template class SOFA_BASE_COLLISION_API IntrAxis<defaulttype::Rigid3dTypes>;
extern template class SOFA_BASE_COLLISION_API IntrConfiguration<double>;
extern template SOFA_BASE_COLLISION_API void ClipConvexPolygonAgainstPlane(const Vec<3,double>&, double, int&,Vec<3,double>*);
extern template SOFA_BASE_COLLISION_API Vec<3,double> GetPointFromIndex (int, const MyBox<double>& );
extern template SOFA_BASE_COLLISION_API Vec<3,typename Rigid3dTypes::Real> getPointFromIndex (int, const TOBB<Rigid3dTypes>& );
extern template SOFA_BASE_COLLISION_API class CapIntrConfiguration<double>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_COLLISION_API FindContactSet<defaulttype::Rigid3fTypes>;
extern template class SOFA_BASE_COLLISION_API IntrAxis<defaulttype::Rigid3fTypes>;
extern template class SOFA_BASE_COLLISION_API IntrConfiguration<float>;
extern template SOFA_BASE_COLLISION_API void ClipConvexPolygonAgainstPlane(const Vec<3,float>&, float, int&,Vec<3,float>*);
extern template SOFA_BASE_COLLISION_API Vec<3,float> GetPointFromIndex (int, const MyBox<float>& );
extern template SOFA_BASE_COLLISION_API Vec<3,typename Rigid3fTypes::Real> getPointFromIndex (int, const TOBB<Rigid3fTypes>& );
extern template SOFA_BASE_COLLISION_API class CapIntrConfiguration<float>;
#endif
#endif

}
}
}


#endif
