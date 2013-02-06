// File modified from GeometricTools
// http://www.geometrictools.com/


#ifndef WM5INTRUTILITY3_H
#define WM5INTRUTILITY3_H


namespace sofa{
namespace component{
namespace collision{


template <typename Real>
struct Math{
public:
    inline static Real ZERO_TOLERANCE(){return 1e-7;}
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
};
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
template <class TDataTypes>
class IntrAxis
{
public:
    typedef typename TDataTypes::Real Real;
    typedef TOBB<TDataTypes> Box;
    typedef typename TOBB<TDataTypes>::Coord Coord;

    // Test-query for intersection of projected intervals.  The velocity
    // input is the difference objectVelocity1 - objectVelocity0.  The
    // first and last times of contact are computed.
//    static bool Test (const Coord& axis,
//        const Vec<3,Real> segment[2], const Triangle3<Real>& triangle,
//        const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast);

//    static bool Test (const Coord& axis,
//        const Vec<3,Real> segment[2], const Box& box,
//        const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast);

//    static bool Test (const Coord& axis,
//        const Triangle3<Real>& triangle, const Box& box,
//        const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast);

    static bool Test (const Coord& axis,
        const Box& box0, const Box& box1,
        const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast);

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
        const Vec<3,Real> segment[2], const Box& box,
        const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast,
        int& side, IntrConfiguration<Real>& segCfgFinal,
        IntrConfiguration<Real>& boxCfgFinal);

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
    static bool Find (const Coord& axis,
        const Vec<3,Real>& velocity,
        const IntrConfiguration<Real>& cfg0Start,
        const IntrConfiguration<Real>& cfg1Start, Real tmax, int& side,
        IntrConfiguration<Real>& cfg0Final,
        IntrConfiguration<Real>& cfg1Final, Real& tfirst, Real& tlast,bool & config_modified);
};
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
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

//    FindContactSet (const Vec<3,Real> segment[2], const Box& box,
//        int side, const IntrConfiguration<Real>& segCfg,
//        const IntrConfiguration<Real>& boxCfg,
//        const Vec<3,Real>& segVelocity, const Vec<3,Real>& boxVelocity,
//        Real tfirst, int& quantity, Vec<3,Real>* P);

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

    FindContactSet (const Box& box0, const Box& box1,
        int side, const IntrConfiguration<Real>& box0Cfg,
        const IntrConfiguration<Real>& box1Cfg,
        const Vec<3,Real>& box0Velocity,
        const Vec<3,Real>& box1Velocity, Real tfirst, int& quantity,
        Vec<3,Real>* POnFirst,Vec<3,Real>* POnSecond);

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
}
}
}


#endif
