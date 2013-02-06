// File modified from GeometricTools
// http://www.geometrictools.com/

#include "IntrUtility3.h"


namespace sofa{
namespace component{
namespace collision{

using namespace sofa::defaulttype;

//----------------------------------------------------------------------------
// IntrAxis<TDataTypes>
//----------------------------------------------------------------------------
//template <class TDataTypes>
//bool IntrAxis<TDataTypes>::Test (const Coord& axis,
//    const Vec<3,Real> segment[2], const Triangle3<Real>& triangle,
//    const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast)
//{
//    Real min0, max0;
//    GetProjection(axis, segment, min0, max0);

//    Real min1, max1;
//    GetProjection(axis, triangle, min1, max1);

//    return Test(axis, velocity, min0, max0, min1, max1, tmax, tfirst, tlast);
//}
//----------------------------------------------------------------------------
//template <class TDataTypes>
//bool IntrAxis<TDataTypes>::Test (const Coord& axis,
//    const Vec<3,Real> segment[2], const Box& box,
//    const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast)
//{
//    Real min0, max0;
//    GetProjection(axis, segment, min0, max0);

//    Real min1, max1;
//    GetProjection(axis, box, min1, max1);
    
//    return Test(axis, velocity, min0, max0, min1, max1, tmax, tfirst, tlast);
//}
//----------------------------------------------------------------------------
//template <class TDataTypes>
//bool IntrAxis<TDataTypes>::Test (const Coord& axis,
//    const Triangle3<Real>& triangle, const Box& box,
//    const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast)
//{
//    Real min0, max0;
//    GetProjection(axis, triangle, min0, max0);

//    Real min1, max1;
//    GetProjection(axis, box, min1, max1);
    
//    return Test(axis, velocity, min0, max0, min1, max1, tmax, tfirst, tlast);
//}
//----------------------------------------------------------------------------
template <class TDataTypes>
bool IntrAxis<TDataTypes>::Test (const Coord& axis, const Box& box0,
    const Box& box1, const Vec<3,Real>& velocity, Real tmax,
    Real& tfirst, Real& tlast)
{
    Real min0, max0;
    GetProjection(axis, box0, min0, max0);

    Real min1, max1;
    GetProjection(axis, box1, min1, max1);
    
    return Test(axis, velocity, min0, max0, min1, max1, tmax, tfirst, tlast);
}
//----------------------------------------------------------------------------
//template <class TDataTypes>
//bool IntrAxis<TDataTypes>::Find (const Coord& axis,
//    const Vec<3,Real> segment[2], const Triangle3<Real>& triangle,
//    const Vec<3,Real>& velocity, Real tmax, Real& tfirst,
//    Real& tlast, int& side, IntrConfiguration<Real>& segCfgFinal,
//    IntrConfiguration<Real>& triCfgFinal)
//{
//    IntrConfiguration<Real> segCfgStart;
//    GetConfiguration(axis, segment, segCfgStart);

//    IntrConfiguration<Real> triCfgStart;
//    GetConfiguration(axis, triangle, triCfgStart);

//    return Find(axis, velocity, segCfgStart, triCfgStart, tmax, side,
//        segCfgFinal, triCfgFinal, tfirst, tlast);
//}
//----------------------------------------------------------------------------
template <class TDataTypes>
bool IntrAxis<TDataTypes>::Find (const Coord& axis,
    const Vec<3,Real> segment[2], const Box& box,
    const Vec<3,Real>& velocity, Real tmax, Real& tfirst,
    Real& tlast, int& side, IntrConfiguration<Real>& segCfgFinal,
    IntrConfiguration<Real>& boxCfgFinal)
{
    IntrConfiguration<Real> segCfgStart;
    GetConfiguration(axis, segment, segCfgStart);

    IntrConfiguration<Real> boxCfgStart;
    GetConfiguration(axis, box, boxCfgStart);

    bool config_modified;

    return Find(axis, velocity, segCfgStart, boxCfgStart, tmax, side,
        segCfgFinal, boxCfgFinal, tfirst, tlast,config_modified);
}
//----------------------------------------------------------------------------
//template <class TDataTypes>
//bool IntrAxis<TDataTypes>::Find (const Coord& axis,
//    const Triangle3<Real>& triangle, const Box& box,
//    const Vec<3,Real>& velocity, Real tmax, Real& tfirst,
//    Real& tlast, int& side, IntrConfiguration<Real>& segCfgFinal,
//    IntrConfiguration<Real>& boxCfgFinal)
//{
//    IntrConfiguration<Real> triCfgStart;
//    GetConfiguration(axis, triangle, triCfgStart);

//    IntrConfiguration<Real> boxCfgStart;
//    GetConfiguration(axis, box, boxCfgStart);

//    return Find(axis, velocity, triCfgStart, boxCfgStart, tmax, side,
//        segCfgFinal, boxCfgFinal, tfirst, tlast);
//}
//----------------------------------------------------------------------------
template <class TDataTypes>
bool IntrAxis<TDataTypes>::Find (const Coord& axis,
    const Box& box0, const Box& box1,
    const Vec<3,Real>& velocity, Real tmax, Real& tfirst,
    Real& tlast, int& side, IntrConfiguration<Real>& box0CfgFinal,
    IntrConfiguration<Real>& box1CfgFinal,bool & config_modified)
{
    IntrConfiguration<Real> box0CfgStart;
    GetConfiguration(axis,box0,box0CfgStart);

    IntrConfiguration<Real> box1CfgStart;
    GetConfiguration(axis,box1,box1CfgStart);

    return Find(axis, velocity, box0CfgStart, box1CfgStart, tmax, side,
        box0CfgFinal, box1CfgFinal, tfirst, tlast,config_modified);
}
//----------------------------------------------------------------------------
template <class TDataTypes>
void IntrAxis<TDataTypes>::GetProjection (const Coord& axis,
    const Coord segment[2], Real& imin, Real& imax)
{
    Real dot[2] =
    {
        axis * segment[0],
        axis * segment[1]
    };

    imin = dot[0];
    imax = imin;

    if (dot[1] < imin)
    {
        imin = dot[1];
    }
    else if (dot[1] > imax)
    {
        imax = dot[1];
    }
}
//----------------------------------------------------------------------------
//template <class TDataTypes>
//void IntrAxis<TDataTypes>::GetProjection (const Coord& axis,
//    const Triangle3<Real>& triangle, Real& imin, Real& imax)
//{
//    Real dot[3] =
//    {
//        axis.dot(triangle.V[0]),
//        axis.dot(triangle.V[1]),
//        axis.dot(triangle.V[2])
//    };

//    imin = dot[0];
//    imax = imin;

//    if (dot[1] < imin)
//    {
//        imin = dot[1];
//    }
//    else if (dot[1] > imax)
//    {
//        imax = dot[1];
//    }

//    if (dot[2] < imin)
//    {
//        imin = dot[2];
//    }
//    else if (dot[2] > imax)
//    {
//        imax = dot[2];
//    }
//}
//----------------------------------------------------------------------------
template <class TDataTypes>
void IntrAxis<TDataTypes>::GetProjection (const Coord& axis,
    const Box& box, Real& imin, Real& imax)
{
    Real origin = axis * box.center();
    Real maximumExtent =
        fabs(box.extent(0)*axis * box.axis(0)) +
        fabs(box.extent(1)*axis * box.axis(1)) +
        fabs(box.extent(2)*axis * box.axis(2));

    imin = origin - maximumExtent;
    imax = origin + maximumExtent;
}
//----------------------------------------------------------------------------
template <class TDataTypes>
void IntrAxis<TDataTypes>::GetConfiguration (const Coord& axis,
    const Vec<3,Real> segment[2], IntrConfiguration<Real>& cfg)
{
    Real dot[2] =
    {
        axis * segment[0],
        axis * segment[1]
    };

    if (fabs(dot[1] - dot[0]) < Math<Real>::ZERO_TOLERANCE())
    {
        cfg.mMap = IntrConfiguration<Real>::m2;
    }
    else
    {
        cfg.mMap = IntrConfiguration<Real>::m11;
    }

    if (dot[0] < dot[1])
    {
        cfg.mMin = dot[0];
        cfg.mMax = dot[1];
        cfg.mIndex[0] = 0;
        cfg.mIndex[1] = 1;
    }
    else
    {
        cfg.mMin = dot[1];
        cfg.mMax = dot[0];
        cfg.mIndex[0] = 1;
        cfg.mIndex[1] = 0;
    }
}
//----------------------------------------------------------------------------
//template <class TDataTypes>
//void IntrAxis<TDataTypes>::GetConfiguration (const Coord& axis,
//    const Triangle3<Real>& triangle, IntrConfiguration<Real>& cfg)
//{
//    // Find projections of vertices onto potential separating axis.
//    Real d0 = axis.dot(triangle.V[0]);
//    Real d1 = axis.dot(triangle.V[1]);
//    Real d2 = axis.dot(triangle.V[2]);

//    // Explicit sort of vertices to construct a IntrConfiguration.
//    if (d0 <= d1)
//    {
//        if (d1 <= d2) // D0 <= D1 <= D2
//        {
//            if (d0 != d1)
//            {
//                if (d1 != d2)
//                {
//                    cfg.mMap = IntrConfiguration<Real>::m111;
//                }
//                else
//                {
//                    cfg.mMap = IntrConfiguration<Real>::m12;
//                }
//            }
//            else // ( D0 == D1 )
//            {
//                if (d1 != d2)
//                {
//                    cfg.mMap = IntrConfiguration<Real>::m21;
//                }
//                else
//                {
//                    cfg.mMap = IntrConfiguration<Real>::m3;
//                }
//            }
//            cfg.mIndex[0] = 0;
//            cfg.mIndex[1] = 1;
//            cfg.mIndex[2] = 2;
//            cfg.mMin = d0;
//            cfg.mMax = d2;
//        }
//        else if (d0 <= d2) // D0 <= D2 < D1
//        {
//            if (d0 != d2)
//            {
//                cfg.mMap = IntrConfiguration<Real>::m111;
//                cfg.mIndex[0] = 0;
//                cfg.mIndex[1] = 2;
//                cfg.mIndex[2] = 1;
//            }
//            else
//            {
//                cfg.mMap = IntrConfiguration<Real>::m21;
//                cfg.mIndex[0] = 2;
//                cfg.mIndex[1] = 0;
//                cfg.mIndex[2] = 1;
//            }
//            cfg.mMin = d0;
//            cfg.mMax = d1;
//        }
//        else // D2 < D0 <= D1
//        {
//            if (d0 != d1)
//            {
//                cfg.mMap = IntrConfiguration<Real>::m111;
//            }
//            else
//            {
//                cfg.mMap = IntrConfiguration<Real>::m12;
//            }

//            cfg.mIndex[0] = 2;
//            cfg.mIndex[1] = 0;
//            cfg.mIndex[2] = 1;
//            cfg.mMin = d2;
//            cfg.mMax = d1;
//        }
//    }
//    else if (d2 <= d1) // D2 <= D1 < D0
//    {
//        if (d2 != d1)
//        {
//            cfg.mMap = IntrConfiguration<Real>::m111;
//            cfg.mIndex[0] = 2;
//            cfg.mIndex[1] = 1;
//            cfg.mIndex[2] = 0;
//        }
//        else
//        {
//            cfg.mMap = IntrConfiguration<Real>::m21;
//            cfg.mIndex[0] = 1;
//            cfg.mIndex[1] = 2;
//            cfg.mIndex[2] = 0;

//        }
//        cfg.mMin = d2;
//        cfg.mMax = d0;
//    }
//    else if (d2 <= d0) // D1 < D2 <= D0
//    {
//        if (d2 != d0)
//        {
//            cfg.mMap = IntrConfiguration<Real>::m111;
//        }
//        else
//        {
//            cfg.mMap = IntrConfiguration<Real>::m12;
//        }

//        cfg.mIndex[0] = 1;
//        cfg.mIndex[1] = 2;
//        cfg.mIndex[2] = 0;
//        cfg.mMin = d1;
//        cfg.mMax = d0;
//    }
//    else // D1 < D0 < D2
//    {
//        cfg.mMap = IntrConfiguration<Real>::m111;
//        cfg.mIndex[0] = 1;
//        cfg.mIndex[1] = 0;
//        cfg.mIndex[2] = 2;
//        cfg.mMin = d1;
//        cfg.mMax = d2;
//    }
//}
//----------------------------------------------------------------------------
template <class TDataTypes>
void IntrAxis<TDataTypes>::GetConfiguration (const Coord& axis,
    const Box& box, IntrConfiguration<Real>& cfg)
{
    // Description of coordinate ordering scheme for IntrConfiguration.mIndex.
    //
    // Vertex number (up/down) vs. sign of extent (only matters in mapping
    // back)
    //   012
    // 0 ---
    // 1 +--
    // 2 -+-
    // 3 ++-
    // 4 --+
    // 5 +-+
    // 6 -++
    // 7 +++
    //
    // When it returns an ordering in the IntrConfiguration, it is also
    // guarenteed to be in-order (if 4 vertices, then they are guarenteed in
    // an order that will create a box, e.g. 0,1,3,2).

    Real axes[3] =
    {
        axis * box.axis(0),
        axis * box.axis(1),
        axis * box.axis(2)
    };

    Real absAxes[3] =
    {
        fabs(axes[0]),
        fabs(axes[1]),
        fabs(axes[2])
    };

    Real maxProjectedExtent;

    if (absAxes[0] < Math<Real>::ZERO_TOLERANCE())
    {
        if (absAxes[1] < Math<Real>::ZERO_TOLERANCE())
        {
            // face-face
            cfg.mMap = IntrConfiguration<Real>::m44;

            maxProjectedExtent = absAxes[2]*box.extent(2);

            // faces have normals along axis[2]
            if (axes[2] > (Real)0)
            {       
                cfg.mIndex[0] = 0;
                cfg.mIndex[1] = 1;
                cfg.mIndex[2] = 3;
                cfg.mIndex[3] = 2;

                cfg.mIndex[4] = 6;
                cfg.mIndex[5] = 7;
                cfg.mIndex[6] = 5;
                cfg.mIndex[7] = 4;
            }
            else
            {
                cfg.mIndex[0] = 6;
                cfg.mIndex[1] = 7;
                cfg.mIndex[2] = 5;
                cfg.mIndex[3] = 4;

                cfg.mIndex[4] = 0;
                cfg.mIndex[5] = 1;
                cfg.mIndex[6] = 3;
                cfg.mIndex[7] = 2;
            }
        }
        else if (absAxes[2] < Math<Real>::ZERO_TOLERANCE())
        {
            // face-face
            cfg.mMap = IntrConfiguration<Real>::m44;

            maxProjectedExtent = absAxes[1]*box.extent(1);

            // faces have normals along axis[1]
            if (axes[1] > (Real)0) 
            {
                cfg.mIndex[0] = 4;
                cfg.mIndex[1] = 5;
                cfg.mIndex[2] = 1;
                cfg.mIndex[3] = 0;

                cfg.mIndex[4] = 2;
                cfg.mIndex[5] = 3;
                cfg.mIndex[6] = 7;
                cfg.mIndex[7] = 6;
            }
            else
            {
                cfg.mIndex[0] = 2;
                cfg.mIndex[1] = 3;
                cfg.mIndex[2] = 7;
                cfg.mIndex[3] = 6;

                cfg.mIndex[4] = 4;
                cfg.mIndex[5] = 5;
                cfg.mIndex[6] = 1;
                cfg.mIndex[7] = 0;
            }
        }
        else // only axes[0] is equal to 0
        {
            // seg-seg
            cfg.mMap = IntrConfiguration<Real>::m2_2;

            maxProjectedExtent = absAxes[1]*box.extent(1) +
                absAxes[2]*box.extent(2);

            // axis 0 is perpendicular to axis
            if (axes[1] > (Real)0)
            {
                if (axes[2] > (Real)0) 
                {
                    cfg.mIndex[0] = 0;
                    cfg.mIndex[1] = 1;

                    cfg.mIndex[6] = 6;
                    cfg.mIndex[7] = 7;
                }
                else 
                {
                    cfg.mIndex[0] = 4;
                    cfg.mIndex[1] = 5;

                    cfg.mIndex[6] = 2;
                    cfg.mIndex[7] = 3;
                }
            }
            else // axes[1] < 0
            {
                if (axes[2] > (Real)0)
                {
                    cfg.mIndex[0] = 2;
                    cfg.mIndex[1] = 3;

                    cfg.mIndex[6] = 4;
                    cfg.mIndex[7] = 5;
                }
                else
                {
                    cfg.mIndex[0] = 6;
                    cfg.mIndex[1] = 7;

                    cfg.mIndex[6] = 0;
                    cfg.mIndex[7] = 1;
                }
            }
        }
    }
    else if (absAxes[1] < Math<Real>::ZERO_TOLERANCE())
    {
        if (absAxes[2] < Math<Real>::ZERO_TOLERANCE())
        {
            // face-face
            cfg.mMap = IntrConfiguration<Real>::m44;

            maxProjectedExtent = absAxes[0]*box.extent(0);

            // faces have normals along axis[0]
            if (axes[0] > (Real)0)
            {
                cfg.mIndex[0] = 0;
                cfg.mIndex[1] = 2;
                cfg.mIndex[2] = 6;
                cfg.mIndex[3] = 4;

                cfg.mIndex[4] = 5;
                cfg.mIndex[5] = 7;
                cfg.mIndex[6] = 3;
                cfg.mIndex[7] = 1;
            }
            else
            {
                cfg.mIndex[4] = 0;
                cfg.mIndex[5] = 2;
                cfg.mIndex[6] = 6;
                cfg.mIndex[7] = 4;

                cfg.mIndex[0] = 5;
                cfg.mIndex[1] = 7;
                cfg.mIndex[2] = 3;
                cfg.mIndex[3] = 1;
            }

        }
        else // only axes[1] is equal to 0
        {
            // seg-seg
            cfg.mMap = IntrConfiguration<Real>::m2_2;

            maxProjectedExtent = absAxes[0]*box.extent(0) +
                absAxes[2]*box.extent(2);

            // axis 1 is perpendicular to axis
            if (axes[0] > (Real)0)
            {
                if (axes[2] > (Real)0) 
                {
                    cfg.mIndex[0] = 0;
                    cfg.mIndex[1] = 2;

                    cfg.mIndex[6] = 5;
                    cfg.mIndex[7] = 7;
                }
                else 
                {
                    cfg.mIndex[0] = 4;
                    cfg.mIndex[1] = 6;

                    cfg.mIndex[6] = 1;
                    cfg.mIndex[7] = 3;
                }
            }
            else // axes[0] < 0
            {
                if (axes[2] > (Real)0)
                {
                    cfg.mIndex[0] = 1;
                    cfg.mIndex[1] = 3;

                    cfg.mIndex[6] = 4;
                    cfg.mIndex[7] = 6;
                }
                else
                {
                    cfg.mIndex[0] = 5;
                    cfg.mIndex[1] = 7;

                    cfg.mIndex[6] = 0;
                    cfg.mIndex[7] = 2;
                }
            }
        }
    }
    
    else if (absAxes[2] < Math<Real>::ZERO_TOLERANCE())
    {
        // only axis2 less than zero
        // seg-seg
        cfg.mMap = IntrConfiguration<Real>::m2_2;

        maxProjectedExtent = absAxes[0]*box.extent(0) +
            absAxes[1]*box.extent(1);

        // axis 2 is perpendicular to axis
        if (axes[0] > (Real)0)
        {
            if (axes[1] > (Real)0) 
            {
                cfg.mIndex[0] = 0;
                cfg.mIndex[1] = 4;

                cfg.mIndex[6] = 3;
                cfg.mIndex[7] = 7;
            }
            else 
            {
                cfg.mIndex[0] = 2;
                cfg.mIndex[1] = 6;

                cfg.mIndex[6] = 1;
                cfg.mIndex[7] = 5;
            }
        }
        else // axes[0] < 0
        {
            if (axes[1] > (Real)0)
            {
                cfg.mIndex[0] = 1;
                cfg.mIndex[1] = 5;

                cfg.mIndex[6] = 2;
                cfg.mIndex[7] = 6;
            }
            else
            {
                cfg.mIndex[0] = 3;
                cfg.mIndex[1] = 7;

                cfg.mIndex[6] = 0;
                cfg.mIndex[7] = 4;
            }
        }
    }
  
    else // no axis is equal to zero
    {
        // point-point (unique maximal and minimal vertex)
        cfg.mMap = IntrConfiguration<Real>::m1_1;

        maxProjectedExtent = absAxes[0]*box.extent(0) +
            absAxes[1]*box.extent(1) + absAxes[2]*box.extent(2);

        // only these two vertices matter, the rest are irrelevant
        cfg.mIndex[0] = 
            (axes[0] > (Real)0.0 ? 0 : 1) + 
            (axes[1] > (Real)0.0 ? 0 : 2) + 
            (axes[2] > (Real)0.0 ? 0 : 4);
        // by ordering the vertices this way, opposite corners add up to 7
        cfg.mIndex[7] = 7 - cfg.mIndex[0];
    }

    // Find projections onto line
    Real origin = axis * box.center();
    cfg.mMin = origin - maxProjectedExtent;
    cfg.mMax = origin + maxProjectedExtent;
}
//----------------------------------------------------------------------------
template <class TDataTypes>
bool IntrAxis<TDataTypes>::Test (const Coord& axis,
    const Vec<3,Real>& velocity, Real min0, Real max0, Real min1,
    Real max1, Real tmax, Real& tfirst, Real& tlast)
{
    // Static separating axis test.  Returns false iff object0 and object1
    // do not intersect in the interval [0,TMax] on any separating axis
    // ( TFirst > TLast || TFirst > TMax ) during the time interval, that is,
    // a quick out.  Returns true otherwise.
    //
    // min0, max0, min1, and max1 are the minimal and maximal points of
    // whatever object object0 and object1 are projected onto the test axis.
    //
    // velocity is Velocity1 - Velocity0

    Real t;
    Real speed = axis * velocity;
    
    if (max1 < min0) // object1 on left of object0
    {
        if (speed <= (Real)0) // object1 moving away from object0
        {
            return false;
        }

        // find first time of contact on this axis
        t = (min0 - max1)/speed;
        if (t > tfirst)
        {
            tfirst = t;
        }

        // quick out: intersection after desired interval
        if (tfirst > tmax)
        {
            return false;   
        }

        // find last time of contact on this axis
        t = (max0 - min1)/speed;
        if (t < tlast)
        {
            tlast = t;
        }

        // quick out: intersection before desired interval
        if (tfirst > tlast)
        {
            return false; 
        }
    }
    else if (max0 < min1)   // object1 on right of object0
    {
        if (speed >= (Real)0) // object1 moving away from object0
        {
            return false;
        }

        // find first time of contact on this axis
        t = (max0 - min1)/speed;
        if (t > tfirst)
        {
            tfirst = t;
        }

        // quick out: intersection after desired interval
        if (tfirst > tmax)
        {
            return false;   
        }

        // find last time of contact on this axis
        t = (min0 - max1)/speed;
        if (t < tlast)
        {
            tlast = t;
        }

        // quick out: intersection before desired interval
        if (tfirst > tlast)
        {
            return false; 
        }

    }
    else // object1 and object0 on overlapping interval
    {
        if (speed > (Real)0)
        {
            // find last time of contact on this axis
            t = (max0 - min1)/speed;
            if (t < tlast)
            {
                tlast = t;
            }

            // quick out: intersection before desired interval
            if (tfirst > tlast)
            {
                return false; 
            }
        }
        else if (speed < (Real)0)
        {
            // find last time of contact on this axis
            t = (min0 - max1)/speed;
            if (t < tlast)
            {
                tlast = t;
            }

            // quick out: intersection before desired interval
            if (tfirst > tlast)
            {
                return false;
            }
        }
    }
    return true;
}
//----------------------------------------------------------------------------
template <class TDataTypes>
bool IntrAxis<TDataTypes>::Find (const Coord& axis,
    const Vec<3,Real>& velocity, const IntrConfiguration<Real>& cfg0Start,
    const IntrConfiguration<Real>& cfg1Start, Real tmax, int& side,
    IntrConfiguration<Real>& cfg0Final, IntrConfiguration<Real>& cfg1Final,
    Real& tfirst, Real& tlast,bool & config_modified)
{
    config_modified = false;
    // Constant velocity separating axis test.  The configurations cfg0Start
    // and cfg1Start are the current potential configurations for contact,
    // and cfg0Final and cfg1Final are improved configurations.
    Real t;
    Real speed = axis * velocity;

    if (cfg1Start.mMax < cfg0Start.mMin) // object1 left of object0
    {
        if (speed <= (Real)0) // object1 moving away from object0
        {
            return false;
        }

        // find first time of contact on this axis
        t = (cfg0Start.mMin - cfg1Start.mMax)/speed;

        // If this is the new maximum first time of contact, set side and
        // configuration.
        if (t > tfirst)
        {
            tfirst = t;
            side = IntrConfiguration<Real>::LEFT;
            cfg0Final = cfg0Start;
            cfg1Final = cfg1Start;
            config_modified = true;
        }

        // quick out: intersection after desired interval
        if (tfirst > tmax)
        {
            return false;
        }

        // find last time of contact on this axis
        t = (cfg0Start.mMax - cfg1Start.mMin)/speed;
        if (t < tlast)
        {
            tlast = t;
        }

        // quick out: intersection before desired interval
        if (tfirst > tlast)
        {
            return false;
        }
    }
    else if (cfg0Start.mMax < cfg1Start.mMin)  // obj1 right of obj0
    {
        if (speed >= (Real)0) // object1 moving away from object0
        {
            return false;
        }

        // find first time of contact on this axis
        t = (cfg0Start.mMax - cfg1Start.mMin)/speed;

        // If this is the new maximum first time of contact,  set side and
        // configuration.
        if (t > tfirst)
        {
            tfirst = t;
            side = IntrConfiguration<Real>::RIGHT;
            cfg0Final = cfg0Start;
            cfg1Final = cfg1Start;
            config_modified = true;
        }

        // quick out: intersection after desired interval
        if (tfirst > tmax)
        {
            return false;   
        }

        // find last time of contact on this axis
        t = (cfg0Start.mMin - cfg1Start.mMax)/speed;
        if (t < tlast)
        {
            tlast = t;
        }

        // quick out: intersection before desired interval
        if (tfirst > tlast)
        {
            return false;
        }
    }
    else // object1 and object0 on overlapping interval
    {
        if (speed > (Real)0)
        {
            // find last time of contact on this axis
            t = (cfg0Start.mMax - cfg1Start.mMin)/speed;
            if (t < tlast)
            {
                tlast = t;
            }

            // quick out: intersection before desired interval
            if (tfirst > tlast)
            {
                return false; 
            }
        }
        else if (speed < (Real)0)
        {
            // find last time of contact on this axis
            t = (cfg0Start.mMin - cfg1Start.mMax)/speed;
            if (t < tlast)
            {
                tlast = t;
            }

            // quick out: intersection before desired interval
            if (tfirst > tlast)
            {
                return false; 
            }
        }
    }

    return true;
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// FindContactSet<Real>
//----------------------------------------------------------------------------
//template <class TDataTypes>
//FindContactSet<Real>::FindContactSet (const Vec<3,Real> segment[2],
//    const Triangle3<Real>& triangle, int side,
//    const IntrConfiguration<Real>& segCfg,
//    const IntrConfiguration<Real>& triCfg,
//    const Vec<3,Real>& segVelocity, const Vec<3,Real>& triVelocity,
//    Real tfirst, int& quantity, Vec<3,Real>* P)
//{
//    // Move the segment to its new position.
//    Vec<3,Real> segFinal[2] =
//    {
//        segment[0] + tfirst*segVelocity,
//        segment[1] + tfirst*segVelocity
//    };

//    // Move the triangle to its new position.
//    Vec<3,Real> triFinal[3] =
//    {
//        triangle.V[0] + tfirst*triVelocity,
//        triangle.V[1] + tfirst*triVelocity,
//        triangle.V[2] + tfirst*triVelocity
//    };

//    const int* sIndex = segCfg.mIndex;
//    const int* tIndex = triCfg.mIndex;

//    if (side == IntrConfiguration<Real>::LEFT) // tri on left of seg
//    {
//        if (segCfg.mMap == IntrConfiguration<Real>::m11)
//        {
//            quantity = 1;
//            P[0] = segFinal[sIndex[0]];
//        }
//        else if (triCfg.mMap == IntrConfiguration<Real>::m111
//        ||  triCfg.mMap == IntrConfiguration<Real>::m21)
//        {
//            quantity = 1;
//            P[0] = triFinal[tIndex[2]];
//        }
//        else if (triCfg.mMap == IntrConfiguration<Real>::m12)
//        {
//            Vec<3,Real> temp[2];
//            temp[0] = triFinal[tIndex[1]];
//            temp[1] = triFinal[tIndex[2]];
//            SegmentSegment(segFinal, temp, quantity, P);
//        }
//        else // seg is m2, tri is m3
//        {
//            ColinearSegmentTriangle(segFinal, triFinal, quantity, P);
//        }
//    }
//    else // seg on left of tri
//    {
//        if (segCfg.mMap == IntrConfiguration<Real>::m11)
//        {
//            quantity = 1;
//            P[0] = segFinal[sIndex[1]];
//        }
//        else if (triCfg.mMap == IntrConfiguration<Real>::m111
//        ||  triCfg.mMap == IntrConfiguration<Real>::m12)
//        {
//            quantity = 1;
//            P[0] = triFinal[tIndex[0]];
//        }
//        else if (triCfg.mMap == IntrConfiguration<Real>::m21)
//        {
//            Vec<3,Real> temp[2];
//            temp[0] = triFinal[tIndex[0]];
//            temp[1] = triFinal[tIndex[1]];
//            SegmentSegment(segFinal, temp, quantity, P);
//        }
//        else // seg is m2, tri is m3
//        {
//            ColinearSegmentTriangle(segFinal, triFinal, quantity, P);
//        }
//    }
//}
//----------------------------------------------------------------------------
//template <class TDataTypes>
//FindContactSet<Real>::FindContactSet (const Vec<3,Real> segment[2],
//    const Box& box, int side, const IntrConfiguration<Real>& segCfg,
//    const IntrConfiguration<Real>& boxCfg, const Vec<3,Real>& segVelocity,
//    const Vec<3,Real>& boxVelocity, Real tfirst, int& quantity,
//    Vec<3,Real>* P)
//{
//    // Move the segment to its new position.
//    Vec<3,Real> segFinal[2] =
//    {
//        segment[0] + tfirst*segVelocity,
//        segment[1] + tfirst*segVelocity
//    };

//    // Move the box to its new position.
//    Box boxFinal;
//    boxFinal.center() = box.center() + tfirst*boxVelocity;
//    for (int i = 0; i < 3; ++i)
//    {
//        boxFinal.Axis[i] = box.Axis[i];
//        boxFinal.Extent[i] = box.Extent[i];
//    }

//    const int* sIndex = segCfg.mIndex;
//    const int* bIndex = boxCfg.mIndex;

//    if (side == IntrConfiguration<Real>::LEFT)
//    {
//        // box on left of seg
//        if (segCfg.mMap == IntrConfiguration<Real>::m11)
//        {
//            quantity = 1;
//            P[0] = segFinal[sIndex[0]];
//        }
//        else if (boxCfg.mMap == IntrConfiguration<Real>::m1_1)
//        {
//            quantity = 1;
//            P[0] = GetPointFromIndex(bIndex[7], boxFinal);
//        }
//        else if (boxCfg.mMap == IntrConfiguration<Real>::m2_2)
//        {
//            // segment-segment intersection
//            Vec<3,Real> boxSeg[2];
//            boxSeg[0] = GetPointFromIndex(bIndex[6], boxFinal);
//            boxSeg[1] = GetPointFromIndex(bIndex[7], boxFinal);
//            SegmentSegment(segFinal, boxSeg, quantity, P);
//        }
//        else // boxCfg.mMap == IntrConfiguration<Real>::m44
//        {
//            // segment-boxface intersection
//            Vec<3,Real> boxFace[4];
//            boxFace[0] = GetPointFromIndex(bIndex[4], boxFinal);
//            boxFace[1] = GetPointFromIndex(bIndex[5], boxFinal);
//            boxFace[2] = GetPointFromIndex(bIndex[6], boxFinal);
//            boxFace[3] = GetPointFromIndex(bIndex[7], boxFinal);
//            CoplanarSegmentRectangle(segFinal, boxFace, quantity, P);
//        }
//    }
//    else // side == RIGHT
//    {
//        // box on right of seg
//        if (segCfg.mMap == IntrConfiguration<Real>::m11)
//        {
//            quantity = 1;
//            P[0] = segFinal[sIndex[1]];
//        }
//        else if (boxCfg.mMap == IntrConfiguration<Real>::m1_1)
//        {
//            quantity = 1;
//            P[0] = GetPointFromIndex(bIndex[0], boxFinal);
//        }
//        else if (boxCfg.mMap == IntrConfiguration<Real>::m2_2)
//        {
//            // segment-segment intersection
//            Vec<3,Real> boxSeg[2];
//            boxSeg[0] = GetPointFromIndex(bIndex[0], boxFinal);
//            boxSeg[1] = GetPointFromIndex(bIndex[1], boxFinal);
//            SegmentSegment(segFinal, boxSeg, quantity, P);
//        }
//        else // boxCfg.mMap == IntrConfiguration<Real>::m44
//        {
//            // segment-boxface intersection
//            Vec<3,Real> boxFace[4];
//            boxFace[0] = GetPointFromIndex(bIndex[0], boxFinal);
//            boxFace[1] = GetPointFromIndex(bIndex[1], boxFinal);
//            boxFace[2] = GetPointFromIndex(bIndex[2], boxFinal);
//            boxFace[3] = GetPointFromIndex(bIndex[3], boxFinal);
//            CoplanarSegmentRectangle(segFinal, boxFace, quantity, P);
//        }
//    }
//}
////----------------------------------------------------------------------------
//template <class TDataTypes>
//FindContactSet<Real>::FindContactSet (const Triangle3<Real>& triangle,
//    const Box& box, int side, const IntrConfiguration<Real>& triCfg,
//    const IntrConfiguration<Real>& boxCfg, const Vec<3,Real>& triVelocity,
//    const Vec<3,Real>& boxVelocity, Real tfirst, int& quantity,
//    Vec<3,Real>* P)
//{
//    // Move the triangle to its new position.
//    Vec<3,Real> triFinal[3] =
//    {
//        triangle.V[0] + tfirst*triVelocity,
//        triangle.V[1] + tfirst*triVelocity,
//        triangle.V[2] + tfirst*triVelocity,
//    };

//    // Move the box to new its position.
//    Box boxFinal;
//    boxFinal.center() = box.center() + tfirst*boxVelocity;
//    for (int i = 0; i < 3; ++i)
//    {
//        boxFinal.Axis[i] = box.Axis[i];
//        boxFinal.Extent[i] = box.Extent[i];
//    }

//    const int* tIndex = triCfg.mIndex;
//    const int* bIndex = boxCfg.mIndex;

//    if (side == IntrConfiguration<Real>::LEFT)
//    {
//        // box on left of tri
//        if (triCfg.mMap == IntrConfiguration<Real>::m111
//        ||  triCfg.mMap == IntrConfiguration<Real>::m12)
//        {
//            quantity = 1;
//            P[0] = triFinal[tIndex[0]];
//        }
//        else if (boxCfg.mMap == IntrConfiguration<Real>::m1_1)
//        {
//            quantity = 1;
//            P[0] = GetPointFromIndex(bIndex[7], boxFinal);
//        }
//        else if (triCfg.mMap == IntrConfiguration<Real>::m21)
//        {
//            if (boxCfg.mMap == IntrConfiguration<Real>::m2_2)
//            {
//                // triseg-boxseg intersection
//                Vec<3,Real> triSeg[2], boxSeg[2];
//                triSeg[0] = triFinal[tIndex[0]];
//                triSeg[1] = triFinal[tIndex[1]];
//                boxSeg[0] = GetPointFromIndex(bIndex[6], boxFinal);
//                boxSeg[1] = GetPointFromIndex(bIndex[7], boxFinal);
//                SegmentSegment(triSeg, boxSeg, quantity, P);
//            }
//            else // boxCfg.mMap == IntrConfiguration<Real>::m44
//            {
//                // triseg-boxface intersection
//                Vec<3,Real> triSeg[2], boxFace[4];
//                triSeg[0] = triFinal[tIndex[0]];
//                triSeg[1] = triFinal[tIndex[1]];
//                boxFace[0] = GetPointFromIndex(bIndex[4], boxFinal);
//                boxFace[1] = GetPointFromIndex(bIndex[5], boxFinal);
//                boxFace[2] = GetPointFromIndex(bIndex[6], boxFinal);
//                boxFace[3] = GetPointFromIndex(bIndex[7], boxFinal);
//                CoplanarSegmentRectangle(triSeg, boxFace, quantity, P);
//            }
//        }
//        else // triCfg.mMap == IntrConfiguration<Real>::m3
//        {
//            if (boxCfg.mMap == IntrConfiguration<Real>::m2_2)
//            {
//                // boxseg-triface intersection
//                Vec<3,Real> boxSeg[2];
//                boxSeg[0] = GetPointFromIndex(bIndex[6], boxFinal);
//                boxSeg[1] = GetPointFromIndex(bIndex[7], boxFinal);
//                ColinearSegmentTriangle(boxSeg, triFinal, quantity, P);
//            }
//            else
//            {
//                // triface-boxface intersection
//                Vec<3,Real> boxFace[4];
//                boxFace[0] = GetPointFromIndex(bIndex[4], boxFinal);
//                boxFace[1] = GetPointFromIndex(bIndex[5], boxFinal);
//                boxFace[2] = GetPointFromIndex(bIndex[6], boxFinal);
//                boxFace[3] = GetPointFromIndex(bIndex[7], boxFinal);
//                CoplanarTriangleRectangle(triFinal, boxFace, quantity, P);
//            }
//        }
//    }
//    else // side == RIGHT
//    {
//        // box on right of tri
//        if (triCfg.mMap == IntrConfiguration<Real>::m111
//        ||  triCfg.mMap == IntrConfiguration<Real>::m21)
//        {
//            quantity = 1;
//            P[0] = triFinal[tIndex[2]];
//        }
//        else if (boxCfg.mMap == IntrConfiguration<Real>::m1_1)
//        {
//            quantity = 1;
//            P[0] = GetPointFromIndex(bIndex[0], boxFinal);
//        }
//        else if (triCfg.mMap == IntrConfiguration<Real>::m12)
//        {
//            if (boxCfg.mMap == IntrConfiguration<Real>::m2_2)
//            {
//                // segment-segment intersection
//                Vec<3,Real> triSeg[2], boxSeg[2];
//                triSeg[0] = triFinal[tIndex[1]];
//                triSeg[1] = triFinal[tIndex[2]];
//                boxSeg[0] = GetPointFromIndex(bIndex[0], boxFinal);
//                boxSeg[1] = GetPointFromIndex(bIndex[1], boxFinal);
//                SegmentSegment(triSeg, boxSeg, quantity, P);
//            }
//            else // boxCfg.mMap == IntrConfiguration<Real>::m44
//            {
//                // triseg-boxface intersection
//                Vec<3,Real> triSeg[2], boxFace[4];
//                triSeg[0] = triFinal[tIndex[1]];
//                triSeg[1] = triFinal[tIndex[2]];
//                boxFace[0] = GetPointFromIndex(bIndex[0], boxFinal);
//                boxFace[1] = GetPointFromIndex(bIndex[1], boxFinal);
//                boxFace[2] = GetPointFromIndex(bIndex[2], boxFinal);
//                CoplanarSegmentRectangle(triSeg, boxFace, quantity, P);
//            }
//        }
//        else // triCfg.mMap == IntrConfiguration<Real>::m3
//        {
//            if (boxCfg.mMap == IntrConfiguration<Real>::m2_2)
//            {
//                // boxseg-triface intersection
//                Vec<3,Real> boxSeg[2];
//                boxSeg[0] = GetPointFromIndex(bIndex[0], boxFinal);
//                boxSeg[1] = GetPointFromIndex(bIndex[1], boxFinal);
//                ColinearSegmentTriangle(boxSeg, triFinal, quantity, P);
//            }
//            else
//            {
//                // triface-boxface intersection
//                Vec<3,Real> boxFace[4];
//                boxFace[0] = GetPointFromIndex(bIndex[0], boxFinal);
//                boxFace[1] = GetPointFromIndex(bIndex[1], boxFinal);
//                boxFace[2] = GetPointFromIndex(bIndex[2], boxFinal);
//                boxFace[3] = GetPointFromIndex(bIndex[3], boxFinal);
//                CoplanarTriangleRectangle(triFinal, boxFace, quantity, P);
//            }
//        }
//    }
//}
//----------------------------------------------------------------------------
template <class TDataTypes>
FindContactSet<TDataTypes>::FindContactSet (const Box& box0,
    const Box& box1, int side, const IntrConfiguration<Real>& box0Cfg,
    const IntrConfiguration<Real>& box1Cfg, const Vec<3,Real>& box0Velocity,
    const Vec<3,Real>& box1Velocity, Real tfirst, int& quantity,
    Vec<3,Real>* P)
{
    // Move the boxes to their new positions.
    MyBox<Real> box0Final, box1Final;
    box0Final.Center = box0.center() + tfirst*box0Velocity;
    box1Final.Center = box1.center() + tfirst*box1Velocity;
    for (int i = 0; i < 3; ++i)
    {
        box0Final.Axis[i] = box0.axis(i);
        box0Final.Extent[i] = box0.extent(i);
        box1Final.Axis[i] = box1.axis(i);
        box1Final.Extent[i] = box1.extent(i);
    }

    const int* b0Index = box0Cfg.mIndex;
    const int* b1Index = box1Cfg.mIndex;

    if (side == IntrConfiguration<Real>::LEFT)
    {
        // box1 on left of box0
        if (box0Cfg.mMap == IntrConfiguration<Real>::m1_1)
        {
            quantity = 1;
            P[0] = GetPointFromIndex(b0Index[0], box0Final);
        }
        else if (box1Cfg.mMap == IntrConfiguration<Real>::m1_1)
        {
            quantity = 1;
            P[0] = GetPointFromIndex(b1Index[7], box1Final);
        }
        else if (box0Cfg.mMap == IntrConfiguration<Real>::m2_2)
        {
            if (box1Cfg.mMap == IntrConfiguration<Real>::m2_2)
            {
                // box0edge-box1edge intersection
                Vec<3,Real> edge0[2], edge1[2];
                edge0[0] = GetPointFromIndex(b0Index[0], box0Final);
                edge0[1] = GetPointFromIndex(b0Index[1], box0Final);
                edge1[0] = GetPointFromIndex(b1Index[6], box1Final);
                edge1[1] = GetPointFromIndex(b1Index[7], box1Final);
                SegmentSegment(edge0, edge1, quantity, P);
            }
            else // box1Cfg.mMap == IntrConfiguration<Real>::m44
            {
                // box0edge-box1face intersection
                Vec<3,Real> edge0[2], face1[4];
                edge0[0] = GetPointFromIndex(b0Index[0], box0Final);
                edge0[1] = GetPointFromIndex(b0Index[1], box0Final); 
                face1[0] = GetPointFromIndex(b1Index[4], box1Final); 
                face1[1] = GetPointFromIndex(b1Index[5], box1Final); 
                face1[2] = GetPointFromIndex(b1Index[6], box1Final); 
                face1[3] = GetPointFromIndex(b1Index[7], box1Final); 
                CoplanarSegmentRectangle(edge0, face1, quantity, P);
            }
        }
        else // box0Cfg.mMap == IntrConfiguration<Real>::m44
        {
            if (box1Cfg.mMap == IntrConfiguration<Real>::m2_2)
            {
                // box0face-box1edge intersection
                Vec<3,Real> face0[4], edge1[2];
                face0[0] = GetPointFromIndex(b0Index[0], box0Final);
                face0[1] = GetPointFromIndex(b0Index[1], box0Final);
                face0[2] = GetPointFromIndex(b0Index[2], box0Final);
                face0[3] = GetPointFromIndex(b0Index[3], box0Final);
                edge1[0] = GetPointFromIndex(b1Index[6], box1Final);
                edge1[1] = GetPointFromIndex(b1Index[7], box1Final);
                CoplanarSegmentRectangle(edge1, face0, quantity, P);
            }
            else
            {
                // box0face-box1face intersection
                Vec<3,Real> face0[4], face1[4];
                face0[0] = GetPointFromIndex(b0Index[0], box0Final);
                face0[1] = GetPointFromIndex(b0Index[1], box0Final);
                face0[2] = GetPointFromIndex(b0Index[2], box0Final);
                face0[3] = GetPointFromIndex(b0Index[3], box0Final);
                face1[0] = GetPointFromIndex(b1Index[4], box1Final);
                face1[1] = GetPointFromIndex(b1Index[5], box1Final);
                face1[2] = GetPointFromIndex(b1Index[6], box1Final);
                face1[3] = GetPointFromIndex(b1Index[7], box1Final);
                CoplanarRectangleRectangle(face0, face1, quantity, P);
            }
        }
    }
    else // side == RIGHT 
    {
        // box1 on right of box0
        if (box0Cfg.mMap == IntrConfiguration<Real>::m1_1)
        {
            quantity = 1;
            P[0] = GetPointFromIndex(b0Index[7], box0Final);
        }
        else if (box1Cfg.mMap == IntrConfiguration<Real>::m1_1)
        {
            quantity = 1;
            P[0] = GetPointFromIndex(b1Index[0], box1Final);
        }
        else if (box0Cfg.mMap == IntrConfiguration<Real>::m2_2)
        {
            if (box1Cfg.mMap == IntrConfiguration<Real>::m2_2)
            {
                // box0edge-box1edge intersection
                Vec<3,Real> edge0[2], edge1[2];
                edge0[0] = GetPointFromIndex(b0Index[6], box0Final);
                edge0[1] = GetPointFromIndex(b0Index[7], box0Final);
                edge1[0] = GetPointFromIndex(b1Index[0], box1Final);
                edge1[1] = GetPointFromIndex(b1Index[1], box1Final);
                SegmentSegment(edge0,edge1,quantity,P);
            }
            else // box1Cfg.mMap == IntrConfiguration<Real>::m44
            {
                // box0edge-box1face intersection
                Vec<3,Real> edge0[2], face1[4];
                edge0[0] = GetPointFromIndex(b0Index[6], box0Final);
                edge0[1] = GetPointFromIndex(b0Index[7], box0Final);
                face1[0] = GetPointFromIndex(b1Index[0], box1Final);
                face1[1] = GetPointFromIndex(b1Index[1], box1Final);
                face1[2] = GetPointFromIndex(b1Index[2], box1Final);
                face1[3] = GetPointFromIndex(b1Index[3], box1Final);
                CoplanarSegmentRectangle(edge0, face1, quantity, P);
            }
        }
        else // box0Cfg.mMap == IntrConfiguration<Real>::m44
        {
            if (box1Cfg.mMap == IntrConfiguration<Real>::m2_2)
            {
                // box0face-box1edge intersection
                Vec<3,Real> face0[4], edge1[2];
                face0[0] = GetPointFromIndex(b0Index[4], box0Final);
                face0[1] = GetPointFromIndex(b0Index[5], box0Final);
                face0[2] = GetPointFromIndex(b0Index[6], box0Final);
                face0[3] = GetPointFromIndex(b0Index[7], box0Final);
                edge1[0] = GetPointFromIndex(b1Index[0], box1Final);
                edge1[1] = GetPointFromIndex(b1Index[1], box1Final);
                CoplanarSegmentRectangle(edge1, face0, quantity, P);
            }
            else // box1Cfg.mMap == IntrConfiguration<Real>::m44
            {
                // box0face-box1face intersection
                Vec<3,Real> face0[4], face1[4];
                face0[0] = GetPointFromIndex(b0Index[4], box0Final);
                face0[1] = GetPointFromIndex(b0Index[5], box0Final);
                face0[2] = GetPointFromIndex(b0Index[6], box0Final);
                face0[3] = GetPointFromIndex(b0Index[7], box0Final);
                face1[0] = GetPointFromIndex(b1Index[0], box1Final);
                face1[1] = GetPointFromIndex(b1Index[1], box1Final);
                face1[2] = GetPointFromIndex(b1Index[2], box1Final);
                face1[3] = GetPointFromIndex(b1Index[3], box1Final);
                CoplanarRectangleRectangle(face0, face1, quantity, P);
            }
        }
    }
}
//----------------------------------------------------------------------------
template <class TDataTypes>
FindContactSet<TDataTypes>::FindContactSet (const Box& box0,
    const Box& box1, int side, const IntrConfiguration<Real>& box0Cfg,
    const IntrConfiguration<Real>& box1Cfg, const Vec<3,Real>& box0Velocity,
    const Vec<3,Real>& box1Velocity, Real tfirst, int& quantity,
    Vec<3,Real>* POnFirst,Vec<3,Real>* POnSecond)
{
    Vec<3,Real> b02b1(tfirst*(box0Velocity - box1Velocity));

    const int* b0Index = box0Cfg.mIndex;
    const int* b1Index = box1Cfg.mIndex;

    if (side == IntrConfiguration<Real>::LEFT)
    {
        // box1 on left of box0
        if (box0Cfg.mMap == IntrConfiguration<Real>::m1_1)
        {
            quantity = 1;
            POnFirst[0] = getPointFromIndex(b0Index[0], box0);
            POnSecond[0] = POnFirst[0] + b02b1;
        }
        else if (box1Cfg.mMap == IntrConfiguration<Real>::m1_1)
        {
            quantity = 1;
            POnSecond[0] = getPointFromIndex(b1Index[7], box1);
            POnFirst[0] = POnSecond[0] - b02b1;
        }
        else if (box0Cfg.mMap == IntrConfiguration<Real>::m2_2)
        {
            if (box1Cfg.mMap == IntrConfiguration<Real>::m2_2)
            {
                std::cout<<"box0edge-box1edge intersection MAYBE..."<<std::endl;
                // box0edge-box1edge intersection
//                Vec<3,Real> edge0[2], edge1[2];
//                edge0[0] = GetPointFromIndex(b0Index[0], box0);
//                edge0[1] = GetPointFromIndex(b0Index[1], box0);
//                edge1[0] = GetPointFromIndex(b1Index[6], box1);
//                edge1[1] = GetPointFromIndex(b1Index[7], box1);
//                SegmentSegment(edge0, edge1, quantity, P);
            }
            else // box1Cfg.mMap == IntrConfiguration<Real>::m44
            {
                std::cout<<"box0edge-box1face intersection"<<std::endl;
                // box0edge-box1face intersection
//                Vec<3,Real> edge0[2], face1[4];
//                edge0[0] = GetPointFromIndex(b0Index[0], box0);
//                edge0[1] = GetPointFromIndex(b0Index[1], box0);
//                face1[0] = GetPointFromIndex(b1Index[4], box1);
//                face1[1] = GetPointFromIndex(b1Index[5], box1);
//                face1[2] = GetPointFromIndex(b1Index[6], box1);
//                face1[3] = GetPointFromIndex(b1Index[7], box1);
//                CoplanarSegmentRectangle(edge0, face1, quantity, P);
            }
        }
        else // box0Cfg.mMap == IntrConfiguration<Real>::m44
        {
            if (box1Cfg.mMap == IntrConfiguration<Real>::m2_2)
            {
                std::cout<<"box0face-box1edge intersection"<<std::endl;
                // box0face-box1edge intersection
//                Vec<3,Real> face0[4], edge1[2];
//                face0[0] = GetPointFromIndex(b0Index[0], box0);
//                face0[1] = GetPointFromIndex(b0Index[1], box0);
//                face0[2] = GetPointFromIndex(b0Index[2], box0);
//                face0[3] = GetPointFromIndex(b0Index[3], box0);
//                edge1[0] = GetPointFromIndex(b1Index[6], box1);
//                edge1[1] = GetPointFromIndex(b1Index[7], box1);
//                CoplanarSegmentRectangle(edge1, face0, quantity, P);
            }
            else
            {
                std::cout<<"box0face-box1face intersection"<<std::endl;
                // box0face-box1face intersection
//                Vec<3,Real> face0[4], face1[4];
//                face0[0] = GetPointFromIndex(b0Index[0], box0);
//                face0[1] = GetPointFromIndex(b0Index[1], box0);
//                face0[2] = GetPointFromIndex(b0Index[2], box0);
//                face0[3] = GetPointFromIndex(b0Index[3], box0);
//                face1[0] = GetPointFromIndex(b1Index[4], box1);
//                face1[1] = GetPointFromIndex(b1Index[5], box1);
//                face1[2] = GetPointFromIndex(b1Index[6], box1);
//                face1[3] = GetPointFromIndex(b1Index[7], box1);
//                CoplanarRectangleRectangle(face0, face1, quantity, P);
            }
        }
    }
    else // side == RIGHT
    {
        // box1 on right of box0
        if (box0Cfg.mMap == IntrConfiguration<Real>::m1_1)
        {
            quantity = 1;
            POnFirst[0] = getPointFromIndex(b0Index[7], box0);
            POnSecond[0] = POnFirst[0] + b02b1;
        }
        else if (box1Cfg.mMap == IntrConfiguration<Real>::m1_1)
        {
            quantity = 1;
            POnSecond[0] = getPointFromIndex(b1Index[0], box1);
            POnFirst[0] = POnSecond[0] - b02b1;
        }
        else if (box0Cfg.mMap == IntrConfiguration<Real>::m2_2)
        {
            if (box1Cfg.mMap == IntrConfiguration<Real>::m2_2)
            {
                std::cout<<"box0edge-box1edge intersection THIS ONE"<<std::endl;
                // box0edge-box1edge intersection
//                Vec<3,Real> edge0[2], edge1[2];
//                edge0[0] = GetPointFromIndex(b0Index[6], box0);
//                edge0[1] = GetPointFromIndex(b0Index[7], box0);
//                edge1[0] = GetPointFromIndex(b1Index[0], box1);
//                edge1[1] = GetPointFromIndex(b1Index[1], box1);
//                SegmentSegment(edge0,edge1,quantity,P);
            }
            else // box1Cfg.mMap == IntrConfiguration<Real>::m44
            {
                std::cout<<"box0edge-box1face intersection"<<std::endl;
                // box0edge-box1face intersection
//                Vec<3,Real> edge0[2], face1[4];
//                edge0[0] = GetPointFromIndex(b0Index[6], box0);
//                edge0[1] = GetPointFromIndex(b0Index[7], box0);
//                face1[0] = GetPointFromIndex(b1Index[0], box1);
//                face1[1] = GetPointFromIndex(b1Index[1], box1);
//                face1[2] = GetPointFromIndex(b1Index[2], box1);
//                face1[3] = GetPointFromIndex(b1Index[3], box1);
//                CoplanarSegmentRectangle(edge0, face1, quantity, P);
            }
        }
        else // box0Cfg.mMap == IntrConfiguration<Real>::m44
        {
            if (box1Cfg.mMap == IntrConfiguration<Real>::m2_2)
            {
                std::cout<<"box0face-box1edge intersection"<<std::endl;
                // box0face-box1edge intersection
//                Vec<3,Real> face0[4], edge1[2];
//                face0[0] = GetPointFromIndex(b0Index[4], box0);
//                face0[1] = GetPointFromIndex(b0Index[5], box0);
//                face0[2] = GetPointFromIndex(b0Index[6], box0);
//                face0[3] = GetPointFromIndex(b0Index[7], box0);
//                edge1[0] = GetPointFromIndex(b1Index[0], box1);
//                edge1[1] = GetPointFromIndex(b1Index[1], box1);
//                CoplanarSegmentRectangle(edge1, face0, quantity, P);
            }
            else // box1Cfg.mMap == IntrConfiguration<Real>::m44
            {
                std::cout<<"box0face-box1face intersection"<<std::endl;
                // box0face-box1face intersection
//                Vec<3,Real> face0[4], face1[4];
//                face0[0] = GetPointFromIndex(b0Index[4], box0);
//                face0[1] = GetPointFromIndex(b0Index[5], box0);
//                face0[2] = GetPointFromIndex(b0Index[6], box0);
//                face0[3] = GetPointFromIndex(b0Index[7], box0);
//                face1[0] = GetPointFromIndex(b1Index[0], box1);
//                face1[1] = GetPointFromIndex(b1Index[1], box1);
//                face1[2] = GetPointFromIndex(b1Index[2], box1);
//                face1[3] = GetPointFromIndex(b1Index[3], box1);
//                CoplanarRectangleRectangle(face0, face1, quantity, P);
            }
        }
    }
}
//----------------------------------------------------------------------------
template <class TDataTypes>
void FindContactSet<TDataTypes>::ColinearSegments (const Vec<3,Real> segment0[2],
    const Vec<3,Real> segment1[2], int& quantity, Vec<3,Real>* P)
{
    // The potential intersection is initialized to segment0 and clipped
    // against segment1.
    quantity = 2;
    for (int i = 0; i < 2; ++i)
    {
        P[i] = segment0[i];
    }

    // point 0
    Vec<3,Real> V = segment1[1] - segment1[0];
    Real c = V * segment1[0];
    ClipConvexPolygonAgainstPlane(V, c, quantity, P);

    // point 1
    V = -V;
    c = V * segment1[1];
    ClipConvexPolygonAgainstPlane(V, c, quantity, P);
}
//----------------------------------------------------------------------------
template <class TDataTypes>
void FindContactSet<TDataTypes>::SegmentThroughPlane (
    const Vec<3,Real> segment[2], const Vec<3,Real>& planeOrigin,
    const Vec<3,Real>& planeNormal, int& quantity, Vec<3,Real>* P)
{
    quantity = 1;

    Real u = planeNormal * planeOrigin;
    Real v0 = planeNormal * segment[0];
    Real v1 = planeNormal * segment[1];

    // Now that there it has been reduced to a 1-dimensional problem via
    // projection, it becomes easy to find the ratio along V that V 
    // intersects with U.
    Real ratio = (u - v0)/(v1 - v0);
    P[0] = segment[0] + ratio*(segment[1] - segment[0]);
}
//----------------------------------------------------------------------------
template <class TDataTypes>
void FindContactSet<TDataTypes>::SegmentSegment (const Vec<3,Real> segment0[2],
    const Vec<3,Real> segment1[2], int& quantity, Vec<3,Real>* P)
{
    Vec<3,Real> dir0 = segment0[1] - segment0[0];
    Vec<3,Real> dir1 = segment1[1] - segment1[0];
    Vec<3,Real> normal = dir0.cross(dir1);

    // The comparison is sin(kDir0,kDir1) < epsilon.
    Real sqrLen0 = dir0.norm2();
    Real sqrLen1 = dir1.norm2();
    Real sqrLenN = normal.norm2();
    if (sqrLenN < Math<Real>::ZERO_TOLERANCE()*sqrLen0*sqrLen1)
    {
        ColinearSegments(segment0, segment1, quantity, P);
    }
    else
    {
        SegmentThroughPlane(segment1, segment0[0],
            normal.cross(segment0[1]-segment0[0]), quantity, P);
    }
}
//----------------------------------------------------------------------------
template <class TDataTypes>
void FindContactSet<TDataTypes>::ColinearSegmentTriangle (
    const Vec<3,Real> segment[2], const Vec<3,Real> triangle[3],
    int& quantity, Vec<3,Real>* P)
{
    // The potential intersection is initialized to the line segment and then
    // clipped against the three sides of the tri
    quantity = 2;
    int i;
    for (i = 0; i < 2; ++i)
    {
        P[i] = segment[i];
    }

    Vec<3,Real> side[3] =
    {
        triangle[1] - triangle[0],
        triangle[2] - triangle[1],
        triangle[0] - triangle[2]
    };

    Vec<3,Real> normal = side[0].cross(side[1]);
    for (i = 0; i < 3; ++i)
    {
        // Normal pointing inside the triangle.
        Vec<3,Real> sideN = normal.cross(side[i]);
        Real constant = sideN * triangle[i];
        ClipConvexPolygonAgainstPlane(sideN, constant, quantity, P);
    }
}
//----------------------------------------------------------------------------
template <class TDataTypes>
void FindContactSet<TDataTypes>::CoplanarSegmentRectangle (
    const Vec<3,Real> segment[2], const Vec<3,Real> rectangle[4],
    int& quantity, Vec<3,Real>* P)
{
    // The potential intersection is initialized to the line segment and then
    // clipped against the four sides of the rect
    quantity = 2;
    for (int i = 0; i < 2; ++i)
    {
        P[i] = segment[i];
    }

    for (int i0 = 3, i1 = 0; i1 < 4; i0 = i1++)
    {
        Vec<3,Real> normal = rectangle[i1] - rectangle[i0];
        Real constant = normal * rectangle[i0];
        ClipConvexPolygonAgainstPlane(normal, constant, quantity, P);
    }
}
//----------------------------------------------------------------------------
template <class TDataTypes>
void FindContactSet<TDataTypes>::CoplanarTriangleRectangle (
    const Vec<3,Real> triangle[3], const Vec<3,Real> rectangle[4],
    int& quantity, Vec<3,Real>* P)
{
    // The potential intersection is initialized to the triangle, and then
    // clipped against the sides of the box
    quantity = 3;
    for (int i = 0; i < 3; ++i)
    {
        P[i] = triangle[i];
    }

    for (int i0 = 3, i1 = 0; i1 < 4; i0 = i1++)
    {
        Vec<3,Real> normal = rectangle[i1] - rectangle[i0];
        Real constant = normal * rectangle[i0];
        ClipConvexPolygonAgainstPlane(normal, constant, quantity, P);
    }
}
//----------------------------------------------------------------------------
template <class TDataTypes>
void FindContactSet<TDataTypes>::CoplanarRectangleRectangle (
    const Vec<3,Real> rectangle0[4], const Vec<3,Real> rectangle1[4],
    int& quantity, Vec<3,Real>* P)
{
    // The potential intersection is initialized to face 0, and then clipped
    // against the four sides of face 1.
    quantity = 4;
    for (int i = 0; i < 4; ++i)
    {
        P[i] = rectangle0[i];
    }

    for (int i0 = 3, i1 = 0; i1 < 4; i0 = i1++)
    {
        Vec<3,Real> normal = rectangle1[i1] - rectangle1[i0];
        Real constant = normal * rectangle1[i0];
        ClipConvexPolygonAgainstPlane(normal, constant, quantity, P);
    }
}
//----------------------------------------------------------------------------
template <class Real>
void ClipConvexPolygonAgainstPlane (const Vec<3,Real>& normal,
    Real constant, int& quantity, Vec<3,Real>* P)
{
    // The input vertices are assumed to be in counterclockwise order.  The
    // ordering is an invariant of this function.  The size of array P is
    // assumed to be large enough to store the clipped polygon vertices.

    // test on which side of line are the vertices
    int positive = 0, negative = 0, pIndex = -1;
    int currQuantity = quantity;

    Real test[8];
    int i;
    for (i = 0; i < quantity; ++i)
    {

        // An epsilon is used here because it is possible for the dot product
        // and 'constant' to be exactly equal to each other (in theory), but
        // differ slightly because of floating point problems.  Thus, add a
        // little to the test number to push actually equal numbers over the
        // edge towards the positive.

        // TODO: This should probably be a relative tolerance.  Multiplying
        // by the constant is probably not the best way to do this.
        test[i] = normal * P[i] - constant +
            fabs(constant)*Math<Real>::ZERO_TOLERANCE();

        if (test[i] >= (Real)0)
        {
            ++positive;
            if (pIndex < 0)
            {
                pIndex = i;
            }
        }
        else
        {
            ++negative;
        }
    }

    if (quantity == 2)
    {
        // Lines are a little different, in that clipping the segment
        // cannot create a new segment, as clipping a polygon would
        if (positive > 0)
        {
            if (negative > 0) 
            {
                int clip;

                if (pIndex == 0)
                {
                    // vertex0 positive, vertex1 is clipped
                    clip = 1;
                }
                else // pIndex == 1
                {
                    // vertex1 positive, vertex0 clipped
                    clip = 0;
                }

                Real t = test[pIndex]/(test[pIndex] - test[clip]);
                P[clip] = P[pIndex] + t*(P[clip] - P[pIndex]);
            }
            // otherwise both positive, no clipping
        }
        else
        {
            // Assert:  The entire line is clipped, but we should not
            // get here.
            quantity = 0;
        }
    }
    else
    {
        if (positive > 0)
        {
            if (negative > 0)
            {
                // plane transversely intersects polygon
                Vec<3,Real> CV[8];
                int cQuantity = 0, cur, prv;
                Real t;

                if (pIndex > 0)
                {
                    // first clip vertex on line
                    cur = pIndex;
                    prv = cur - 1;
                    t = test[cur]/(test[cur] - test[prv]);
                    CV[cQuantity++] = P[cur] + t*(P[prv] - P[cur]);

                    // vertices on positive side of line
                    while (cur < currQuantity && test[cur] >= (Real)0)
                    {
                        CV[cQuantity++] = P[cur++];
                    }

                    // last clip vertex on line
                    if (cur < currQuantity)
                    {
                        prv = cur - 1;
                    }
                    else
                    {
                        cur = 0;
                        prv = currQuantity - 1;
                    }
                    t = test[cur]/(test[cur] - test[prv]);
                    CV[cQuantity++] = P[cur] + t*(P[prv] - P[cur]);
                }
                else  // pIndex is 0
                {
                    // vertices on positive side of line
                    cur = 0;
                    while (cur < currQuantity && test[cur] >= (Real)0)
                    {
                        CV[cQuantity++] = P[cur++];
                    }

                    // last clip vertex on line
                    prv = cur - 1;
                    t = test[cur]/(test[cur] - test[prv]);
                    CV[cQuantity++] = P[cur] + t*(P[prv] - P[cur]);

                    // skip vertices on negative side
                    while (cur < currQuantity && test[cur] < (Real)0)
                    {
                        cur++;
                    }

                    // first clip vertex on line
                    if (cur < currQuantity)
                    {
                        prv = cur - 1;
                        t = test[cur]/(test[cur] - test[prv]);
                        CV[cQuantity++] = P[cur] + t*(P[prv] - P[cur]);

                        // vertices on positive side of line
                        while (cur < currQuantity && test[cur] >= (Real)0)
                        {
                            CV[cQuantity++] = P[cur++];
                        }
                    }
                    else
                    {
                        // cur = 0
                        prv = currQuantity - 1;
                        t = test[0]/(test[0] - test[prv]);
                        CV[cQuantity++] = P[0] + t*(P[prv] - P[0]);
                    }
                }

                currQuantity = cQuantity;
                memcpy(P, CV, cQuantity*sizeof(Vec<3,Real>));
            }
            // else polygon fully on positive side of plane, nothing to do

            quantity = currQuantity;
        }
        else
        {
            // Polygon does not intersect positive side of plane, clip all.
            // This should not ever happen if called by the findintersect
            // routines after an intersection has been determined.
            quantity = 0;
        }    
    }
}
//----------------------------------------------------------------------------
template <class TReal>
Vec<3,TReal> GetPointFromIndex (int index, const MyBox<TReal> &box)
{
    Vec<3,TReal> point = box.Center;

    if (index & 4)
    {
        point += box.Extent[2]*box.Axis[2];
    }
    else
    {
        point -= box.Extent[2]*box.Axis[2];
    }

    if (index & 2)
    {
        point += box.Extent[1]*box.Axis[1];
    }
    else
    {
        point -= box.Extent[1]*box.Axis[1];
    }

    if (index & 1)
    {
        point += box.Extent[0]*box.Axis[0];
    }
    else
    {
        point -= box.Extent[0]*box.Axis[0];
    }

    return point;
}

template <typename TDataTypes>
Vec<3,typename TDataTypes::Real> getPointFromIndex(int index, const TOBB<TDataTypes>& box)
{
    Vec<3,typename TDataTypes::Real> point = box.center();

    if (index & 4)
    {
        point += box.extent(2)*box.axis(2);
    }
    else
    {
        point -= box.extent(2)*box.axis(2);
    }

    if (index & 2)
    {
        point += box.extent(1)*box.axis(1);
    }
    else
    {
        point -= box.extent(1)*box.axis(1);
    }

    if (index & 1)
    {
        point += box.extent(0)*box.axis(0);
    }
    else
    {
        point -= box.extent(0)*box.axis(0);
    }

    return point;
}

template <class TReal>
void MyBox<TReal>::showVertices()const{
    std::vector<Vec<3,TReal> > vs;
    Vec<3,TReal> a0(Axis[0] * Extent[0]);
    Vec<3,TReal> a1(Axis[1] * Extent[1]);
    Vec<3,TReal> a2(Axis[2] * Extent[2]);

    vs.push_back(Center - a0 + a1 - a2);
    vs.push_back(Center + a0 + a1 - a2);
    vs.push_back(Center + a0 + a1 + a2);
    vs.push_back(Center - a0 + a1 + a2);
    vs.push_back(Center - a0 - a1 - a2);
    vs.push_back(Center + a0 - a1 - a2);
    vs.push_back(Center + a0 - a1 + a2);
    vs.push_back(Center - a0 - a1 + a2);

    for(int i = 0 ; i < 8 ; ++i){
        std::cout<<"    "<<vs[i]<<std::endl;
    }
}

}
}
}
