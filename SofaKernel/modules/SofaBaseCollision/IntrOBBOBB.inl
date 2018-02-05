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
// File modified from GeometricTools
// http://www.geometrictools.com/

#include "IntrOBBOBB.h"
#include "IntrUtility3.h"
#include <limits>

namespace sofa{
namespace component{
namespace collision{

//----------------------------------------------------------------------------
template <class TDataTypes>
TIntrOBBOBB<TDataTypes>::TIntrOBBOBB (const Box& box0,
    const Box& box1)
    :
    mBox0(&box0),
    mBox1(&box1)
{
    _is_colliding = false;
}
//----------------------------------------------------------------------------
template <class TDataTypes>
const typename TIntrOBBOBB<TDataTypes>::Box & TIntrOBBOBB<TDataTypes>::GetBox0 () const
{
    return *mBox0;
}
//----------------------------------------------------------------------------
template <class TDataTypes>
const typename TIntrOBBOBB<TDataTypes>::Box &  TIntrOBBOBB<TDataTypes>::GetBox1 () const
{
    return *mBox1;
}
//----------------------------------------------------------------------------
template <class TDataTypes>
bool TIntrOBBOBB<TDataTypes>::Test ()
{
    // Cutoff for cosine of angles between box axes.  This is used to catch
    // the cases when at least one pair of axes are parallel.  If this
    // happens, there is no need to test for separation along the
    // Cross(A[i],B[j]) directions.
    const Real cutoff = (Real)1 - IntrUtil<Real>::ZERO_TOLERANCE();
    bool existsParallelPair = false;
    int i;

    // Convenience variables.
    Coord A[3];
    mBox0->axes(A);
    Coord B[3];
    mBox1->axes(B);

    Coord EA = mBox0->extents();
    Coord EB = mBox1->extents();

    // Compute difference of box centers, D = C1-C0.
    Coord D = mBox1->center() - mBox0->center();

    Real C[3][3];     // matrix C = A^T B, c_{ij} = Dot(A_i,B_j)
    Real AbsC[3][3];  // |c_{ij}|
    Real AD[3];       // Dot(A_i,D)
    Real r0, r1, r;   // interval radii and distance between centers
    Real r01;         // = R0 + R1

    // axis C0+t*A0
    for (i = 0; i < 3; ++i)
    {
        C[0][i] = A[0] * B[i];
        AbsC[0][i] = fabs(C[0][i]);
        if (AbsC[0][i] > cutoff)
        {
            existsParallelPair = true;
        }
    }
    AD[0] = A[0] * D;
    r = fabs(AD[0]);
    r1 = EB[0]*AbsC[0][0] + EB[1]*AbsC[0][1] + EB[2]*AbsC[0][2];
    r01 = EA[0] + r1;
    if (r > r01)
    {
        return false;
    }

    // axis C0+t*A1
    for (i = 0; i < 3; ++i)
    {
        C[1][i] = A[1] * B[i];
        AbsC[1][i] = fabs(C[1][i]);
        if (AbsC[1][i] > cutoff)
        {
            existsParallelPair = true;
        }
    }
    AD[1] = A[1] * D;
    r = fabs(AD[1]);
    r1 = EB[0]*AbsC[1][0] + EB[1]*AbsC[1][1] + EB[2]*AbsC[1][2];
    r01 = EA[1] + r1;
    if (r > r01)
    {
        return false;
    }

    // axis C0+t*A2
    for (i = 0; i < 3; ++i)
    {
        C[2][i] = A[2] * B[i];
        AbsC[2][i] = fabs(C[2][i]);
        if (AbsC[2][i] > cutoff)
        {
            existsParallelPair = true;
        }
    }
    AD[2] = A[2] * D;
    r = fabs(AD[2]);
    r1 = EB[0]*AbsC[2][0] + EB[1]*AbsC[2][1] + EB[2]*AbsC[2][2];
    r01 = EA[2] + r1;
    if (r > r01)
    {
        return false;
    }

    // axis C0+t*B0
    r = fabs(B[0] * D);
    r0 = EA[0]*AbsC[0][0] + EA[1]*AbsC[1][0] + EA[2]*AbsC[2][0];
    r01 = r0 + EB[0];
    if (r > r01)
    {
        return false;
    }

    // axis C0+t*B1
    r = fabs(B[1] * D);
    r0 = EA[0]*AbsC[0][1] + EA[1]*AbsC[1][1] + EA[2]*AbsC[2][1];
    r01 = r0 + EB[1];
    if (r > r01)
    {
        return false;
    }

    // axis C0+t*B2
    r = fabs(B[2] * D);
    r0 = EA[0]*AbsC[0][2] + EA[1]*AbsC[1][2] + EA[2]*AbsC[2][2];
    r01 = r0 + EB[2];
    if (r > r01)
    {
        return false;
    }

    // At least one pair of box axes was parallel, so the separation is
    // effectively in 2D where checking the "edge" normals is sufficient for
    // the separation of the boxes.
    if (existsParallelPair)
    {
        return true;
    }

    // axis C0+t*A0xB0
    r = fabs(AD[2]*C[1][0] - AD[1]*C[2][0]);
    r0 = EA[1]*AbsC[2][0] + EA[2]*AbsC[1][0];
    r1 = EB[1]*AbsC[0][2] + EB[2]*AbsC[0][1];
    r01 = r0 + r1;
    if (r > r01)
    {
        return false;
    }

    // axis C0+t*A0xB1
    r = fabs(AD[2]*C[1][1] - AD[1]*C[2][1]);
    r0 = EA[1]*AbsC[2][1] + EA[2]*AbsC[1][1];
    r1 = EB[0]*AbsC[0][2] + EB[2]*AbsC[0][0];
    r01 = r0 + r1;
    if (r > r01)
    {
        return false;
    }

    // axis C0+t*A0xB2
    r = fabs(AD[2]*C[1][2] - AD[1]*C[2][2]);
    r0 = EA[1]*AbsC[2][2] + EA[2]*AbsC[1][2];
    r1 = EB[0]*AbsC[0][1] + EB[1]*AbsC[0][0];
    r01 = r0 + r1;
    if (r > r01)
    {
        return false;
    }

    // axis C0+t*A1xB0
    r = fabs(AD[0]*C[2][0] - AD[2]*C[0][0]);
    r0 = EA[0]*AbsC[2][0] + EA[2]*AbsC[0][0];
    r1 = EB[1]*AbsC[1][2] + EB[2]*AbsC[1][1];
    r01 = r0 + r1;
    if (r > r01)
    {
        return false;
    }

    // axis C0+t*A1xB1
    r = fabs(AD[0]*C[2][1] - AD[2]*C[0][1]);
    r0 = EA[0]*AbsC[2][1] + EA[2]*AbsC[0][1];
    r1 = EB[0]*AbsC[1][2] + EB[2]*AbsC[1][0];
    r01 = r0 + r1;
    if (r > r01)
    {
        return false;
    }

    // axis C0+t*A1xB2
    r = fabs(AD[0]*C[2][2] - AD[2]*C[0][2]);
    r0 = EA[0]*AbsC[2][2] + EA[2]*AbsC[0][2];
    r1 = EB[0]*AbsC[1][1] + EB[1]*AbsC[1][0];
    r01 = r0 + r1;
    if (r > r01)
    {
        return false;
    }

    // axis C0+t*A2xB0
    r = fabs(AD[1]*C[0][0] - AD[0]*C[1][0]);
    r0 = EA[0]*AbsC[1][0] + EA[1]*AbsC[0][0];
    r1 = EB[1]*AbsC[2][2] + EB[2]*AbsC[2][1];
    r01 = r0 + r1;
    if (r > r01)
    {
        return false;
    }

    // axis C0+t*A2xB1
    r = fabs(AD[1]*C[0][1] - AD[0]*C[1][1]);
    r0 = EA[0]*AbsC[1][1] + EA[1]*AbsC[0][1];
    r1 = EB[0]*AbsC[2][2] + EB[2]*AbsC[2][0];
    r01 = r0 + r1;
    if (r > r01)
    {
        return false;
    }

    // axis C0+t*A2xB2
    r = fabs(AD[1]*C[0][2] - AD[0]*C[1][2]);
    r0 = EA[0]*AbsC[1][2] + EA[1]*AbsC[0][2];
    r1 = EB[0]*AbsC[2][1] + EB[1]*AbsC[2][0];
    r01 = r0 + r1;
    if (r > r01)
    {
        return false;
    }

    return true;
}
//----------------------------------------------------------------------------
template <class TDataTypes>
bool TIntrOBBOBB<TDataTypes>::Test (Real tmax,
    const defaulttype::Vec<3,Real>& velocity0, const defaulttype::Vec<3,Real>& velocity1)
{
    if (velocity0 == velocity1)
    {
        if (Test())
        {
            mContactTime = (Real)0;
            return true;
        }
        return false;
    }

    // Cutoff for cosine of angles between box axes.  This is used to catch
    // the cases when at least one pair of axes are parallel.  If this
    // happens, there is no need to include the cross-product axes for
    // separation.
    const Real cutoff = (Real)1 - IntrUtil<Real>::ZERO_TOLERANCE();
    bool existsParallelPair = false;

    // convenience variables
    // Convenience variables.
    Coord A[3];
    mBox0->axes(A);
    Coord B[3];
    mBox1->axes(B);

    const Coord & EA = mBox0->extents();
    const Coord & EB = mBox1->extents();

    // Compute difference of box centers, D = C1-C0.
    Coord D = mBox1->center() - mBox0->center();

    defaulttype::Vec<3,Real> W = velocity1 - velocity0;
    Real C[3][3];     // matrix C = A^T B, c_{ij} = Dot(A_i,B_j)
    Real AbsC[3][3];  // |c_{ij}|
    Real AD[3];       // Dot(A_i,D)
    Real AW[3];       // Dot(A_i,W)
    Real min0, max0, min1, max1, center, radius, speed;
    int i, j;

    mContactTime = (Real)0;
    Real tlast = std::numeric_limits<Real>::max();

    // axes C0+t*A[i]
    for (i = 0; i < 3; ++i)
    {
        for (j = 0; j < 3; ++j)
        {
            C[i][j] = A[i] * B[j];
            AbsC[i][j] = fabs(C[i][j]);
            if (AbsC[i][j] > cutoff)
            {
                existsParallelPair = true;
            }
        }
        AD[i] = A[i] * D;
        AW[i] = A[i] * W;
        min0 = -EA[i];
        max0 = +EA[i];
        radius = EB[0]*AbsC[i][0] + EB[1]*AbsC[i][1] + EB[2]*AbsC[i][2];
        min1 = AD[i] - radius;
        max1 = AD[i] + radius;
        speed = AW[i];
        if (IsSeparated(min0, max0, min1, max1, speed, tmax, tlast))
        {
            return false;
        }
    }

    // axes C0+t*B[i]
    for (i = 0; i < 3; ++i)
    {
        radius = EA[0]*AbsC[0][i] + EA[1]*AbsC[1][i] + EA[2]*AbsC[2][i];
        min0 = -radius;
        max0 = +radius;
        center = B[i] * D;
        min1 = center - EB[i];
        max1 = center + EB[i];
        speed = W * B[i];
        if (IsSeparated(min0, max0, min1, max1, speed, tmax, tlast))
        {
            return false;
        }
    }

    // At least one pair of box axes was parallel, so the separation is
    // effectively in 2D where checking the "edge" normals is sufficient for
    // the separation of the boxes.
    if (existsParallelPair)
    {
        return true;
    }

    // axis C0+t*A0xB0
    radius = EA[1]*AbsC[2][0] + EA[2]*AbsC[1][0];
    min0 = -radius;
    max0 = +radius;
    center = AD[2]*C[1][0] - AD[1]*C[2][0];
    radius = EB[1]*AbsC[0][2] + EB[2]*AbsC[0][1];
    min1 = center - radius;
    max1 = center + radius;
    speed = AW[2]*C[1][0] - AW[1]*C[2][0];
    if (IsSeparated(min0, max0, min1, max1, speed, tmax, tlast))
    {
        return false;
    }

    // axis C0+t*A0xB1
    radius = EA[1]*AbsC[2][1] + EA[2]*AbsC[1][1];
    min0 = -radius;
    max0 = +radius;
    center = AD[2]*C[1][1] - AD[1]*C[2][1];
    radius = EB[0]*AbsC[0][2] + EB[2]*AbsC[0][0];
    min1 = center - radius;
    max1 = center + radius;
    speed = AW[2]*C[1][1] - AW[1]*C[2][1];
    if (IsSeparated(min0, max0, min1, max1, speed, tmax, tlast))
    {
        return false;
    }

    // axis C0+t*A0xB2
    radius = EA[1]*AbsC[2][2] + EA[2]*AbsC[1][2];
    min0 = -radius;
    max0 = +radius;
    center = AD[2]*C[1][2] - AD[1]*C[2][2];
    radius = EB[0]*AbsC[0][1] + EB[1]*AbsC[0][0];
    min1 = center - radius;
    max1 = center + radius;
    speed = AW[2]*C[1][2] - AW[1]*C[2][2];
    if (IsSeparated(min0, max0, min1, max1, speed, tmax, tlast))
    {
        return false;
    }

    // axis C0+t*A1xB0
    radius = EA[0]*AbsC[2][0] + EA[2]*AbsC[0][0];
    min0 = -radius;
    max0 = +radius;
    center = AD[0]*C[2][0] - AD[2]*C[0][0];
    radius = EB[1]*AbsC[1][2] + EB[2]*AbsC[1][1];
    min1 = center - radius;
    max1 = center + radius;
    speed = AW[0]*C[2][0] - AW[2]*C[0][0];
    if (IsSeparated(min0, max0, min1, max1, speed, tmax, tlast))
    {
        return false;
    }

    // axis C0+t*A1xB1
    radius = EA[0]*AbsC[2][1] + EA[2]*AbsC[0][1];
    min0 = -radius;
    max0 = +radius;
    center = AD[0]*C[2][1] - AD[2]*C[0][1];
    radius = EB[0]*AbsC[1][2] + EB[2]*AbsC[1][0];
    min1 = center - radius;
    max1 = center + radius;
    speed = AW[0]*C[2][1] - AW[2]*C[0][1];
    if (IsSeparated(min0, max0, min1, max1, speed, tmax, tlast))
    {
        return false;
    }

    // axis C0+t*A1xB2
    radius = EA[0]*AbsC[2][2] + EA[2]*AbsC[0][2];
    min0 = -radius;
    max0 = +radius;
    center = AD[0]*C[2][2] - AD[2]*C[0][2];
    radius = EB[0]*AbsC[1][1] + EB[1]*AbsC[1][0];
    min1 = center - radius;
    max1 = center + radius;
    speed = AW[0]*C[2][2] - AW[2]*C[0][2];
    if (IsSeparated(min0, max0, min1, max1, speed, tmax, tlast))
    {
        return false;
    }

    // axis C0+t*A2xB0
    radius = EA[0]*AbsC[1][0] + EA[1]*AbsC[0][0];
    min0 = -radius;
    max0 = +radius;
    center = AD[1]*C[0][0] - AD[0]*C[1][0];
    radius = EB[1]*AbsC[2][2] + EB[2]*AbsC[2][1];
    min1 = center - radius;
    max1 = center + radius;
    speed = AW[1]*C[0][0] - AW[0]*C[1][0];
    if (IsSeparated(min0, max0, min1, max1, speed, tmax, tlast))
    {
        return false;
    }

    // axis C0+t*A2xB1
    radius = EA[0]*AbsC[1][1] + EA[1]*AbsC[0][1];
    min0 = -radius;
    max0 = +radius;
    center = AD[1]*C[0][1] - AD[0]*C[1][1];
    radius = EB[0]*AbsC[2][2] + EB[2]*AbsC[2][0];
    min1 = center - radius;
    max1 = center + radius;
    speed = AW[1]*C[0][1] - AW[0]*C[1][1];
    if (IsSeparated(min0, max0, min1, max1, speed, tmax, tlast))
    {
        return false;
    }

    // axis C0+t*A2xB2
    radius = EA[0]*AbsC[1][2] + EA[1]*AbsC[0][2];
    min0 = -radius;
    max0 = +radius;
    center = AD[1]*C[0][2] - AD[0]*C[1][2];
    radius = EB[0]*AbsC[2][1] + EB[1]*AbsC[2][0];
    min1 = center - radius;
    max1 = center + radius;
    speed = AW[1]*C[0][2] - AW[0]*C[1][2];
    if (IsSeparated(min0, max0, min1, max1, speed, tmax, tlast))
    {
        return false;
    }

    return true;
}
//----------------------------------------------------------------------------
template <class TDataTypes>
bool TIntrOBBOBB<TDataTypes>::Find (Real dmax)
{
    mContactTime = -std::numeric_limits<Real>::max();

    int i0, i1;
    int side = IntrConfiguration<Real>::NONE;
    IntrConfiguration<Real> box0Cfg, box1Cfg;
    defaulttype::Vec<3,Real> axis;
    bool config_modified;

    // box 0 normals
    for (i0 = 0; i0 < 3; ++i0)
    {
        axis = mBox0->axis(i0);
        if(!IntrAxis<Box>::Find(axis, *mBox0, *mBox1, dmax,
            mContactTime, side, box0Cfg,box1Cfg,config_modified))
            return false;

        if(config_modified){
            _sep_axis = axis;
        }
    }

    // box 1 normals
    for (i1 = 0; i1 < 3; ++i1)
    {
        axis = mBox1->axis(i1);
        if(!IntrAxis<Box>::Find(axis, *mBox0, *mBox1, dmax,
            mContactTime, side, box0Cfg, box1Cfg,config_modified))
                return false;

        if(config_modified){
            _sep_axis = axis;
        }
    }

    // box 0 edges cross box 1 edges
    for (i0 = 0; i0 < 3; ++i0)
    {
        for (i1 = 0; i1 < 3; ++i1)
        {
            axis = (mBox0->axis(i0)).cross(mBox1->axis(i1));

            // Since all axes are unit length (assumed), then can just compare
            // against a constant (not relative) epsilon.
            if (axis.norm2() <= IntrUtil<Real>::ZERO_TOLERANCE())
            {
                // Axis i0 and i1 are parallel.  If any two axes are parallel,
                // then the only comparisons that needed are between the faces
                // themselves.  At this time the faces have already been
                // tested, and without separation, so all further separation
                // tests will show only overlaps.
                FindContactSet<Box,Box>(*mBox0,*mBox1,_sep_axis,side,box0Cfg,box1Cfg,mContactTime,_pt_on_first,_pt_on_second);

                if(side == IntrConfiguration<Real>::LEFT)
                    _sep_axis *= -1.0;

                if(mContactTime < 0){
                    _is_colliding = true;
                }

                return true;
            }

            IntrUtil<Real>::normalize(axis);

            if(!IntrAxis<Box>::Find(axis, *mBox0, *mBox1,
                dmax, mContactTime, side, box0Cfg, box1Cfg,config_modified))
                return false;

            if(config_modified){
                _sep_axis = axis;
            }
        }
    }

    defaulttype::Vec<3,Real> relVelocity = mBox1->v() - mBox0->v();
    // velocity cross box 0 edges
    for (i0 = 0; i0 < 3; ++i0)
    {
        axis = relVelocity.cross(mBox0->axis(i0));

        if(axis.norm2() > IntrUtil<Real>::SQ_ZERO_TOLERANCE()){
            IntrUtil<Real>::normalize(axis);
            if(!IntrAxis<Box>::Find(axis, *mBox0, *mBox1, dmax,
                mContactTime, side, box0Cfg, box1Cfg,config_modified))
               return false;

            if(config_modified){
                _sep_axis = axis;
            }
        }
    }

    // velocity cross box 1 edges
    for (i0 = 0; i0 < 3; ++i0)
    {
        axis = relVelocity.cross(mBox1->axis(i0));

        if(axis.norm2() > IntrUtil<Real>::SQ_ZERO_TOLERANCE()){
            IntrUtil<Real>::normalize(axis);
            if(!IntrAxis<Box>::Find(axis, *mBox0, *mBox1, dmax,
                mContactTime, side, box0Cfg, box1Cfg,config_modified))
               return false;

            if(config_modified){
                _sep_axis = axis;
            }
        }
    }

    if (mContactTime <= (Real)0)
    {
        if(side == IntrConfiguration<Real>::NONE)
            return false;

        _is_colliding = true;
//        Real max_ext_0 = std::max(mBox0->extent(0),std::max(mBox0->extent(1),mBox0->extent(2)));
//        Real max_ext_1 = std::max(mBox1->extent(0),std::max(mBox1->extent(1),mBox1->extent(2)));
//        Real max_ext = std::max(max_ext_0,max_ext_1);

//        if(mContactTime < max_ext){
//            _is_colliding = true;
//        }
//        else{
//            return false;
//        }
    }

    FindContactSet<Box,Box>(*mBox0,*mBox1,_sep_axis,side,box0Cfg,box1Cfg,mContactTime,_pt_on_first,_pt_on_second);

//    if((_pt_on_first - mBox0->center()) * _sep_axis < 0)
//        _sep_axis *= -1;
    if(side == IntrConfiguration<Real>::LEFT)
        _sep_axis *= -1.0;

    return true;
}
//----------------------------------------------------------------------------
template <class TDataTypes>
bool TIntrOBBOBB<TDataTypes>::IsSeparated (Real min0, Real max0, Real min1,
    Real max1, Real speed, Real tmax, Real& tlast)
{
    Real invSpeed, t;

    if (max1 < min0) // box1 initially on left of box0
    {
        if (speed <= (Real)0)
        {
            // The projection intervals are moving apart.
            return true;
        }
        invSpeed = ((Real)1)/speed;

        t = (min0 - max1)*invSpeed;
        if (t > mContactTime)
        {
            mContactTime = t;
        }

        if (mContactTime > tmax)
        {
            // Intervals do not intersect during the specified time.
            return true;
        }

        t = (max0 - min1)*invSpeed;
        if (t < tlast)
        {
            tlast = t;
        }

        if (mContactTime > tlast)
        {
            // Physically inconsistent times--the objects cannot intersect.
            return true;
        }
    }
    else if (max0 < min1) // box1 initially on right of box0
    {
        if (speed >= (Real)0)
        {
            // The projection intervals are moving apart.
            return true;
        }
        invSpeed = ((Real)1)/speed;

        t = (max0 - min1)*invSpeed;
        if (t > mContactTime)
        {
            mContactTime = t;
        }

        if (mContactTime > tmax)
        {
            // Intervals do not intersect during the specified time.
            return true;
        }

        t = (min0 - max1)*invSpeed;
        if (t < tlast)
        {
            tlast = t;
        }

        if (mContactTime > tlast)
        {
            // Physically inconsistent times--the objects cannot intersect.
            return true;
        }
    }
    else // box0 and box1 initially overlap
    {
        if (speed > (Real)0)
        {
            t = (max0 - min1)/speed;
            if (t < tlast)
            {
                tlast = t;
            }

            if (mContactTime > tlast)
            {
                // Physically inconsistent times--the objects cannot
                // intersect.
                return true;
            }
        }
        else if (speed < (Real)0)
        {
            t = (min0 - max1)/speed;
            if (t < tlast)
            {
                tlast = t;
            }

            if (mContactTime > tlast)
            {
                // Physically inconsistent times--the objects cannot
                // intersect.
                return true;
            }
        }
    }

    return false;
}

//----------------------------------------------------------------------------

}
}
}
