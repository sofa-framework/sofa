/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef INTRUTILITY3_INL
#define INTRUTILITY3_INL
#include "IntrUtility3.h"


namespace sofa{
namespace component{
namespace collision{

template <typename Real>
IntrConfiguration<Real> & IntrConfiguration<Real>::operator=(const IntrConfiguration & other){
    this->mMap = other.mMap;

    for(int i = 0 ; i < 8 ; ++i)
        this->mIndex[i] = other.mIndex[i];

    this->mMin = other.mMin;
    this->mMax = other.mMax;

    return *this;
}

template <typename Real>
CapIntrConfiguration<Real> & CapIntrConfiguration<Real>::operator=(const CapIntrConfiguration & other){
    IntrConfiguration<Real>::operator =(other);

    this->axis = other.axis;

    return *this;
}

template <typename Real>
defaulttype::Vec<3,Real> CapIntrConfiguration<Real>::leftContactPoint(const defaulttype::Vec<3,Real> * seg,Real radius)const{
    return seg[this->mIndex[0]] - (axis) * radius;
}

template <typename Real>
defaulttype::Vec<3,Real> CapIntrConfiguration<Real>::rightContactPoint(const defaulttype::Vec<3,Real> * seg,Real radius)const{
    return seg[this->mIndex[1]] + (axis) * radius;
}

template <typename Real>
void CapIntrConfiguration<Real>::leftSegment(const defaulttype::Vec<3,Real> * seg,Real radius, defaulttype::Vec<3,Real> *lseg)const{
    for(int i = 0 ; i < 2 ; ++i)
        lseg[i] = seg[i] - (axis) * radius;
}


template <typename Real>
void CapIntrConfiguration<Real>::rightSegment(const defaulttype::Vec<3,Real> * seg,Real radius,defaulttype::Vec<3,Real> * rseg)const{
    for(int i = 0 ; i < 2 ; ++i)
        rseg[i] = seg[i] + (axis) * radius;
}

template <typename Real>
CapIntrConfiguration<Real>::CapIntrConfiguration(){
    have_naxis = false;
}
//----------------------------------------------------------------------------
template <class TDataTypes>
bool IntrAxis<TOBB<TDataTypes> >::Find (const Coord& axis,
    const Box& box0, const Box& box1,
    Real dmax, Real& dfirst,
    int& side, IntrConfiguration<Real>& box0CfgFinal,
    IntrConfiguration<Real>& box1CfgFinal,bool & config_modified)
{
    IntrConfiguration<Real> box0CfgStart;
    IntrConfigManager<Box>::init(axis,box0,box0CfgStart);

    IntrConfiguration<Real> box1CfgStart;
    IntrConfigManager<Box>::init(axis,box1,box1CfgStart);

    return IntrConfigManager<Real>::Find(box0CfgStart, box1CfgStart, side,
        box0CfgFinal, box1CfgFinal, dmax,dfirst, config_modified);
}
//----------------------------------------------------------------------------
template <class TDataTypes>
bool IntrAxis<TOBB<TDataTypes> >::Find(const Coord& axis,
    const defaulttype::Vec<3,Real> segment[2],Real radius, const Box& box,
    Real dmax, Real& dfirst,
    int& side, CapIntrConfiguration<Real>& capCfgFinal,
    IntrConfiguration<Real>& boxCfgFinal,bool & config_modified)
{
    CapIntrConfiguration<Real> capCfgStart;
    IntrConfigManager<Real>::init(axis,segment,radius,capCfgStart);

    IntrConfiguration<Real> boxCfgStart;
    IntrConfigManager<Box>::init(axis,box,boxCfgStart);

    return IntrConfigManager<Real>::Find(capCfgStart, boxCfgStart, side,
        capCfgFinal, boxCfgFinal, dmax,dfirst, config_modified);
}
//----------------------------------------------------------------------------
template <typename Real>
void IntrConfigManager<Real>::init (const defaulttype::Vec<3,Real>& axis,
    const defaulttype::Vec<3,Real> segment[2], IntrConfiguration<Real>& cfg)
{
    Real dot[2] =
    {
        axis * segment[0],
        axis * segment[1]
    };

    if (fabs(dot[1] - dot[0]) < IntrUtil<Real>::ZERO_TOLERANCE())
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

template <typename Real>
void IntrConfigManager<Real>::init(const defaulttype::Vec<3,Real> & axis,
                                const defaulttype::Vec<3,Real> segment[2],Real radius,CapIntrConfiguration<Real> &cfg)
{
    cfg.axis = axis;

    Real dot[2] =
    {
        axis * segment[0],
        axis * segment[1]
    };

    if (fabs(dot[1] - dot[0]) < IntrUtil<Real>::ZERO_TOLERANCE())
    {
        cfg.mMap = IntrConfiguration<Real>::m2;
    }
    else
    {
        cfg.mMap = IntrConfiguration<Real>::m11;
    }

    if (dot[0] < dot[1])
    {
        cfg.mMin = dot[0] - radius;
        cfg.mMax = dot[1] + radius;
        cfg.mIndex[0] = 0;
        cfg.mIndex[1] = 1;
    }
    else
    {
        cfg.mMin = dot[1] - radius;
        cfg.mMax = dot[0] + radius;
        cfg.mIndex[0] = 1;
        cfg.mIndex[1] = 0;
    }
}
//----------------------------------------------------------------------------
template <class TDataTypes>
void IntrConfigManager<TOBB<TDataTypes> >::init (const defaulttype::Vec<3,Real> & axis,
    const Box & box, IntrConfiguration<Real>& cfg)
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
        helper::rabs(axes[0]),
        helper::rabs(axes[1]),
        helper::rabs(axes[2])
    };

    Real maxProjectedExtent;

    if (absAxes[0] < IntrUtil<Real>::ZERO_TOLERANCE())
    {
        if (absAxes[1] < IntrUtil<Real>::ZERO_TOLERANCE())
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
        else if (absAxes[2] < IntrUtil<Real>::ZERO_TOLERANCE())
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
    else if (absAxes[1] < IntrUtil<Real>::ZERO_TOLERANCE())
    {
        if (absAxes[2] < IntrUtil<Real>::ZERO_TOLERANCE())
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

    else if (absAxes[2] < IntrUtil<Real>::ZERO_TOLERANCE())
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
template <typename Real> template <class Config0,class Config1>
bool IntrConfigManager<Real>::Find (const Config0& cfg0Start,
    const Config1& cfg1Start, int& side,
    Config0& cfg0Final, Config1& cfg1Final,
    Real dmax,Real& dfirst,bool & config_modified)
{
    config_modified = false;
    // Constant velocity separating axis test.  The configurations cfg0Start
    // and cfg1Start are the current potential configurations for contact,
    // and cfg0Final and cfg1Final are improved configurations.
    Real d;

    if (cfg1Start.mMax < cfg0Start.mMin) // object1 left of object0
    {
        // find first time of contact on this axis
        d = (cfg0Start.mMin - cfg1Start.mMax);
        assert(d > 0);

        // If this is the new maximum first time of contact, set side and
        // configuration.
        if(d >= dmax)
            return false;

        if (d > dfirst)
        {
            dfirst = d;
            side = IntrConfiguration<Real>::LEFT;
            cfg0Final = cfg0Start;
            cfg1Final = cfg1Start;
            config_modified = true;
        }
    }
    else if (cfg0Start.mMax < cfg1Start.mMin)  // obj1 right of obj0
    {
        // find first time of contact on this axis
        d = (cfg1Start.mMin - cfg0Start.mMax);
        assert(d > 0);

        // If this is the new maximum first time of contact,  set side and
        // configuration.
        if(d >= dmax)
            return false;

        if (d > dfirst)
        {
            dfirst = d;
            side = IntrConfiguration<Real>::RIGHT;
            cfg0Final = cfg0Start;
            cfg1Final = cfg1Start;
            config_modified = true;
        }
    }
    else // object1 and object0 on overlapping interval
    {
        Real midpoint0 = (cfg0Start.mMin + cfg0Start.mMax)/((Real)(2.0));
        Real midpoint1 = (cfg1Start.mMin + cfg1Start.mMax)/((Real)(2.0));
        if (midpoint1 < midpoint0)
        {
            // find first time of contact on this axis
            d = (cfg0Start.mMin - cfg1Start.mMax);
            assert(d <= 0);

            // If this is the new maximum first time of contact, set side and
            // configuration.
            if (/*-d < dmax && */d > dfirst)
            {
                dfirst = d;
                side = IntrConfiguration<Real>::LEFT;
                cfg0Final = cfg0Start;
                cfg1Final = cfg1Start;
                config_modified = true;
            }
        }
        else// if (cfg1Start.mMin < cfg0Start.mMax)
        {
            // find first time of contact on this axis
            d = (cfg1Start.mMin - cfg0Start.mMax);
            assert(d < 0);

            // If this is the new maximum first time of contact,  set side and
            // configuration.
            if (/*-d < dmax &&*/ d > dfirst)
            {
                dfirst = d;
                side = IntrConfiguration<Real>::RIGHT;
                cfg0Final = cfg0Start;
                cfg1Final = cfg1Start;
                config_modified = true;
            }
        }
    }

    return true;
}
//----------------------------------------------------------------------------
template <class TDataTypes>
void FindContactSet<TOBB<TDataTypes> >::FindContactConfig(const defaulttype::Vec<3,Real> & axis,const defaulttype::Vec<3,Real> & segP0, Real radius,const Box & box,CapIntrConfiguration<Real> &capCfg,
    int side,defaulttype::Vec<3, Real> & pt_on_capsule, defaulttype::Vec<3, Real> &pt_on_box){
    bool adjust = false;
    pt_on_box = box.center();

    Real coord_i;
    defaulttype::Vec<3,Real> centered_seg = segP0 - box.center();

    for(int i = 0 ; i < 3 ; ++i){
        coord_i = box.axis(i) * (centered_seg);

        if(coord_i < -box.extent(i) - IntrUtil<Real>::ZERO_TOLERANCE()){
            adjust = true;
            coord_i = -box.extent(i);
        }
        else if(coord_i > box.extent(i) + IntrUtil<Real>::ZERO_TOLERANCE()){
            coord_i = box.extent(i);
            adjust = true;
        }

        pt_on_box += coord_i * box.axis(i);
    }

    if(adjust){
        defaulttype::Vec<3,Real> segPP0n(pt_on_box - segP0);
        if((segPP0n.cross(axis)).norm2() < IntrUtil<Real>::ZERO_TOLERANCE()){
            if(side == IntrConfiguration<Real>::LEFT)
                capCfg.axis *= -1.0;
        }
        else{
            IntrUtil<Real>::normalize(segPP0n);
            capCfg.axis = segPP0n;
        }

        pt_on_capsule = segP0 + radius * capCfg.axis;
    }
    else{
        pt_on_capsule = pt_on_box;

        if(side == IntrConfiguration<Real>::LEFT)
            capCfg.axis *= -1.0;
    }

    capCfg.have_naxis = true;
}


template <class TDataTypes>
FindContactSet<TOBB<TDataTypes> >::FindContactSet (const defaulttype::Vec<3,Real> segment[2], Real radius,const Box& box,const defaulttype::Vec<3,Real> & axis,
    int side, CapIntrConfiguration<Real> &capCfg,
    const IntrConfiguration<Real>& boxCfg,
    Real tfirst, defaulttype::Vec<3,Real> & pt_on_capsule,defaulttype::Vec<3,Real> & pt_on_box){
    int quantity;

    const int* bIndex = boxCfg.mIndex;
    const int* capIndex = capCfg.mIndex;

    if (side == IntrConfiguration<Real>::LEFT)
    {
        // Move the segment to its new position.
        defaulttype::Vec<3,Real> segFinal[2] =
        {
            segment[0] - tfirst*axis,
            segment[1] - tfirst*axis
        };

        // box on left of seg
        if (capCfg.mMap == IntrConfiguration<Real>::m11)
        {
            FindContactConfig(axis,segment[capIndex[0]],radius,box,capCfg,side,pt_on_capsule,pt_on_box);
        }
        else if (boxCfg.mMap == IntrConfiguration<Real>::m1_1)
        {
            pt_on_box = getPointFromIndex(bIndex[7], box);
            IntrUtil<Real>::projectPointOnCapsuleAndFindCapNormal(pt_on_box,segment,radius,capCfg,pt_on_capsule);
        }
        else if (boxCfg.mMap == IntrConfiguration<Real>::m2_2)
        {
            // segment-segment intersection
            defaulttype::Vec<3,Real> boxSeg[2],cap_pt;
            boxSeg[0] = getPointFromIndex(bIndex[6], box);
            boxSeg[1] = getPointFromIndex(bIndex[7], box);

            IntrUtil<Real>::segNearestPoints(segment,boxSeg,cap_pt,pt_on_box);

            capCfg.axis = pt_on_box - cap_pt;
            IntrUtil<Real>::normalize(capCfg.axis);
            capCfg.have_naxis = true;

            pt_on_capsule = cap_pt + capCfg.axis * radius;
        }
        else // boxCfg.mMap == IntrConfiguration<Real>::m44
        {
            // segment-boxface intersection
            defaulttype::Vec<3,Real> boxFace[4];
            boxFace[0] = getPointFromIndex(bIndex[4], box);
            boxFace[1] = getPointFromIndex(bIndex[5], box);
            boxFace[2] = getPointFromIndex(bIndex[6], box);
            boxFace[3] = getPointFromIndex(bIndex[7], box);

            defaulttype::Vec<3,Real> capSeg[2];
            capCfg.leftSegment(segFinal,radius,capSeg);

            defaulttype::Vec<3,Real> P[2];
            IntrUtil<Real>::CoplanarSegmentRectangle(capSeg, boxFace, quantity,P);

            if(quantity != 0){
                IntrUtil<Real>::projectIntPoints(axis,tfirst,P,quantity,pt_on_capsule);
                pt_on_box = pt_on_capsule - axis * tfirst;
                capCfg.axis *= -1;
                capCfg.have_naxis = true;
            }
            else{
                IntrUtil<Real>::faceSegNearestPoints(boxFace,capSeg,pt_on_box,pt_on_capsule);
                IntrUtil<Real>::projectPointOnCapsuleAndFindCapNormal(pt_on_box,segment,radius,capCfg,pt_on_capsule);
            }
        }
    }
    else // side == RIGHT
    {
        // Move the segment to its new position.
        defaulttype::Vec<3,Real> segFinal[2] =
        {
            segment[0] + tfirst*axis,
            segment[1] + tfirst*axis
        };

        // box on right of seg
        if (capCfg.mMap == IntrConfiguration<Real>::m11)
        {
            FindContactConfig(axis,segment[capIndex[1]],radius,box,capCfg,side,pt_on_capsule,pt_on_box);
        }
        else if (boxCfg.mMap == IntrConfiguration<Real>::m1_1)
        {

            pt_on_box = getPointFromIndex(bIndex[0], box);
            IntrUtil<Real>::projectPointOnCapsuleAndFindCapNormal(pt_on_box,segment,radius,capCfg,pt_on_capsule);
        }
        else if (boxCfg.mMap == IntrConfiguration<Real>::m2_2)
        {
            // segment-segment intersection
            defaulttype::Vec<3,Real> boxSeg[2],cap_pt;
            boxSeg[0] = getPointFromIndex(bIndex[0], box);
            boxSeg[1] = getPointFromIndex(bIndex[1], box);

            IntrUtil<Real>::segNearestPoints(segment,boxSeg,cap_pt,pt_on_box);

            capCfg.axis = pt_on_box - cap_pt;
            IntrUtil<Real>::normalize(capCfg.axis);
            capCfg.have_naxis = true;

            pt_on_capsule = cap_pt + capCfg.axis * radius;
        }
        else // boxCfg.mMap == IntrConfiguration<Real>::m44
        {
            // segment-boxface intersection
            defaulttype::Vec<3,Real> boxFace[4];
            boxFace[0] = getPointFromIndex(bIndex[0], box);
            boxFace[1] = getPointFromIndex(bIndex[1], box);
            boxFace[2] = getPointFromIndex(bIndex[2], box);
            boxFace[3] = getPointFromIndex(bIndex[3], box);

            defaulttype::Vec<3,Real> capSeg[2];
            capCfg.rightSegment(segFinal,radius,capSeg);

            defaulttype::Vec<3,Real> P[2];
            IntrUtil<Real>::CoplanarSegmentRectangle(capSeg, boxFace, quantity,P);


            if(quantity != 0){
                IntrUtil<Real>::projectIntPoints(-axis,tfirst,P,quantity,pt_on_capsule);
                pt_on_box = pt_on_capsule + axis * tfirst;
                capCfg.have_naxis = true;
            }
            else{
                IntrUtil<Real>::faceSegNearestPoints(boxFace,capSeg,pt_on_box,pt_on_capsule);
                IntrUtil<Real>::projectPointOnCapsuleAndFindCapNormal(pt_on_box,segment,radius,capCfg,pt_on_capsule);
            }
        }
    }
}

template <typename Real>
defaulttype::Vec<3,Real> IntrUtil<Real>::nearestPointOnSeg(const defaulttype::Vec<3,Real> & seg0,const defaulttype::Vec<3,Real> & seg1,const defaulttype::Vec<3,Real> & point){
    const defaulttype::Vec<3,Real> AB = seg1-seg0;
    const defaulttype::Vec<3,Real> AQ = point -seg0;
    Real A;
    Real b;
    A = AB*AB;
    b = AQ*AB;

    Real alpha = b/A;
    if (alpha <= 0.0){
        return seg0;
    }
    else if (alpha >= 1.0){
        return seg1;
    }
    else{
        return seg0 + AB * alpha;
    }
}


template <typename Real>
void IntrUtil<Real>::segNearestPoints(const defaulttype::Vec<3,Real> & p0,const defaulttype::Vec<3,Real> & p1, const defaulttype::Vec<3,Real> & q0,const defaulttype::Vec<3,Real> & q1,defaulttype::Vec<3,Real> & P,defaulttype::Vec<3,Real> & Q,
                                      SReal & alpha,SReal & beta){
    const defaulttype::Vec<3,Real> AB = p1-p0;
    const defaulttype::Vec<3,Real> CD = q1-q0;
    const defaulttype::Vec<3,Real> AC = q0-p0;

    defaulttype::Matrix2 Amat;//matrix helping us to find the two nearest points lying on the segments of the two segments
    defaulttype::Vector2 b;

    Amat[0][0] = AB*AB;
    Amat[1][1] = CD*CD;
    Amat[0][1] = Amat[1][0] = -CD*AB;
    b[0] = AB*AC;
    b[1] = -CD*AC;
    const SReal det = determinant(Amat);

    SReal AB_norm2 = AB.norm2();
    SReal CD_norm2 = CD.norm2();
    alpha = 0.5;
    beta = 0.5;
    //Check that the determinant is not null which would mean that the segment segments are lying on a same plane.
    //in this case we can solve the little system which gives us
    //the two coefficients alpha and beta. We obtain the two nearest points P and Q lying on the segments of the two segments.
    //P = A + AB * alpha;
    //Q = C + CD * beta;
    if (det < -IntrUtil<Real>::ZERO_TOLERANCE() || det > IntrUtil<Real>::ZERO_TOLERANCE())
    {
        alpha = (b[0]*Amat[1][1] - b[1]*Amat[0][1])/det;
        beta  = (b[1]*Amat[0][0] - b[0]*Amat[1][0])/det;
    }
    else{//segment segments on a same plane. Here the idea to find the nearest points
        //is to project segment apexes on the other segment.
        //Visual example with semgents AB and CD :
        //            A----------------B
        //                     C----------------D
        //After projection :
        //            A--------c-------B
        //                     C-------b--------D
        //So the nearest points are p and q which are respecively in the middle of cB and Cb:
        //            A--------c---p---B
        //                     C---q---b--------D
        defaulttype::Vec<3,Real> AD = q1 - p0;
        defaulttype::Vec<3,Real> CB = p1 - q0;

        SReal c_proj= b[0]/AB_norm2;//alpha = (AB * AC)/AB_norm2
        SReal d_proj = (AB * AD)/AB_norm2;
        SReal a_proj = b[1]/CD_norm2;//beta = (-CD*AC)/CD_norm2
        SReal b_proj= (CD*CB)/CD_norm2;

        if(c_proj >= 0 && c_proj <= 1){//projection of C on AB is lying on AB
            if(d_proj > 1){//case :
                           //             A----------------B
                           //                      C---------------D
                alpha = (1.0 + c_proj)/2.0;
                beta = b_proj/2.0;
            }
            else if(d_proj < 0){//case :
                                //             A----------------B
                                //     D----------------C
                alpha = c_proj/2.0;
                beta = (1 + a_proj)/2.0;
            }
            else{//case :
                //             A----------------B
                //                 C------D
                alpha = (c_proj + d_proj)/2.0;
                beta  = 0.5;
            }
        }
        else if(d_proj >= 0 && d_proj <= 1){
            if(c_proj < 0){//case :
                           //             A----------------B
                           //     C----------------D
                alpha = d_proj /2.0;
                beta = (1 + a_proj)/2.0;
            }
            else{//case :
                 //          A---------------B
                 //                 D-------------C
                alpha = (1 + d_proj)/2.0;
                beta = b_proj/2.0;
            }
        }
        else{
            if(c_proj * d_proj < 0){//case :
                                    //           A--------B
                                    //       D-----------------C
                alpha = 0.5;
                beta = (a_proj + b_proj)/2.0;
            }
            else{
                if(c_proj < 0){//case :
                               //                    A---------------B
                               // C-------------D
                    alpha = 0;
                }
                else{
                    alpha = 1;
                }

                if(a_proj < 0){//case :
                               // A---------------B
                               //                     C-------------D
                    beta = 0;
                }
                else{//case :
                     //                     A---------------B
                     //   C-------------D
                    beta = 1;
                }
            }
        }

        P = p0 + AB * alpha;
        Q = q0 + CD * beta;

        return;
    }

    if(alpha < 0){
        alpha = 0;
        beta = (CD * (p0 - q0))/CD_norm2;
    }
    else if(alpha > 1){
        alpha = 1;
        beta = (CD * (p1 - q0))/CD_norm2;
    }

    if(beta < 0){
        beta = 0;
        alpha = (AB * (q0 - p0))/AB_norm2;
    }
    else if(beta > 1){
        beta = 1;
        alpha = (AB * (q1 - p0))/AB_norm2;
    }

    if(alpha < 0)
        alpha = 0;
    else if (alpha > 1)
        alpha = 1;

    assert(alpha >= 0);
    assert(alpha <= 1);
    assert(beta >= 0);
    assert(beta <= 1);

    P = p0 + AB * alpha;
    Q = q0 + CD * beta;
}


template <typename Real>
void IntrUtil<Real>::segNearestPoints(const defaulttype::Vec<3,Real> & p0,const defaulttype::Vec<3,Real> & p1, const defaulttype::Vec<3,Real> & q0,const defaulttype::Vec<3,Real> & q1,defaulttype::Vec<3,Real> & P,defaulttype::Vec<3,Real> & Q){
    SReal alpha,beta;
    segNearestPoints(p0,p1,q0,q1,P,Q,alpha,beta);
}



template <typename Real>
void IntrUtil<Real>::segNearestPoints(const defaulttype::Vec<3,Real> * p, const defaulttype::Vec<3,Real> * q,defaulttype::Vec<3,Real> & P,defaulttype::Vec<3,Real> & Q)
{
    segNearestPoints(p[0],p[1],q[0],q[1],P,Q);
}

//----------------------------------------------------------------------------
template <class TDataTypes>
FindContactSet<TOBB<TDataTypes> >::FindContactSet (const Box& box0,
    const Box& box1,const defaulttype::Vec<3,Real> & axis,int side, const IntrConfiguration<Real>& box0Cfg,
    const IntrConfiguration<Real>& box1Cfg,
    Real tfirst,defaulttype::Vec<3,Real> & pt_on_first,defaulttype::Vec<3,Real> & pt_on_second)
{
    int quantity;
    defaulttype::Vec<3,Real> P[8];

    const int* b0Index = box0Cfg.mIndex;
    const int* b1Index = box1Cfg.mIndex;

    if (side == IntrConfiguration<Real>::LEFT)
    {
        // box1 on left of box0
        if (box0Cfg.mMap == IntrConfiguration<Real>::m1_1)
        {
            pt_on_first = getPointFromIndex(b0Index[0], box0);
            pt_on_second = pt_on_first - axis * tfirst;
            //IntrUtil<Box>::project(pt_on_second,box1);
        }
        else if (box1Cfg.mMap == IntrConfiguration<Real>::m1_1)
        {
            pt_on_second = getPointFromIndex(b1Index[7], box1);
            pt_on_first = pt_on_second + axis * tfirst;
            //IntrUtil<Box>::project(pt_on_first,box0);
        }
        else if (box0Cfg.mMap == IntrConfiguration<Real>::m2_2)
        {
            if (box1Cfg.mMap == IntrConfiguration<Real>::m2_2)
            {
                // box0edge-box1edge intersection
                defaulttype::Vec<3,Real> edge0[2], edge1[2];
                edge0[0] = getPointFromIndex(b0Index[0], box0);
                edge0[1] = getPointFromIndex(b0Index[1], box0);
                edge1[0] = getPointFromIndex(b1Index[6], box1);
                edge1[1] = getPointFromIndex(b1Index[7], box1);

                IntrUtil<Real>::segNearestPoints(edge0,edge1,pt_on_first,pt_on_second);
            }
            else // box1Cfg.mMap == IntrConfiguration<Real>::m44
            {
                MyBox<Real> box1Final;
                box1Final.Center = box1.center() + tfirst*axis;
                for (int i = 0; i < 3; ++i)
                {
                    box1Final.Extent[i] = box1.extent(i);
                    box1Final.Axis[i] = box1.axis(i);
                }

                // box0edge-box1face intersection
                defaulttype::Vec<3,Real> edge0[2], face1[4];
                edge0[0] = getPointFromIndex(b0Index[0], box0);
                edge0[1] = getPointFromIndex(b0Index[1], box0);
                face1[0] = GetPointFromIndex(b1Index[4], box1Final);
                face1[1] = GetPointFromIndex(b1Index[5], box1Final);
                face1[2] = GetPointFromIndex(b1Index[6], box1Final);
                face1[3] = GetPointFromIndex(b1Index[7], box1Final);

                IntrUtil<Real>::CoplanarSegmentRectangle(edge0, face1, quantity, P);
                if(quantity != 0){
                    IntrUtil<Real>::projectIntPoints(-axis,tfirst,P,quantity,pt_on_second);
                    pt_on_first = pt_on_second + axis * tfirst;
                }
                else{
                    IntrUtil<Real>::faceSegNearestPoints(face1,edge0,pt_on_second,pt_on_first);
                    pt_on_second -= tfirst * axis;
                }
            }
        }
        else // box0Cfg.mMap == IntrConfiguration<Real>::m44
        {
            if (box1Cfg.mMap == IntrConfiguration<Real>::m2_2)
            {
                MyBox<Real> box1Final;
                box1Final.Center = box1.center() + tfirst*axis;
                for (int i = 0; i < 3; ++i)
                {
                    box1Final.Extent[i] = box1.extent(i);
                    box1Final.Axis[i] = box1.axis(i);
                }

                // box0face-box1edge intersection
                defaulttype::Vec<3,Real> face0[4], edge1[2];
                face0[0] = getPointFromIndex(b0Index[0], box0);
                face0[1] = getPointFromIndex(b0Index[1], box0);
                face0[2] = getPointFromIndex(b0Index[2], box0);
                face0[3] = getPointFromIndex(b0Index[3], box0);
                edge1[0] = GetPointFromIndex(b1Index[6], box1Final);
                edge1[1] = GetPointFromIndex(b1Index[7], box1Final);

                IntrUtil<Real>::CoplanarSegmentRectangle(edge1, face0, quantity, P);
                if(quantity != 0){
                    IntrUtil<Real>::projectIntPoints(-axis,tfirst,P,quantity,pt_on_second);
                    pt_on_first = pt_on_second + axis * tfirst;
                }
                else{
                    IntrUtil<Real>::faceSegNearestPoints(face0,edge1,pt_on_first,pt_on_second);
                    pt_on_second -= tfirst * axis;
                }

            }
            else
            {
                MyBox<Real> box1Final;
                box1Final.Center = box1.center() + tfirst*axis;
                for (int i = 0; i < 3; ++i)
                {
                    box1Final.Extent[i] = box1.extent(i);
                    box1Final.Axis[i] = box1.axis(i);
                }

                // box0face-box1face intersection
                defaulttype::Vec<3,Real> face0[4], face1[4];
                face0[0] = getPointFromIndex(b0Index[0], box0);
                face0[1] = getPointFromIndex(b0Index[1], box0);
                face0[2] = getPointFromIndex(b0Index[2], box0);
                face0[3] = getPointFromIndex(b0Index[3], box0);
                face1[0] = GetPointFromIndex(b1Index[4], box1Final);
                face1[1] = GetPointFromIndex(b1Index[5], box1Final);
                face1[2] = GetPointFromIndex(b1Index[6], box1Final);
                face1[3] = GetPointFromIndex(b1Index[7], box1Final);

                IntrUtil<Real>::CoplanarRectangleRectangle(face0, face1, quantity, P);
                if(quantity == 0){
                    IntrUtil<Real>::facesNearestPoints(face0,4,face1,4,pt_on_first,pt_on_second);
                    pt_on_second -= tfirst * axis;
                }
                else{
                    IntrUtil<Real>::projectIntPoints(-axis,tfirst,P,quantity,pt_on_second);
                    pt_on_first = pt_on_second + axis * tfirst;
                }
            }
        }
    }
    else // side == RIGHT
    {
        // box1 on right of box0
        if (box0Cfg.mMap == IntrConfiguration<Real>::m1_1)
        {
            pt_on_first = getPointFromIndex(b0Index[7], box0);
            pt_on_second = pt_on_first + tfirst * axis;
        }
        else if (box1Cfg.mMap == IntrConfiguration<Real>::m1_1)
        {
            pt_on_second = getPointFromIndex(b1Index[0], box1);
            pt_on_first = pt_on_second - tfirst * axis;
        }
        else if (box0Cfg.mMap == IntrConfiguration<Real>::m2_2)
        {
            if (box1Cfg.mMap == IntrConfiguration<Real>::m2_2)
            {
                // box0edge-box1edge intersection
                defaulttype::Vec<3,Real> edge0[2], edge1[2];
                edge0[0] = getPointFromIndex(b0Index[6], box0);
                edge0[1] = getPointFromIndex(b0Index[7], box0);
                edge1[0] = getPointFromIndex(b1Index[0], box1);
                edge1[1] = getPointFromIndex(b1Index[1], box1);

                IntrUtil<Real>::segNearestPoints(edge0,edge1,pt_on_first,pt_on_second);
            }
            else // box1Cfg.mMap == IntrConfiguration<Real>::m44
            {
                MyBox<Real> box1Final;
                box1Final.Center = box1.center() - tfirst*axis;
                for (int i = 0; i < 3; ++i)
                {
                    box1Final.Extent[i] = box1.extent(i);
                    box1Final.Axis[i] = box1.axis(i);
                }

                // box0edge-box1face intersection
                defaulttype::Vec<3,Real> edge0[2], face1[4];
                edge0[0] = getPointFromIndex(b0Index[6], box0);
                edge0[1] = getPointFromIndex(b0Index[7], box0);
                face1[0] = GetPointFromIndex(b1Index[0], box1Final);
                face1[1] = GetPointFromIndex(b1Index[1], box1Final);
                face1[2] = GetPointFromIndex(b1Index[2], box1Final);
                face1[3] = GetPointFromIndex(b1Index[3], box1Final);

                IntrUtil<Real>::CoplanarSegmentRectangle(edge0, face1, quantity, P);
                if(quantity != 0){
                    IntrUtil<Real>::projectIntPoints(axis,tfirst,P,quantity,pt_on_second);
                    pt_on_first = pt_on_second - axis * tfirst;
                }
                else{
                    IntrUtil<Real>::faceSegNearestPoints(face1,edge0,pt_on_second,pt_on_first);
                    pt_on_second += axis * tfirst;
                }
            }
        }
        else // box0Cfg.mMap == IntrConfiguration<Real>::m44
        {
            if (box1Cfg.mMap == IntrConfiguration<Real>::m2_2)
            {
                MyBox<Real> box1Final;
                box1Final.Center = box1.center() - tfirst*axis;
                for (int i = 0; i < 3; ++i)
                {
                    box1Final.Extent[i] = box1.extent(i);
                    box1Final.Axis[i] = box1.axis(i);
                }

                // box0face-box1edge intersection
                defaulttype::Vec<3,Real> face0[4], edge1[2];
                face0[0] = getPointFromIndex(b0Index[4], box0);
                face0[1] = getPointFromIndex(b0Index[5], box0);
                face0[2] = getPointFromIndex(b0Index[6], box0);
                face0[3] = getPointFromIndex(b0Index[7], box0);
                edge1[0] = GetPointFromIndex(b1Index[0], box1Final);
                edge1[1] = GetPointFromIndex(b1Index[1], box1Final);

                IntrUtil<Real>::CoplanarSegmentRectangle(edge1, face0, quantity, P);
                if(quantity != 0){
                    IntrUtil<Real>::projectIntPoints(axis,tfirst,P,quantity,pt_on_second);
                    pt_on_first = pt_on_second - axis * tfirst;
                }
                else{
                    IntrUtil<Real>::faceSegNearestPoints(face0,edge1,pt_on_first,pt_on_second);
                    pt_on_second += axis * tfirst;
                }
            }
            else // box1Cfg.mMap == IntrConfiguration<Real>::m44
            {
                MyBox<Real> box1Final;
                box1Final.Center = box1.center() - tfirst*axis;
                for (int i = 0; i < 3; ++i)
                {
                    box1Final.Extent[i] = box1.extent(i);
                    box1Final.Axis[i] = box1.axis(i);
                }

                // box0face-box1face intersection
                defaulttype::Vec<3,Real> face0[4], face1[4];
                face0[0] = getPointFromIndex(b0Index[4], box0);
                face0[1] = getPointFromIndex(b0Index[5], box0);
                face0[2] = getPointFromIndex(b0Index[6], box0);
                face0[3] = getPointFromIndex(b0Index[7], box0);
                face1[0] = GetPointFromIndex(b1Index[0], box1Final);
                face1[1] = GetPointFromIndex(b1Index[1], box1Final);
                face1[2] = GetPointFromIndex(b1Index[2], box1Final);
                face1[3] = GetPointFromIndex(b1Index[3], box1Final);

                IntrUtil<Real>::CoplanarRectangleRectangle(face0, face1, quantity, P);

                if(quantity == 0){
                    IntrUtil<Real>::facesNearestPoints(face0,4,face1,4,pt_on_first,pt_on_second);
                    pt_on_second += tfirst * axis;
                }
                else{
                    IntrUtil<Real>::projectIntPoints(axis,tfirst,P,quantity,pt_on_second);
                    pt_on_first = pt_on_second - axis * tfirst;
                }
            }
        }
    }
}
//----------------------------------------------------------------------------
template <typename Real>
void IntrUtil<Real>::ColinearSegments (const defaulttype::Vec<3,Real> segment0[2],
    const defaulttype::Vec<3,Real> segment1[2], int& quantity, defaulttype::Vec<3,Real>* P)
{
    // The potential intersection is initialized to segment0 and clipped
    // against segment1.
    quantity = 2;
    for (int i = 0; i < 2; ++i)
    {
        P[i] = segment0[i];
    }

    // point 0
    defaulttype::Vec<3,Real> V = segment1[1] - segment1[0];
    Real c = V * segment1[0];
    ClipConvexPolygonAgainstPlane(V, c, quantity, P);

    // point 1
    V = -V;
    c = V * segment1[1];
    ClipConvexPolygonAgainstPlane(V, c, quantity, P);
}
//----------------------------------------------------------------------------
template <typename Real>
void IntrUtil<Real>::SegmentThroughPlane (
    const defaulttype::Vec<3,Real> segment[2], const defaulttype::Vec<3,Real>& planeOrigin,
    const defaulttype::Vec<3,Real>& planeNormal, int& quantity, defaulttype::Vec<3,Real>* P)
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
template <typename Real>
void IntrUtil<Real>::SegmentSegment (const defaulttype::Vec<3,Real> segment0[2],
    const defaulttype::Vec<3,Real> segment1[2], int& quantity, defaulttype::Vec<3,Real>* P)
{
    defaulttype::Vec<3,Real> dir0 = segment0[1] - segment0[0];
    defaulttype::Vec<3,Real> dir1 = segment1[1] - segment1[0];
    defaulttype::Vec<3,Real> normal = dir0.cross(dir1);

    // The comparison is sin(kDir0,kDir1) < epsilon.
    Real sqrLen0 = dir0.norm2();
    Real sqrLen1 = dir1.norm2();
    Real sqrLenN = normal.norm2();
    if (sqrLenN < IntrUtil<Real>::ZERO_TOLERANCE()*sqrLen0*sqrLen1)
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
template <typename Real>
void IntrUtil<Real>::ColinearSegmentTriangle (
    const defaulttype::Vec<3,Real> segment[2], const defaulttype::Vec<3,Real> triangle[3],
    int& quantity, defaulttype::Vec<3,Real>* P)
{
    // The potential intersection is initialized to the line segment and then
    // clipped against the three sides of the tri
    quantity = 2;
    int i;
    for (i = 0; i < 2; ++i)
    {
        P[i] = segment[i];
    }

    defaulttype::Vec<3,Real> side[3] =
    {
        triangle[1] - triangle[0],
        triangle[2] - triangle[1],
        triangle[0] - triangle[2]
    };

    defaulttype::Vec<3,Real> normal = side[0].cross(side[1]);
    for (i = 0; i < 3; ++i)
    {
        // Normal pointing inside the triangle.
        defaulttype::Vec<3,Real> sideN = normal.cross(side[i]);
        Real constant = sideN * triangle[i];
        ClipConvexPolygonAgainstPlane(sideN, constant, quantity, P);
    }
}
//----------------------------------------------------------------------------
template <typename Real>
void IntrUtil<Real>::CoplanarSegmentRectangle (
    const defaulttype::Vec<3,Real> segment[2], const defaulttype::Vec<3,Real> rectangle[4],
    int& quantity, defaulttype::Vec<3,Real>* P)
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
        defaulttype::Vec<3,Real> normal = rectangle[i1] - rectangle[i0];
        Real constant = normal * rectangle[i0];
        ClipConvexPolygonAgainstPlane(normal, constant, quantity, P);
    }
}
//----------------------------------------------------------------------------
template <typename Real>
void IntrUtil<Real>::CoplanarTriangleRectangle (
    const defaulttype::Vec<3,Real> triangle[3], const defaulttype::Vec<3,Real> rectangle[4],
    int& quantity, defaulttype::Vec<3,Real>* P)
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
        defaulttype::Vec<3,Real> normal = rectangle[i1] - rectangle[i0];
        Real constant = normal * rectangle[i0];
        ClipConvexPolygonAgainstPlane(normal, constant, quantity, P);
    }
}
//----------------------------------------------------------------------------
template <typename Real>
void IntrUtil<Real>::CoplanarRectangleRectangle (
    const defaulttype::Vec<3,Real> rectangle0[4], const defaulttype::Vec<3,Real> rectangle1[4],
    int& quantity, defaulttype::Vec<3,Real>* P)
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
        defaulttype::Vec<3,Real> normal = rectangle1[i1] - rectangle1[i0];
        Real constant = normal * rectangle1[i0];
        ClipConvexPolygonAgainstPlane(normal, constant, quantity, P);
    }
}
//----------------------------------------------------------------------------
template <typename Real>
void IntrUtil<Real>::projectIntPoints(const defaulttype::Vec<3, Real> &velocity, Real contactTime, const defaulttype::Vec<3, Real> *points, int n, defaulttype::Vec<3, Real> &proj_pt){
    proj_pt.set(0,0,0);
    defaulttype::Vec<3,Real> v0 = velocity * contactTime;
    for(int i = 0 ; i < n ; ++i){
        proj_pt += points[i] + v0;
    }

    proj_pt /= n;
}
//----------------------------------------------------------------------------
template <typename Real>
void IntrUtil<Real>::projectPointOnCapsuleAndFindCapNormal(const defaulttype::Vec<3,Real> & pt,const defaulttype::Vec<3,Real> segment[2],Real radius,CapIntrConfiguration<Real> & capCfg,defaulttype::Vec<3,Real> & pt_on_capsule){
    defaulttype::Vec<3,Real> dir(segment[1] - segment[0]);
    Real alpha = dir * (pt - segment[0]) / dir.norm2();

    if(alpha < 0)
        alpha = 0;
    else if(alpha > 1)
        alpha = 1;

    defaulttype::Vec<3,Real> segP = segment[0] + alpha * dir;
    defaulttype::Vec<3,Real> segPpt = pt - segP;

    IntrUtil<Real>::normalize(segPpt);

    capCfg.axis = segPpt;
    capCfg.have_naxis = true;

    pt_on_capsule = segP + radius*segPpt;
}

template <typename Real>
Real IntrUtil<Real>::projectOnTriangle(defaulttype::Vec<3,Real> & pt,const defaulttype::Vec<3,Real> & t_p0,const defaulttype::Vec<3,Real> & t_p1,const defaulttype::Vec<3,Real> & t_p2,Real & s,Real & t){
    defaulttype::Vec<3,Real> diff = t_p0 - pt;
    defaulttype::Vec<3,Real> edge0 = t_p1 - t_p0;
    defaulttype::Vec<3,Real> edge1 = t_p2 - t_p0;
    Real a00 = edge0.norm2();
    Real a01 = edge0 *edge1;
    Real a11 = edge1.norm2();
    Real b0 = diff * edge0;
    Real b1 = diff * edge1;
    Real c = diff.norm2();
    Real det = fabs(a00*a11 - a01*a01);
    s = a01*b1 - a11*b0;
    t = a01*b0 - a00*b1;
    Real sqrDistance;

    if (s + t <= det)
    {
        if (s < (Real)0)
        {
            if (t < (Real)0)  // region 4
            {
                if (b0 < (Real)0)
                {
                    t = (Real)0;
                    if (-b0 >= a00)
                    {
                        s = (Real)1;
                        sqrDistance = a00 + ((Real)2)*b0 + c;
                    }
                    else
                    {
                        s = -b0/a00;
                        sqrDistance = b0*s + c;
                    }
                }
                else
                {
                    s = (Real)0;
                    if (b1 >= (Real)0)
                    {
                        t = (Real)0;
                        sqrDistance = c;
                    }
                    else if (-b1 >= a11)
                    {
                        t = (Real)1;
                        sqrDistance = a11 + ((Real)2)*b1 + c;
                    }
                    else
                    {
                        t = -b1/a11;
                        sqrDistance = b1*t + c;
                    }
                }
            }
            else  // region 3
            {
                s = (Real)0;
                if (b1 >= (Real)0)
                {
                    t = (Real)0;
                    sqrDistance = c;
                }
                else if (-b1 >= a11)
                {
                    t = (Real)1;
                    sqrDistance = a11 + ((Real)2)*b1 + c;
                }
                else
                {
                    t = -b1/a11;
                    sqrDistance = b1*t + c;
                }
            }
        }
        else if (t < (Real)0)  // region 5
        {
            t = (Real)0;
            if (b0 >= (Real)0)
            {
                s = (Real)0;
                sqrDistance = c;
            }
            else if (-b0 >= a00)
            {
                s = (Real)1;
                sqrDistance = a00 + ((Real)2)*b0 + c;
            }
            else
            {
                s = -b0/a00;
                sqrDistance = b0*s + c;
            }
        }
        else  // region 0
        {
            // minimum at interior point
            Real invDet = ((Real)1)/det;
            s *= invDet;
            t *= invDet;
            sqrDistance = s*(a00*s + a01*t + ((Real)2)*b0) +
                t*(a01*s + a11*t + ((Real)2)*b1) + c;
        }
    }
    else
    {
        Real tmp0, tmp1, numer, denom;

        if (s < (Real)0)  // region 2
        {
            tmp0 = a01 + b0;
            tmp1 = a11 + b1;
            if (tmp1 > tmp0)
            {
                numer = tmp1 - tmp0;
                denom = a00 - ((Real)2)*a01 + a11;
                if (numer >= denom)
                {
                    s = (Real)1;
                    t = (Real)0;
                    sqrDistance = a00 + ((Real)2)*b0 + c;
                }
                else
                {
                    s = numer/denom;
                    t = (Real)1 - s;
                    sqrDistance = s*(a00*s + a01*t + ((Real)2)*b0) +
                        t*(a01*s + a11*t + ((Real)2)*b1) + c;
                }
            }
            else
            {
                s = (Real)0;
                if (tmp1 <= (Real)0)
                {
                    t = (Real)1;
                    sqrDistance = a11 + ((Real)2)*b1 + c;
                }
                else if (b1 >= (Real)0)
                {
                    t = (Real)0;
                    sqrDistance = c;
                }
                else
                {
                    t = -b1/a11;
                    sqrDistance = b1*t + c;
                }
            }
        }
        else if (t < (Real)0)  // region 6
        {
            tmp0 = a01 + b1;
            tmp1 = a00 + b0;
            if (tmp1 > tmp0)
            {
                numer = tmp1 - tmp0;
                denom = a00 - ((Real)2)*a01 + a11;
                if (numer >= denom)
                {
                    t = (Real)1;
                    s = (Real)0;
                    sqrDistance = a11 + ((Real)2)*b1 + c;
                }
                else
                {
                    t = numer/denom;
                    s = (Real)1 - t;
                    sqrDistance = s*(a00*s + a01*t + ((Real)2)*b0) +
                        t*(a01*s + a11*t + ((Real)2)*b1) + c;
                }
            }
            else
            {
                t = (Real)0;
                if (tmp1 <= (Real)0)
                {
                    s = (Real)1;
                    sqrDistance = a00 + ((Real)2)*b0 + c;
                }
                else if (b0 >= (Real)0)
                {
                    s = (Real)0;
                    sqrDistance = c;
                }
                else
                {
                    s = -b0/a00;
                    sqrDistance = b0*s + c;
                }
            }
        }
        else  // region 1
        {
            numer = a11 + b1 - a01 - b0;
            if (numer <= (Real)0)
            {
                s = (Real)0;
                t = (Real)1;
                sqrDistance = a11 + ((Real)2)*b1 + c;
            }
            else
            {
                denom = a00 - ((Real)2)*a01 + a11;
                if (numer >= denom)
                {
                    s = (Real)1;
                    t = (Real)0;
                    sqrDistance = a00 + ((Real)2)*b0 + c;
                }
                else
                {
                    s = numer/denom;
                    t = (Real)1 - s;
                    sqrDistance = s*(a00*s + a01*t + ((Real)2)*b0) +
                        t*(a01*s + a11*t + ((Real)2)*b1) + c;
                }
            }
        }
    }

    // Account for numerical round-off error.
    if (sqrDistance < (Real)0)
    {
        sqrDistance = (Real)0;
    }

    pt = t_p0 + s*edge0 + t*edge1;
    return sqrDistance;
}


template <typename Real>
Real IntrUtil<Real>::facesNearestPoints(const defaulttype::Vec<3,Real> *first_face,int first_size,const defaulttype::Vec<3,Real> *second_face,int second_size, defaulttype::Vec<3,Real> &pt_on_first, defaulttype::Vec<3,Real> &pt_on_second){
    Real min = std::numeric_limits<Real>::max();
    Real new_min;

    defaulttype::Vec<3,Real> cur_pt_on_first,cur_pt_on_second;
    defaulttype::Vec<3,Real> * seg1;
    defaulttype::Vec<3,Real> * seg2;

    for(int i = 0 ; i < first_size ; ++i){
        if(i < first_size - 1){
            seg1 = const_cast<defaulttype::Vec<3,Real> *>(&first_face[i]);
        }
        else{
            seg1 = new defaulttype::Vec<3,Real>[2];
            seg1[0] = first_face[first_size - 1];
            seg1[1] = first_face[0];
        }

        for(int j = 0 ; j < second_size ; ++j){
            if(j < second_size - 1){
                seg2 = const_cast<defaulttype::Vec<3,Real> *>(&second_face[j]);
            }
            else{
                seg2 = new defaulttype::Vec<3,Real>[2];
                seg2[0] = second_face[second_size - 1];
                seg2[1] = second_face[0];
            }

            segNearestPoints(seg1,seg2,cur_pt_on_first,cur_pt_on_second);

            new_min = (cur_pt_on_first - cur_pt_on_second).norm2();
            if(min > new_min){
                min = new_min;
                pt_on_first = cur_pt_on_first;
                pt_on_second = cur_pt_on_second;
            }

            if(j == second_size - 1)
                delete[] seg2;
        }

        if(i == first_size - 1)
            delete[] seg1;
    }

    return min;
}

template <typename Real>
Real IntrUtil<Real>::faceSegNearestPoints(const defaulttype::Vec<3,Real> face[4],const defaulttype::Vec<3,Real> seg[2], defaulttype::Vec<3,Real> & pt_on_face,defaulttype::Vec<3,Real> & pt_on_seg){
    return faceSegNearestPoints(face,4,seg,pt_on_face,pt_on_seg);
}


template <typename Real>
Real IntrUtil<Real>::faceSegNearestPoints(const defaulttype::Vec<3,Real> * face,int n,const defaulttype::Vec<3,Real> seg[2], defaulttype::Vec<3,Real> & pt_on_face,defaulttype::Vec<3,Real> & pt_on_seg){
    Real min = std::numeric_limits<Real>::max();
    defaulttype::Vec<3,Real> cur_pt_on_face,cur_pt_on_seg;
    defaulttype::Vec<3,Real> face_seg[2];
    Real new_min;

    for(int j = 0 ; j < n ; ++j){
        face_seg[0] = face[j];
        if(j < n - 1){
            face_seg[1] = face[j + 1];
        }
        else{
            face_seg[1] = face[0];
        }

        segNearestPoints(face_seg,seg,cur_pt_on_face,cur_pt_on_seg);

        if((new_min = (cur_pt_on_face - cur_pt_on_seg).norm2()) < min){
            min = new_min;
            pt_on_face = cur_pt_on_face;
            pt_on_seg = cur_pt_on_seg;
        }
    }

    return min;
}

template <typename Real>
bool IntrUtil<Real>::nequal(Real a,Real b){
    return a < b - ZERO_TOLERANCE() || b < a - ZERO_TOLERANCE();
}

template <typename Real>
bool IntrUtil<Real>::strInf(Real a,Real b){
    return a < b - ZERO_TOLERANCE();
}

template <typename Real>
bool IntrUtil<Real>::inf(Real a,Real b){
    return a < b + ZERO_TOLERANCE();
}

template <class TDataTypes>
void IntrUtil<TOBB<TDataTypes> >::project(defaulttype::Vec<3,Real> & point,const Box & box){
    int min_ind = -1;
    bool neg = false;
    bool is_in = true;
    SReal diff = std::numeric_limits<SReal>::max();
    defaulttype::Vec<3,Real> centeredPt = point - box.center();
    point = box.center();

    Real coord_i;
    for(int i = 0 ; i < 3 ; ++i){
        coord_i = box.axis(i) * centeredPt;

        if(coord_i < -box.extent(i) - IntrUtil<Real>::ZERO_TOLERANCE()){
            is_in = false;
            coord_i = -box.extent(i);
        }
        else if(coord_i > box.extent(i) + IntrUtil<Real>::ZERO_TOLERANCE()){
            is_in = false;
            coord_i = box.extent(i);
        }
        else{
            SReal cur_diff = fabs(fabs(coord_i) - box.extent(i))/box.extent(i);
            if(cur_diff < diff){
                diff = cur_diff;
                min_ind = i;
                if(coord_i < 0.0)
                    neg = true;
            }
        }

        point += coord_i * box.axis(i);
    }

    if(is_in){
        if(neg){
            point[min_ind] = box.center()[min_ind] - box.extent(min_ind);
        }
        else{
            point[min_ind] = box.center()[min_ind] + box.extent(min_ind);
        }
    }
}

template <typename Real>
bool IntrUtil<Real>::equal(const defaulttype::Vec<3,Real> & vec0,const defaulttype::Vec<3,Real> & vec1){
    return (vec0 - vec1).norm2() < SQ_ZERO_TOLERANCE();
}

//----------------------------------------------------------------------------
template <class Real>
void ClipConvexPolygonAgainstPlane (const defaulttype::Vec<3,Real>& normal,
    Real constant, int& quantity, defaulttype::Vec<3,Real>* P)
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
            fabs(constant)*IntrUtil<Real>::ZERO_TOLERANCE();

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
                defaulttype::Vec<3,Real> CV[8];
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
                memcpy(P, CV, cQuantity*sizeof(defaulttype::Vec<3,Real>));
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
defaulttype::Vec<3,TReal> GetPointFromIndex (int index, const MyBox<TReal> &box)
{
    defaulttype::Vec<3,TReal> point = box.Center;

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
defaulttype::Vec<3,typename TDataTypes::Real> getPointFromIndex(int index, const TOBB<TDataTypes>& box)
{
    defaulttype::Vec<3,typename TDataTypes::Real> point = box.center();

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
    std::vector<defaulttype::Vec<3,TReal> > vs;
    defaulttype::Vec<3,TReal> a0(Axis[0] * Extent[0]);
    defaulttype::Vec<3,TReal> a1(Axis[1] * Extent[1]);
    defaulttype::Vec<3,TReal> a2(Axis[2] * Extent[2]);

    vs.push_back(Center - a0 + a1 - a2);
    vs.push_back(Center + a0 + a1 - a2);
    vs.push_back(Center + a0 + a1 + a2);
    vs.push_back(Center - a0 + a1 + a2);
    vs.push_back(Center - a0 - a1 - a2);
    vs.push_back(Center + a0 - a1 - a2);
    vs.push_back(Center + a0 - a1 + a2);
    vs.push_back(Center - a0 - a1 + a2);

    std::stringstream tmpmsg;
    for(int i = 0 ; i < 8 ; ++i){
        tmpmsg<<"    "<<vs[i]<<msgendl;
    dmsg_info("MyBox<TReal>") << tmpmsg ;
    }
}

}
}
}

#endif // INTRUTILITY3_INL
