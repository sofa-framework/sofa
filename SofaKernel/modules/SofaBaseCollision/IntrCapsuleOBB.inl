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
// Geometric Tools, LLC
// Copyright (c) 1998-2012
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
//
// File Version: 5.0.1 (2010/10/01)

#include <SofaBaseCollision/IntrUtility3.h>
#include <SofaBaseCollision/IntrCapsuleOBB.h>

namespace sofa{
namespace component{
namespace collision{


//----------------------------------------------------------------------------
template <typename TDataTypes,typename TDataTypes2>
TIntrCapsuleOBB<TDataTypes,TDataTypes2>::TIntrCapsuleOBB (const IntrCap& segment,
    const Box & box)
    :
    _cap(&segment),
    mBox(&box)
{
    _is_colliding = false;
}


template <typename TDataTypes,typename TDataTypes2>
bool TIntrCapsuleOBB<TDataTypes,TDataTypes2>::Find (Real dmax)
{
    bool config_modified;

    // Get the endpoints of the segment.
    const Vec3 segment[2] =
    {
        _cap->point1(),
        _cap->point2()
    };
    Real radius = _cap->radius();

    // Get the box velocity relative to the segment.

    mContactTime = -std::numeric_limits<Real>::max();

    int i;
    Vec3 axis;
    int side = IntrConfiguration<Real>::NONE;
    IntrConfiguration<Real> boxContact;
    CapIntrConfiguration<Real> capContact;

    // Test box normals.
    for (i = 0; i < 3; ++i)
    {
        axis = mBox->axis(i);
        if(!IntrAxis<Box>::Find(axis, segment,radius, *mBox, dmax,
            mContactTime, side, capContact, boxContact,config_modified))
            return false;


        if(config_modified){
            _sep_axis = axis;
        }
    }

    Vec3 cap_direction = _cap->point2() - _cap->point1();

    // Test seg-direction cross box-edges.
    for (i = 0; i < 3; i++)
    {
        axis = mBox->axis(i).cross(cap_direction);
        IntrUtil<Real>::normalize(axis);

        if(!IntrAxis<Box>::Find(axis, segment,radius, *mBox, dmax,
            mContactTime, side, capContact, boxContact,config_modified))
            return false;

        if(config_modified){
            _sep_axis = axis;
        }
    }

    defaulttype::Vec<3,Real> relVelocity = mBox->v() - _cap->v();
    // Test velocity cross box-faces.
    for (i = 0; i < 3; i++)
    {
        axis = relVelocity.cross(mBox->axis(i));
        IntrUtil<Real>::normalize(axis);

        if(!IntrAxis<Box>::Find(axis, segment,radius, *mBox, dmax,
            mContactTime, side, capContact, boxContact,config_modified))
            return false;

        if(config_modified){
            _sep_axis = axis;
        }
    }

    if(mContactTime < (Real)0)
        _is_colliding = true;

    FindContactSet<Box>(segment,radius ,*mBox, _sep_axis,side, capContact, boxContact,
        mContactTime, _pt_on_first, _pt_on_second);

    if(capContact.have_naxis){
        _sep_axis = capContact.axis;
    }
    else{
        defaulttype::Vec<3,Real> projP = segment[0] + cap_direction * ((cap_direction  * (_pt_on_first - segment[0]))/cap_direction.norm2());

        _sep_axis = _pt_on_first - projP;
        _sep_axis.normalize();
    }

    return true;
}

}
}
}
