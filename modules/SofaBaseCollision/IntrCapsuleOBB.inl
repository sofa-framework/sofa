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
    Vec3 segment[2] =
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

    Vec3 cap_direction = segment[1] - segment[0];

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
