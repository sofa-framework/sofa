// Geometric Tools, LLC
// Copyright (c) 1998-2012
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
//
// File Version: 5.0.1 (2010/10/01)

#include <sofa/component/collision/IntrUtility3.h>
#include <sofa/component/collision/IntrCapsuleOBB.h>

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
    mQuantity = 0;
}

template <typename TDataTypes,typename TDataTypes2>
bool TIntrCapsuleOBB<TDataTypes,TDataTypes2>::Find (Real tmax,
    const Vec3& velocity0, const Vec3& velocity1)
{
    bool config_modified;
    bool simple_sep_axis = true;
    bool found_unoriented_axis = false;
    mQuantity = 0;

    // Get the endpoints of the segment.
    Vec3 segment[2] =
    {
        _cap->point1(),
        _cap->point2()
    };
    Real radius = _cap->radius();

    // Get the box velocity relative to the segment.
    Vec3 relVelocity = velocity1 - velocity0;

    mContactTime = -std::numeric_limits<Real>::max();
    Real tlast = std::numeric_limits<Real>::max();

    std::cout<<"FIRST mContactTime "<<mContactTime<<std::endl;
    std::cout<<"FIRST TLAST "<<tlast<<std::endl;

    int i;
    Vec3 axis;
    int side = IntrConfiguration<Real>::NONE;
    IntrConfiguration<Real> boxContact;
    CapIntrConfiguration<Real> capContact;

    // Test box normals.
    for (i = 0; i < 3; ++i)
    {
        axis = mBox->axis(i);
        if (!IntrAxis<TDataTypes2>::Find(axis, segment,radius, *mBox, relVelocity, tmax,
            mContactTime, tlast, side, capContact, boxContact,config_modified))
        {
            return false;
        }

        if(config_modified){
            _sep_axis = -axis;
            simple_sep_axis = true;
        }
    }

    Vec3 cap_direction = segment[1] - segment[0];

    // Test seg-direction cross box-edges.
    for (i = 0; i < 3; i++)
    {
        axis = mBox->axis(i).cross(cap_direction);
        IntrUtil<Real>::normalize(axis);

        if (!IntrAxis<TDataTypes2>::Find(axis, segment,radius, *mBox, relVelocity, tmax,
                                  mContactTime, tlast, side, capContact, boxContact,config_modified))
        {
            return false;
        }

        if(config_modified){
            _sep_axis = axis;
            found_unoriented_axis = true;
        }
    }

    // Test velocity cross box-faces.
    for (i = 0; i < 3; i++)
    {
        axis = relVelocity.cross(mBox->axis(i));
        IntrUtil<Real>::normalize(axis);

        if (!IntrAxis<TDataTypes2>::Find(axis, segment,radius, *mBox, relVelocity, tmax,
            mContactTime, tlast, side, capContact, boxContact,config_modified))
        {
            return false;
        }

        if(config_modified){
            simple_sep_axis = false;
        }
    }

//    if (mContactTime < (Real)0 || side == IntrConfiguration<Real>::NONE)
//    {
//        // intersecting now
////        _colliding = true;

////        Real alpha =  (cap_direction  * (mBox->center() - segment[0]))/cap_direction.norm2();
////        if(alpha > 1)
////            alpha  = 1;
////        else if(alpha < 0)
////            alpha = 0;

////        Vec3 projC;
////        if(alpha < 0){
////            _sep_axis = mBox->center() - segment[0];
////        }
////        else if(alpha > 1){
////            _sep_axis = mBox->center() - segment[1];
////        }
////        else{
////            _sep_axis = mBox->center() - (segment[0] + cap_direction * alpha);
////        }

////        _distance = _sep_axis.norm();

//        std::cout<<"mContactTime !!!!!!!! "<<mContactTime<<std::endl;

//        return false;
//    }

    if(mContactTime < (Real)0)
        _is_colliding = true;

    FindContactSet<TDataTypes2>(segment,radius ,*mBox, side, capContact, boxContact,
        velocity0, velocity1, mContactTime, mQuantity, mPoint);

    projectIntPoints<Real>(velocity0,velocity1,mContactTime,mPoint,mQuantity,_pt_on_first,_pt_on_second);

    if(capContact.have_naxis){
        _sep_axis = capContact.axis;
    }
    else if(found_unoriented_axis){
        Vec<3,Real> projP = segment[0] + cap_direction * ((cap_direction  * (_pt_on_first - segment[0]))/cap_direction.norm2());

        if((_pt_on_first - projP) * _sep_axis < 0)
            _sep_axis = -_sep_axis;
    }
    else if(!simple_sep_axis){
        Vec<3,Real> projP = segment[0] + cap_direction * ((cap_direction  * (_pt_on_first - segment[0]))/cap_direction.norm2());

        _sep_axis = _pt_on_first - projP;
        _sep_axis.normalize();
    }

    return true;
}


template <typename TDataTypes,typename TDataTypes2>
bool TIntrCapsuleOBB<TDataTypes,TDataTypes2>::FindStatic (Real dmax)
{
    bool config_modified;
    mQuantity = 0;

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
        IntrAxis<TDataTypes2>::FindStatic(axis, segment,radius, *mBox, dmax,
            mContactTime, side, capContact, boxContact,config_modified);

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

        IntrAxis<TDataTypes2>::FindStatic(axis, segment,radius, *mBox, dmax,
            mContactTime, side, capContact, boxContact,config_modified);

        if(config_modified){
            _sep_axis = axis;
        }
    }

    Vec<3,Real> relVelocity = mBox->lvelocity() - _cap->velocity();
    // Test velocity cross box-faces.
    for (i = 0; i < 3; i++)
    {
        axis = relVelocity.cross(mBox->axis(i));
        IntrUtil<Real>::normalize(axis);

        IntrAxis<TDataTypes2>::FindStatic(axis, segment,radius, *mBox, dmax,
            mContactTime, side, capContact, boxContact,config_modified);

        if(config_modified){
            _sep_axis = axis;
        }
    }

//    if (mContactTime < (Real)0 || side == IntrConfiguration<Real>::NONE)
//    {
//        // intersecting now
////        _colliding = true;

////        Real alpha =  (cap_direction  * (mBox->center() - segment[0]))/cap_direction.norm2();
////        if(alpha > 1)
////            alpha  = 1;
////        else if(alpha < 0)
////            alpha = 0;

////        Vec3 projC;
////        if(alpha < 0){
////            _sep_axis = mBox->center() - segment[0];
////        }
////        else if(alpha > 1){
////            _sep_axis = mBox->center() - segment[1];
////        }
////        else{
////            _sep_axis = mBox->center() - (segment[0] + cap_direction * alpha);
////        }

////        _distance = _sep_axis.norm();

//        std::cout<<"mContactTime !!!!!!!! "<<mContactTime<<std::endl;

//        return false;
//    }

    if(mContactTime < (Real)0)
        _is_colliding = true;

    FindContactSet<TDataTypes2>(segment,radius ,*mBox, _sep_axis,side, capContact, boxContact,
        mContactTime, _pt_on_first, _pt_on_second);

    if(capContact.have_naxis){
        _sep_axis = capContact.axis;
    }
    else{
        Vec<3,Real> projP = segment[0] + cap_direction * ((cap_direction  * (_pt_on_first - segment[0]))/cap_direction.norm2());

        _sep_axis = _pt_on_first - projP;
        _sep_axis.normalize();
    }

    return true;
}



//----------------------------------------------------------------------------
template <typename TDataTypes,typename TDataTypes2>
int TIntrCapsuleOBB<TDataTypes,TDataTypes2>::GetQuantity () const
{
    return mQuantity;
}
//----------------------------------------------------------------------------
template <typename TDataTypes,typename TDataTypes2>
const Vec<3,typename TIntrCapsuleOBB<TDataTypes,TDataTypes2>::Real>& TIntrCapsuleOBB<TDataTypes,TDataTypes2>::GetPoint (int i) const
{
    return mPoint[i];
}

}
}
}
