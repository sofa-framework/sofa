// File modified from GeometricTools
// http://www.geometrictools.com/


#ifndef WM5INTRSEGMENT3BOX3_H
#define WM5INTRSEGMENT3BOX3_H

#include <sofa/component/collision/Intersector.h>
#include <sofa/component/collision/CapsuleModel.h>
#include <sofa/component/collision/OBBModel.h>

namespace sofa{
namespace component{
namespace collision{


/**
  *TDataTypes is the capsule type and TDataTypes2 the OBB type.
  */
template <typename TDataTypes,typename TDataTypes2>
class TIntrCapsuleOBB : public Intersector<typename TDataTypes::Real>
{
public:
    typedef TCapsule<TDataTypes> IntrCap;
    typedef typename IntrCap::Real Real;
    typedef typename IntrCap::Coord Coord;
    typedef TOBB<TDataTypes2> Box;
    typedef Vec<3,Real> Vec3;

    TIntrCapsuleOBB (const IntrCap& capsule, const Box & box);

    // Dynamic find-intersection query.  The first point of contact is
    // accessed by GetPoint(0), when there is a single contact, or by
    // GetPoint(0) and GetPoint(1), when the contact is a segment, in which
    // case the fetched points are the segment endpoints.  The first time of
    // contact is accessed by GetContactTime().
    bool Find (Real tmax, const Vec<3,Real>& velocity0,
        const Vec<3,Real>& velocity1);

    bool FindStatic (Real dmax);

    int GetQuantity () const;
    const Vec<3,Real>& GetPoint (int i) const;
private:
    // The objects to intersect.
    const IntrCap* _cap;
    const Box * mBox;

    // Information about the intersection set.
    int mQuantity;
    Vec<3,Real> mPoint[2];
    using Intersector<Real>::_is_colliding;
    using Intersector<Real>::_pt_on_first;
    using Intersector<Real>::_pt_on_second;
    using Intersector<Real>::mContactTime;
    using Intersector<Real>::_sep_axis;
};

typedef TIntrCapsuleOBB<Vec3Types,Rigid3Types> IntrCapsuleOBB;

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_BASE_COLLISION)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_COLLISION_API TIntrCapsuleOBB<Vec3dTypes,Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_COLLISION_API TIntrCapsuleOBB<Vec3fTypes,Rigid3fTypes>;
#endif
#endif

}
}
}

#endif
