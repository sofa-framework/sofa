// File modified from GeometricTools
// http://www.geometrictools.com/

#ifndef WM5INTRBOX3BOX3_H
#define WM5INTRBOX3BOX3_H

#include <sofa/component/collision/Intersector.h>
#include <sofa/component/collision/OBBModel.h>

namespace sofa{
namespace component{
namespace collision{

template <class TDataTypes>
class TIntrOBBOBB : public Intersector<typename TDataTypes::Real>
{
public:
    typedef typename TDataTypes::Real Real;
    typedef typename TDataTypes::Deriv TVector;
    typedef TOBB<TDataTypes> Box;
    typedef typename Box::Coord Coord;

    TIntrOBBOBB (const Box& box0, const Box& box1);

    // Object access.
    const Box& GetBox0 () const;
    const Box& GetBox1 () const;

    // Static test-intersection query.
    virtual bool Test ();

    // Dynamic test-intersection query.  The first time of contact (if any)
    // is computed, but not any information about the contact set.
    virtual bool Test (Real tmax, const Vec<3,Real>& velocity0,
        const Vec<3,Real>& velocity1);

    // Dynamic find-intersection query.  The contact set is computed.
    bool Find (Real tmax, const Vec<3,Real>& velocity0,
        const Vec<3,Real>& velocity1);

    // Dynamic find-intersection query.  The contact set is computed.
    bool FindStatic (Real dmax);

    // The intersection set for dynamic find-intersection.
    int GetQuantity () const;
    const Vec<3,Real>& GetPoint (int i) const;

    // Dynamic test-intersection query where the boxes have constant linear
    // velocities *and* constant angular velocities.  The length of the
    // rotation axes are the angular speeds.  A differential equation solver
    // is used to predict the intersection.  The input numSteps is the
    // number of iterations for the numerical ODE solver.
    bool Test (Real tmax, int numSteps, const Vec<3,Real>& velocity0,
        const Vec<3,Real>& rotCenter0, const Vec<3,Real>& rotAxis0,
        const Vec<3,Real>& velocity1, const Vec<3,Real>& rotCenter1,
        const Vec<3,Real>& rotAxis1);
private:
    // Support for dynamic queries.  The inputs are the projection intervals
    // for the boxes onto a potential separating axis, the relative speed of
    // the intervals, and the maximum time for the query.  The outputs are
    // the first time when separating fails and the last time when separation
    // begins again along that axis.  The outputs are *updates* in the sense
    // that this function is called repeatedly for the potential separating
    // axes.  The output first time is updated only if it is larger than
    // the input first time.  The output last time is updated only if it is
    // smaller than the input last time.
    //
    // NOTE:  The BoxBoxAxisTest function could be used, but the box-box
    // code optimizes the projections of the boxes onto the various axes.
    // This function is effectively BoxBoxAxisTest but without the dot product
    // of axis-direction and velocity to obtain speed.  The optimizations are
    // to compute the speed with fewer operations.
    bool IsSeparated (Real min0, Real max0, Real min1, Real max1, Real speed,
        Real tmax, Real& tlast);

    // The objects to intersect.
    const Box* mBox0;
    const Box* mBox1;

    // The intersection set for dynamic find-intersection.  The worst case
    // is a polygon with 8 vertices.
    int mQuantity;
    Vec<3,Real> mPoint[8];
    using Intersector<Real>::_is_colliding;
    using Intersector<Real>::_pt_on_first;
    using Intersector<Real>::_pt_on_second;
    using Intersector<Real>::mContactTime;
    using Intersector<Real>::_sep_axis;

    int mIntersectionType;
};

typedef TIntrOBBOBB<RigidTypes> IntrOBBOBB;

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_BASE_COLLISION)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_COLLISION_API TIntrOBBOBB<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_COLLISION_API TIntrOBBOBB<defaulttype::Rigid3fTypes>;
#endif
#endif

}
}
}

#endif
