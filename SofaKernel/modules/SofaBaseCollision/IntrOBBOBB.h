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
// File modified from GeometricTools
// http://www.geometrictools.com/

#ifndef SOFA_COMPONENT_COLLISION_INTROBBOBB_H
#define SOFA_COMPONENT_COLLISION_INTROBBOBB_H
#include "config.h"

#include <SofaBaseCollision/Intersector.h>
#include <SofaBaseCollision/OBBModel.h>

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
    virtual bool Test (Real tmax, const sofa::defaulttype::Vec<3,Real>& velocity0,
        const sofa::defaulttype::Vec<3,Real>& velocity1);

    // Dynamic find-intersection query.  The contact set is computed.
    bool Find (Real dmax);
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

    using Intersector<Real>::_is_colliding;
    using Intersector<Real>::_pt_on_first;
    using Intersector<Real>::_pt_on_second;
    using Intersector<Real>::mContactTime;
    using Intersector<Real>::_sep_axis;
};

typedef TIntrOBBOBB<sofa::defaulttype::RigidTypes> IntrOBBOBB;

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_INTROBBOBB_CPP)
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
