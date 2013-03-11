#ifndef INTROBBTRIANGLE_H
#define INTROBBTRIANGLE_H
#include <sofa/core/collision/Intersection.h>
#include <sofa/component/collision/OBBModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/IntrMeshUtility.h>
#include <sofa/component/collision/Intersector.h>

namespace sofa{
namespace component{
namespace collision{

/**
  *TDataTypes is the sphere type and TDataTypes2 the OBB type.
  */
template <class TDataTypes,class TDataTypes2>
class TIntrTriangleOBB : public Intersector<typename TDataTypes::Real>
{
public:
    typedef TTriangle<TDataTypes> IntrTri;
    typedef typename TDataTypes::Real Real;
    typedef typename IntrTri::Coord Coord;
    typedef TOBB<TDataTypes2> Box;
    typedef Vec<3,Real> Vec3;

    TIntrTriangleOBB (const IntrTri& tri, const Box & box);

    bool Find(Real tmax);

    Real distance()const;
private:
    using Intersector<Real>::_is_colliding;
    using Intersector<Real>::_pt_on_first;
    using Intersector<Real>::_pt_on_second;
    using Intersector<Real>::mContactTime;
    using Intersector<Real>::_sep_axis;

    // The objects to intersect.
    const IntrTri* _tri;
    const Box * mBox;
};

typedef TIntrTriangleOBB<Vec3Types,Rigid3Types> IntrTriangleOBB;

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_BASE_COLLISION)
#ifndef SOFA_FLOAT
extern template class SOFA_MESH_COLLISION_API TIntrTriangleOBB<defaulttype::Vec3dTypes,defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MESH_COLLISION_API TIntrTriangleOBB<defaulttype::Vec3fTypes,defaulttype::Rigid3fTypes>;
#endif
#endif

}
}
}

#endif // INTROBBTRIANGLE_H
