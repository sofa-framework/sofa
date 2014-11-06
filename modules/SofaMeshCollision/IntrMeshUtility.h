#ifndef INTRMESHUTIL_H
#define INTRMESHUTIL_H

#include <sofa/SofaCommon.h>
#include <SofaBaseCollision/IntrUtility3.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa{
namespace component{
namespace collision{

template <class DataType>
struct IntrUtil<TTriangle<DataType> >{
    typedef typename DataType::Real Real;
    typedef TTriangle<DataType> IntrTri;

    /**
      *Returns the squared distance between old pt and projected pt.
      */
    static Real project(defaulttype::Vec<3,Real> & pt,const TTriangle<DataType> & tri);

    static SReal triSegNearestPoints(const IntrTri & tri,const defaulttype::Vec<3,Real> seg[2],defaulttype::Vec<3,Real> & pt_on_tri,defaulttype::Vec<3,Real> & pt_on_seg);

    static void triFaceNearestPoints(const IntrTri & tri,const defaulttype::Vec<3,Real> * face,int n,defaulttype::Vec<3,Real> & pt_on_tri,defaulttype::Vec<3,Real> & pt_on_face);
};


template <class TDataTypes1,class TDataTypes2>
class IntrAxis<TTriangle<TDataTypes1>,TOBB<TDataTypes2> >
{
public:
    typedef typename TDataTypes1::Real Real;
    typedef TOBB<TDataTypes2> Box;
    typedef typename TTriangle<TDataTypes1>::Coord Coord;
    typedef IntrConfiguration<Real> IntrConf;
    typedef TTriangle<TDataTypes1> IntrTri;

    // Find-query for intersection of projected intervals.  The velocity
    // input is the difference objectVelocity1 - objectVelocity0.  The
    // first and last times of contact are computed, as is information about
    // the contact configuration and the ordering of the projections (the
    // contact side).

        static bool Find (const Coord& axis,
                          const IntrTri & triangle, const Box& box,
                          Real dmax, Real& tfirst,
                          int& side, IntrConfiguration<Real>& triCfgFinal,
                          IntrConfiguration<Real>& boxCfgFinal,bool & config_modified);

    // if axis is found as the final separating axis then final_axis is updated and
    // become equal axis after this method
};

template<class TDataTypes>
struct IntrConfigManager<TTriangle<TDataTypes> >{
    typedef TTriangle<TDataTypes> IntrTri;
    typedef typename TDataTypes::Real Real;
    typedef typename TTriangle<TDataTypes>::Coord Coord;

    static void init(const  Coord& axis,
                   const IntrTri & tri, IntrConfiguration<Real>& cfg);
};

template <class TDataTypes1,class TDataTypes2>
class  FindContactSet<TTriangle<TDataTypes1>,TOBB<TDataTypes2> >
{
public:
    typedef typename TDataTypes1::Real Real;
    typedef TOBB<TDataTypes2> Box;
    typedef TTriangle<TDataTypes1> IntrTri;

    FindContactSet (const IntrTri& triangle,
                    const Box& box,const defaulttype::Vec<3,Real> & axis ,int side, const IntrConfiguration<Real>& triCfg,
                    const IntrConfiguration<Real>& boxCfg,Real tfirst,
                    defaulttype::Vec<3,Real> & pt_on_tri,defaulttype::Vec<3,Real> & pt_on_box);

};



#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_MESH_COLLISION)
#ifndef SOFA_FLOAT
extern template struct SOFA_MESH_COLLISION_API IntrUtil<TTriangle<defaulttype::Vec3dTypes> >;
extern template class SOFA_MESH_COLLISION_API FindContactSet<TTriangle<defaulttype::Vec3dTypes>,TOBB<defaulttype::Rigid3dTypes> >;
extern template class SOFA_MESH_COLLISION_API IntrAxis<TTriangle<defaulttype::Vec3dTypes>,TOBB<defaulttype::Rigid3dTypes> >;
extern template struct SOFA_MESH_COLLISION_API IntrConfigManager<TTriangle<defaulttype::Vec3dTypes> >;
#endif
#ifndef SOFA_DOUBLE
extern template struct SOFA_MESH_COLLISION_API IntrUtil<TTriangle<defaulttype::Vec3fTypes> >;
extern template class SOFA_MESH_COLLISION_API FindContactSet<TTriangle<defaulttype::Vec3fTypes>,TOBB<defaulttype::Rigid3fTypes> >;
extern template class SOFA_MESH_COLLISION_API IntrAxis<TTriangle<defaulttype::Vec3fTypes>,TOBB<defaulttype::Rigid3fTypes> >;
extern template struct SOFA_MESH_COLLISION_API IntrConfigManager<TTriangle<defaulttype::Vec3fTypes> >;
#endif
#endif

}
}
}
#endif // INTRMESHUTIL_H
