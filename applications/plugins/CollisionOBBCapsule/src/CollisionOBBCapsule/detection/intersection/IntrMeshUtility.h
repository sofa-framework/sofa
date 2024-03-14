/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <CollisionOBBCapsule/config.h>

#include <CollisionOBBCapsule/detection/intersection/IntrUtility3.h>
#include <sofa/component/collision/geometry/TriangleModel.h>
#include <sofa/defaulttype/VecTypes.h>

namespace collisionobbcapsule::detection::intersection
{
using sofa::component::collision::geometry::TTriangle;
using collisionobbcapsule::geometry::TOBB;

template <class DataType>
struct IntrUtil<TTriangle<DataType> >{
    typedef typename DataType::Real Real;
    typedef TTriangle<DataType> IntrTri;

    /**
      *Returns the squared distance between old pt and projected pt.
      */
    static Real project(type::Vec<3,Real> & pt,const TTriangle<DataType> & tri);

    static SReal triSegNearestPoints(const IntrTri & tri,const type::Vec<3,Real> seg[2],type::Vec<3,Real> & pt_on_tri,type::Vec<3,Real> & pt_on_seg);

    static void triFaceNearestPoints(const IntrTri & tri,const type::Vec<3,Real> * face,int n,type::Vec<3,Real> & pt_on_tri,type::Vec<3,Real> & pt_on_face);
};


template <class TDataTypes1,class TDataTypes2>
class COLLISIONOBBCAPSULE_API IntrAxis<TTriangle<TDataTypes1>,TOBB<TDataTypes2> >
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
                    const Box& box,const type::Vec<3,Real> & axis ,int side, const IntrConfiguration<Real>& triCfg,
                    const IntrConfiguration<Real>& boxCfg,Real tfirst,
                    type::Vec<3,Real> & pt_on_tri,type::Vec<3,Real> & pt_on_box);

};



#if  !defined(SOFA_COMPONENT_COLLISION_INTRMESHUTILITY_CPP)
extern template struct COLLISIONOBBCAPSULE_API IntrUtil<TTriangle<defaulttype::Vec3Types> >;
extern template class COLLISIONOBBCAPSULE_API FindContactSet<TTriangle<defaulttype::Vec3Types>,TOBB<defaulttype::Rigid3Types> >;
extern template class COLLISIONOBBCAPSULE_API IntrAxis<TTriangle<defaulttype::Vec3Types>,TOBB<defaulttype::Rigid3Types> >;
extern template struct COLLISIONOBBCAPSULE_API IntrConfigManager<TTriangle<defaulttype::Vec3Types> >;

#endif

} // namespace collisionobbcapsule::detection::intersection
