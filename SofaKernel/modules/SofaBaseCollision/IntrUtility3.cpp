/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#define SOFA_COMPONENT_COLLISION_INTRUTILITY3_CPP
#include <SofaBaseCollision/IntrUtility3.inl>


namespace sofa{
using namespace defaulttype;
namespace component{
namespace collision{

//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Explicit instantiation.
//----------------------------------------------------------------------------
template struct SOFA_BASE_COLLISION_API IntrUtil<double>;

#ifndef SOFA_FLOAT
template struct SOFA_BASE_COLLISION_API IntrUtil<TOBB<RigidTypes> >;
template class SOFA_BASE_COLLISION_API IntrConfiguration<double>;
template struct SOFA_BASE_COLLISION_API IntrConfigManager<double>;
template struct SOFA_BASE_COLLISION_API IntrConfigManager<TOBB<Rigid3dTypes> >;
template class SOFA_BASE_COLLISION_API IntrAxis<TOBB<Rigid3dTypes> >;
template class SOFA_BASE_COLLISION_API FindContactSet<TOBB<Rigid3dTypes> >;
template SOFA_BASE_COLLISION_API void ClipConvexPolygonAgainstPlane<double> (const Vec<3,double>&, double,int&, Vec<3,double>*);
template SOFA_BASE_COLLISION_API Vec<3,double> GetPointFromIndex<double> (int, const MyBox<double>&);
template SOFA_BASE_COLLISION_API Vec<3,Rigid3dTypes::Real> getPointFromIndex<Rigid3dTypes> (int index, const TOBB<Rigid3dTypes>& box);
template SOFA_BASE_COLLISION_API class CapIntrConfiguration<double>;
#endif
#ifndef SOFA_DOUBLE
template struct SOFA_BASE_COLLISION_API IntrUtil<float>;
template struct SOFA_BASE_COLLISION_API IntrUtil<TOBB<Rigid3fTypes> >;
template class SOFA_BASE_COLLISION_API IntrConfiguration<float>;
template struct SOFA_BASE_COLLISION_API IntrConfigManager<float>;
template struct SOFA_BASE_COLLISION_API IntrConfigManager<TOBB<Rigid3fTypes> >;
template class SOFA_BASE_COLLISION_API IntrAxis<TOBB<Rigid3fTypes> >;
template class SOFA_BASE_COLLISION_API FindContactSet<TOBB<Rigid3fTypes> >;
template SOFA_BASE_COLLISION_API void ClipConvexPolygonAgainstPlane<float> (const Vec<3,float>&, float,int&, Vec<3,float>*);
template SOFA_BASE_COLLISION_API Vec<3,float> GetPointFromIndex<float> (int index, const MyBox<float>& box);
template SOFA_BASE_COLLISION_API Vec<3,Rigid3fTypes::Real> getPointFromIndex<Rigid3fTypes> (int index, const TOBB<Rigid3fTypes>& box);
template SOFA_BASE_COLLISION_API class CapIntrConfiguration<float>;
#endif
//----------------------------------------------------------------------------

}
}
}
