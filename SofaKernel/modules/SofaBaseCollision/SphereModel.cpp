/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_COMPONENT_COLLISION_SPHEREMODEL_CPP
#include <SofaBaseCollision/SphereModel.inl>
#include <sofa/core/ObjectFactory.h>




namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace helper;



#ifndef SOFA_FLOAT
template <> SOFA_BASE_COLLISION_API
Vector3 TSphere<defaulttype::Vec3dTypes >::getContactPointByNormal( const Vector3& )
{
    return center();
}
template <> SOFA_BASE_COLLISION_API
Vector3 TSphere<defaulttype::Vec3dTypes >::getContactPointWithSurfacePoint( const Vector3& )
{
    return center();
}
#endif

#ifndef SOFA_DOUBLE
template <> SOFA_BASE_COLLISION_API
Vector3 TSphere<defaulttype::Vec3fTypes >::getContactPointByNormal( const Vector3& )
{
    return center();
}
template <> SOFA_BASE_COLLISION_API
Vector3 TSphere<defaulttype::Vec3fTypes >::getContactPointWithSurfacePoint( const Vector3& )
{
    return center();
}
#endif



SOFA_DECL_CLASS(Sphere)

int SphereModelClass = core::RegisterObject("Collision model which represents a set of Spheres")
#ifndef SOFA_FLOAT
        .add<  TSphereModel<Vec3dTypes> >()
        .add<TSphereModel<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add < TSphereModel<Vec3fTypes> >()
        .add<TSphereModel<Rigid3fTypes> >()
#endif
        .addAlias("Sphere")
        .addAlias("SphereModel")
        ;

#ifndef SOFA_FLOAT
template class SOFA_BASE_COLLISION_API TSphere<defaulttype::Vec3dTypes>;
//template class SOFA_BASE_COLLISION_API TSphere<defaulttype::Rigid3dTypes>; // Can't compile due to type mismatches in pFree() method.
template class SOFA_BASE_COLLISION_API TSphereModel<defaulttype::Vec3dTypes>;
template class SOFA_BASE_COLLISION_API TSphereModel<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_COLLISION_API TSphere<defaulttype::Vec3fTypes>;
//template class SOFA_BASE_COLLISION_API TSphere<defaulttype::Rigid3fTypes>; // Can't compile due to type mismatches in pFree() method.
template class SOFA_BASE_COLLISION_API TSphereModel<defaulttype::Vec3fTypes>;
template class SOFA_BASE_COLLISION_API TSphereModel<defaulttype::Rigid3fTypes>;
#endif

} // namespace collision

} // namespace component

} // namespace sofa

