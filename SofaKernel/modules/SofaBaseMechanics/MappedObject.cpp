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
#define SOFA_COMPONENT_CONTAINER_MAPPEDOBJECT_CPP
#include <SofaBaseMechanics/MappedObject.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace container
{

using namespace defaulttype;

SOFA_DECL_CLASS(MappedObject)

int MappedObjectClass = core::RegisterObject("Mapped state vectors")
#ifndef SOFA_FLOAT
        .add< MappedObject<Vec2dTypes> >()
        .add< MappedObject<Vec1dTypes> >()
        .add< MappedObject<Vec6dTypes> >()
        .add< MappedObject<Rigid3dTypes> >()
        .add< MappedObject<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< MappedObject<Vec2fTypes> >()
        .add< MappedObject<Vec1fTypes> >()
        .add< MappedObject<Vec6fTypes> >()
        .add< MappedObject<Rigid3fTypes> >()
        .add< MappedObject<Rigid2fTypes> >()
#endif

#ifdef SOFA_FLOAT
        .add< MappedObject<Vec3fTypes> >(true) // default template
#else
#ifndef SOFA_DOUBLE
        .add< MappedObject<Vec3fTypes> >() // default template
#endif
        .add< MappedObject<Vec3dTypes> >(true) // default template
#endif
        .add< MappedObject<LaparoscopicRigid3Types> >()
        ;

// template specialization must be in the same namespace as original namespace for GCC 4.1
// g++ 4.1 requires template instantiations to be declared on a parent namespace from the template class.
#ifndef SOFA_FLOAT
template class MappedObject<Vec3dTypes>;
template class MappedObject<Vec2dTypes>;
template class MappedObject<Vec1dTypes>;
template class MappedObject<Vec6dTypes>;
template class MappedObject<Rigid3dTypes>;
template class MappedObject<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class MappedObject<Vec3fTypes>;
template class MappedObject<Vec2fTypes>;
template class MappedObject<Vec1fTypes>;
template class MappedObject<Vec6fTypes>;
template class MappedObject<Rigid3fTypes>;
template class MappedObject<Rigid2fTypes>;
#endif
template class MappedObject<LaparoscopicRigid3Types>;

}

} // namespace component

} // namespace sofa
