/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace container
{

using namespace defaulttype;

int MappedObjectClass = core::RegisterObject("Mapped state vectors")
        .add< MappedObject<Vec1Types> >()
        .add< MappedObject<Vec3Types> >(true) // default template
        .add< MappedObject<Vec2Types> >()
        .add< MappedObject<Vec6Types> >()
        .add< MappedObject<Rigid3Types> >()
        .add< MappedObject<Rigid2Types> >()
        ;

// template specialization must be in the same namespace as original namespace for GCC 4.1
// g++ 4.1 requires template instantiations to be declared on a parent namespace from the template class.
template class MappedObject<Vec1Types>;
template class MappedObject<Vec2Types>;
template class MappedObject<Vec3Types>;
template class MappedObject<Vec6Types>;
template class MappedObject<Rigid3Types>;
template class MappedObject<Rigid2Types>;

}

} // namespace component

} // namespace sofa
