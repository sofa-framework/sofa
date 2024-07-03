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
#define SOFA_COMPONENT_CONTAINER_MAPPEDOBJECT_CPP

#include <sofa/component/statecontainer/MappedObject.inl>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/core/ObjectFactory.h>

namespace sofa::component::statecontainer
{

using namespace defaulttype;

template class SOFA_COMPONENT_STATECONTAINER_API MappedObject<Vec1Types>;
template class SOFA_COMPONENT_STATECONTAINER_API MappedObject<Vec2Types>;
template class SOFA_COMPONENT_STATECONTAINER_API MappedObject<Vec3Types>;
template class SOFA_COMPONENT_STATECONTAINER_API MappedObject<Vec6Types>;
template class SOFA_COMPONENT_STATECONTAINER_API MappedObject<Rigid3Types>;
template class SOFA_COMPONENT_STATECONTAINER_API MappedObject<Rigid2Types>;

void registerMappedObject(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Mapped state vectors")
        .add< MappedObject<Vec1Types> >()
        .add< MappedObject<Vec3Types> >(true) // default template
        .add< MappedObject<Vec2Types> >()
        .add< MappedObject<Vec6Types> >()
        .add< MappedObject<Rigid3Types> >()
        .add< MappedObject<Rigid2Types> >());
}

} // namespace sofa::component::statecontainer
