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
#define SOFA_COMPONENT_ENGINE_TRANSFORMENGINE_CPP
#include <sofa/component/engine/transform/TransformEngine.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::engine::transform
{

void registerTransformEngine(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Engine applying a transformation on 3d points (translation / rotation).")
        .add< TransformEngine<defaulttype::Vec3Types> >(true) // default template
        .add< TransformEngine<defaulttype::Vec1Types> >()
        .add< TransformEngine<defaulttype::Vec2Types> >()
        .add< TransformEngine<defaulttype::Rigid2Types> >()
        .add< TransformEngine<defaulttype::Rigid3Types> >());
}

template class SOFA_COMPONENT_ENGINE_TRANSFORM_API TransformEngine<defaulttype::Vec1Types>;
template class SOFA_COMPONENT_ENGINE_TRANSFORM_API TransformEngine<defaulttype::Vec2Types>;
template class SOFA_COMPONENT_ENGINE_TRANSFORM_API TransformEngine<defaulttype::Vec3Types>;
template class SOFA_COMPONENT_ENGINE_TRANSFORM_API TransformEngine<defaulttype::Rigid2Types>;
template class SOFA_COMPONENT_ENGINE_TRANSFORM_API TransformEngine<defaulttype::Rigid3Types>;

} //namespace sofa::component::engine::transform
