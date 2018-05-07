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
#define SOFA_COMPONENT_ENGINE_TRANSFORMENGINE_CPP
#include <SofaGeneralEngine/TransformEngine.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

SOFA_DECL_CLASS(TransformEngine)

int TransformEngineClass = core::RegisterObject("Transform position of 3d points")
#ifdef SOFA_FLOAT
        .add< TransformEngine<defaulttype::Vec3fTypes> >(true) // default template
#else
        .add< TransformEngine<defaulttype::Vec3dTypes> >(true) // default template
#ifndef SOFA_DOUBLE
        .add< TransformEngine<defaulttype::Vec3fTypes> >()
#endif
#endif
#ifndef SOFA_FLOAT
        .add< TransformEngine<defaulttype::Vec1dTypes> >()
        .add< TransformEngine<defaulttype::Vec2dTypes> >()
        .add< TransformEngine<defaulttype::Rigid2dTypes> >()
        .add< TransformEngine<defaulttype::Rigid3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< TransformEngine<defaulttype::Vec1fTypes> >()
        .add< TransformEngine<defaulttype::Vec2fTypes> >()
        .add< TransformEngine<defaulttype::Rigid2fTypes> >()
        .add< TransformEngine<defaulttype::Rigid3fTypes> >()
#endif //SOFA_DOUBLE
        .add< TransformEngine<defaulttype::ExtVec3fTypes> >()
        ;

#ifndef SOFA_FLOAT
template class SOFA_GENERAL_ENGINE_API TransformEngine<defaulttype::Vec1dTypes>;
template class SOFA_GENERAL_ENGINE_API TransformEngine<defaulttype::Vec2dTypes>;
template class SOFA_GENERAL_ENGINE_API TransformEngine<defaulttype::Vec3dTypes>;
template class SOFA_GENERAL_ENGINE_API TransformEngine<defaulttype::Rigid2dTypes>;
template class SOFA_GENERAL_ENGINE_API TransformEngine<defaulttype::Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_ENGINE_API TransformEngine<defaulttype::Vec1fTypes>;
template class SOFA_GENERAL_ENGINE_API TransformEngine<defaulttype::Vec2fTypes>;
template class SOFA_GENERAL_ENGINE_API TransformEngine<defaulttype::Vec3fTypes>;
template class SOFA_GENERAL_ENGINE_API TransformEngine<defaulttype::Rigid2fTypes>;
template class SOFA_GENERAL_ENGINE_API TransformEngine<defaulttype::Rigid3fTypes>;
#endif //SOFA_DOUBLE
template class SOFA_GENERAL_ENGINE_API TransformEngine<defaulttype::ExtVec3fTypes>;


} // namespace constraint

} // namespace component

} // namespace sofa

