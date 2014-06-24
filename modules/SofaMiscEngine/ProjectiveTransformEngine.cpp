/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_COMPONENT_ENGINE_PROJECTIVETRANSFORMENGINE_CPP
#include <SofaMiscEngine/ProjectiveTransformEngine.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

  SOFA_DECL_CLASS(ProjectiveTransformEngine)

  int ProjectiveTransformEngineClass = core::RegisterObject("Project the position of 3d points onto a plane according to a projection matrix")
#ifdef SOFA_FLOAT
        .add< ProjectiveTransformEngine<defaulttype::Vec3fTypes> >(true) // default template
#else
        .add< ProjectiveTransformEngine<defaulttype::Vec3dTypes> >(true) // default template
#endif
#ifndef SOFA_DOUBLE
        .add< ProjectiveTransformEngine<defaulttype::Vec3fTypes> >()
#endif
        .add< ProjectiveTransformEngine<defaulttype::ExtVec3fTypes> >()
        ;

#ifndef SOFA_FLOAT
template class SOFA_MISC_ENGINE_API ProjectiveTransformEngine<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_MISC_ENGINE_API ProjectiveTransformEngine<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
template class SOFA_MISC_ENGINE_API ProjectiveTransformEngine<defaulttype::ExtVec3fTypes>;


} // namespace constraint

} // namespace component

} // namespace sofa

