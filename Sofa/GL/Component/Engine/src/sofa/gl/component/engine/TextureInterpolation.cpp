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
#define SOFA_COMPONENT_ENGINE_TEXTUREINTERPOLATION_CPP
#include <sofa/gl/component/engine/TextureInterpolation.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::gl::component::engine
{

using namespace sofa::defaulttype;

void registerTextureInterpolation(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Create texture coordinate for a given field.")
        .add< TextureInterpolation <Vec1Types> >()
        .add< TextureInterpolation <Vec2Types> >()
        .add< TextureInterpolation <Vec3Types> >());
}

template class SOFA_GL_COMPONENT_ENGINE_API TextureInterpolation <Vec1Types>;
template class SOFA_GL_COMPONENT_ENGINE_API TextureInterpolation <Vec2Types>;
template class SOFA_GL_COMPONENT_ENGINE_API TextureInterpolation <Vec3Types>;
 


} //namespace sofa::gl::component::engine
