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
#ifndef SOFA_COMPONENT_OGLOITSHADER
#define SOFA_COMPONENT_OGLOITSHADER
#include "config.h"

#include <SofaOpenglVisual/OglShader.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class SOFA_OPENGL_VISUAL_API OglOITShader : public sofa::component::visualmodel::OglShader
{
public:
    SOFA_CLASS(OglOITShader, sofa::component::visualmodel::OglShader);
protected:
    OglOITShader();
    ~OglOITShader() override;

public:
    helper::gl::GLSLShader* accumulationShader();

public:
    static const std::string PATH_TO_OIT_ACCUMULATION_VERTEX_SHADERS;
    static const std::string PATH_TO_OIT_ACCUMULATION_FRAGMENT_SHADERS;

};

}//namespace visualmodel

} //namespace component

} //namespace sofa

#endif //SOFA_COMPONENT_OGLOITSHADER
