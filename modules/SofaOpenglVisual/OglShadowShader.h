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
//
// C++ Interface: Shader
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_COMPONENT_OGLSHADOWSHADER
#define SOFA_COMPONENT_OGLSHADOWSHADER
#include "config.h"

#include <SofaOpenglVisual/OglShader.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class SOFA_OPENGL_VISUAL_API OglShadowShader : public sofa::component::visualmodel::OglShader
{
public:
    SOFA_CLASS(OglShadowShader, sofa::component::visualmodel::OglShader);
protected:
    OglShadowShader();
    virtual ~OglShadowShader();
public:
    void init();

    virtual void initShaders(unsigned int numberOfLights, bool softShadow);

protected:
    static const std::string PATH_TO_SHADOW_VERTEX_SHADERS;
    static const std::string PATH_TO_SHADOW_FRAGMENT_SHADERS;
    static const std::string PATH_TO_SOFT_SHADOW_VERTEX_SHADERS;
    static const std::string PATH_TO_SOFT_SHADOW_FRAGMENT_SHADERS;

};

}//namespace visualmodel

} //namespace component

} //namespace sofa

#endif //SOFA_COMPONENT_OGLSHADOWSHADER
