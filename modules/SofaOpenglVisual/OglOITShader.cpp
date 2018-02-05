/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
// C++ Implementation: Shader
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <SofaOpenglVisual/OglOITShader.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/FileRepository.h>

#include <SofaOpenglVisual/LightManager.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{


SOFA_DECL_CLASS(OglOITShader)

//Register OglShader in the Object Factory
int OglOITShaderClass = core::RegisterObject("OglOITShader")
        .add< OglOITShader >()
        ;

const std::string OglOITShader::PATH_TO_OIT_ACCUMULATION_VERTEX_SHADERS = "shaders/orderIndependentTransparency/accumulation.vert";
const std::string OglOITShader::PATH_TO_OIT_ACCUMULATION_FRAGMENT_SHADERS = "shaders/orderIndependentTransparency/accumulation.frag";

OglOITShader::OglOITShader()
{
    passive.setValue(false);

    helper::vector<std::string>& vertF = *vertFilename.beginEdit();
    vertF.resize(1);
    vertF[0] = PATH_TO_OIT_ACCUMULATION_VERTEX_SHADERS;
    vertFilename.endEdit();
    helper::vector<std::string>& fragF = *fragFilename.beginEdit();
    fragF.resize(1);
    fragF[0] = PATH_TO_OIT_ACCUMULATION_FRAGMENT_SHADERS;
    fragFilename.endEdit();
}

OglOITShader::~OglOITShader()
{

}

helper::gl::GLSLShader* OglOITShader::accumulationShader()
{
    if(shaderVector.size() < 1)
        return 0;

    return shaderVector[0];
}

}//namespace visualmodel

} //namespace component

} //namespace sofa
