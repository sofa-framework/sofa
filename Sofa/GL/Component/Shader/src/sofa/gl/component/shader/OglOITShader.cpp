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
#include <sofa/gl/component/shader/OglOITShader.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/FileRepository.h>

#include <sofa/gl/component/shader/LightManager.h>

namespace sofa::gl::component::shader
{

//Register OglShader in the Object Factory
int OglOITShaderClass = core::RegisterObject("OglOITShader")
        .add< OglOITShader >()
        ;

const std::string OglOITShader::PATH_TO_OIT_ACCUMULATION_VERTEX_SHADERS = "shaders/orderIndependentTransparency/accumulation.vert";
const std::string OglOITShader::PATH_TO_OIT_ACCUMULATION_FRAGMENT_SHADERS = "shaders/orderIndependentTransparency/accumulation.frag";

OglOITShader::OglOITShader()
{
    passive.setValue(false);

    type::vector<std::string>& vertF = *vertFilename.beginEdit();
    vertF.resize(1);
    vertF[0] = PATH_TO_OIT_ACCUMULATION_VERTEX_SHADERS;
    vertFilename.endEdit();
    type::vector<std::string>& fragF = *fragFilename.beginEdit();
    fragF.resize(1);
    fragF[0] = PATH_TO_OIT_ACCUMULATION_FRAGMENT_SHADERS;
    fragFilename.endEdit();
}

OglOITShader::~OglOITShader()
{

}

sofa::gl::GLSLShader* OglOITShader::accumulationShader()
{
    if(shaderVector.size() < 1)
        return nullptr;

    return shaderVector[0];
}

} // namespace sofa::gl::component::shader
