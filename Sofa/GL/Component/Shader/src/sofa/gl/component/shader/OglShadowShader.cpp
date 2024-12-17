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
#include <sofa/gl/component/shader/OglShadowShader.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/FileRepository.h>

#include <sofa/gl/component/shader/LightManager.h>

namespace sofa::gl::component::shader
{

void registerOglShadowShader(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("This component sets the shader system responsible of the shadowing.")
        .add< OglShadowShader >());
}

const std::string OglShadowShader::PATH_TO_SHADOW_VERTEX_SHADERS = "shaders/hardShadows/shadowMapping.vert";
const std::string OglShadowShader::PATH_TO_SHADOW_FRAGMENT_SHADERS = "shaders/hardShadows/shadowMapping.frag";
const std::string OglShadowShader::PATH_TO_SOFT_SHADOW_VERTEX_SHADERS = "shaders/softShadows/VSM/variance_shadow_mapping.vert";
const std::string OglShadowShader::PATH_TO_SOFT_SHADOW_FRAGMENT_SHADERS = "shaders/softShadows/VSM/variance_shadow_mapping.frag";

OglShadowShader::OglShadowShader()
{
    passive.setValue(false);
    turnOn.setValue(true);
    type::vector<std::string>& vertF = *vertFilename.beginEdit();
    vertF.resize(2);
    vertF[0] = PATH_TO_SHADOW_VERTEX_SHADERS;
    vertF[1] = PATH_TO_SHADOW_VERTEX_SHADERS;
    vertFilename.endEdit();
    type::vector<std::string>& fragF = *fragFilename.beginEdit();
    fragF.resize(2);
    fragF[0] = PATH_TO_SHADOW_FRAGMENT_SHADERS;
    fragF[1] = PATH_TO_SHADOW_FRAGMENT_SHADERS;
    fragFilename.endEdit();
}

OglShadowShader::~OglShadowShader()
{
}

void OglShadowShader::init()
{
    OglShader::init();

    std::ostringstream oss;
    oss << LightManager::MAX_NUMBER_OF_LIGHTS;

    addDefineMacro(0,std::string("MAX_NUMBER_OF_LIGHTS"), oss.str());
    addDefineMacro(0,std::string("ENABLE_SHADOW"), "0");
    addDefineMacro(1,std::string("MAX_NUMBER_OF_LIGHTS"), oss.str());
    addDefineMacro(1,std::string("ENABLE_SHADOW"), "1");
}

void OglShadowShader::initShaders(unsigned int /* numberOfLights */, bool softShadow)
{
    type::vector<std::string>& vertF = *vertFilename.beginEdit();
    vertF.resize(2);
    vertF[0] = (softShadow ? PATH_TO_SOFT_SHADOW_VERTEX_SHADERS : PATH_TO_SHADOW_VERTEX_SHADERS);
    vertF[1] = (softShadow ? PATH_TO_SOFT_SHADOW_VERTEX_SHADERS : PATH_TO_SHADOW_VERTEX_SHADERS);
    vertFilename.endEdit();
    type::vector<std::string>& fragF = *fragFilename.beginEdit();
    fragF.resize(2);
    fragF[0] = (softShadow ? PATH_TO_SOFT_SHADOW_FRAGMENT_SHADERS : PATH_TO_SHADOW_FRAGMENT_SHADERS);
    fragF[1] = (softShadow ? PATH_TO_SOFT_SHADOW_FRAGMENT_SHADERS : PATH_TO_SHADOW_FRAGMENT_SHADERS);
    fragFilename.endEdit();
}

} // namespace sofa::gl::component::shader
