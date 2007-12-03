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
#include <sofa/component/visualmodel/OglShader.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/FileRepository.h>


namespace sofa
{

namespace component
{

SOFA_DECL_CLASS(OglShader)

//Register DirectionalLight in the Object Factory
int OglShaderClass = core::RegisterObject("OglShader")
        .add< OglShader >()
        ;

OglShader::OglShader():
    vertFilename(initData(&vertFilename, (std::string) "toonShading.vert", "vert", "Set the vertex shader filename to load")),
    fragFilename(initData(&fragFilename, (std::string) "toonShading.frag", "frag", "Set the fragment shader filename to load"))
{


}

OglShader::~OglShader()
{
    m_shader.TurnOff();
    m_shader.Release();
    std::cout << "Shader released." << std::endl;
}


void OglShader::init()
{

}

void OglShader::initTextures()
{
    if (sofa::helper::gl::CShader::InitGLSL())
        std::cout << "GLSL OK" << std::endl;
    else    std::cout << "init GLSL failed" << std::endl;

    m_shader.InitShaders(sofa::helper::system::DataRepository.getFile("shaders/" + vertFilename.getValue()), sofa::helper::system::DataRepository.getFile("shaders/" + fragFilename.getValue()));

}

void OglShader::draw()
{

}

void OglShader::stop()
{
    m_shader.TurnOff();
}

void OglShader::start()
{
    m_shader.TurnOn();
}

void OglShader::update()
{

}

} //namespace component

} //namespace sofa
