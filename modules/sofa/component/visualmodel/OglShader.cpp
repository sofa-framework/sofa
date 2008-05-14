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

namespace visualmodel
{


SOFA_DECL_CLASS(OglShader)

//Register OglShader in the Object Factory
int OglShaderClass = core::RegisterObject("OglShader")
        .add< OglShader >()
        ;

OglShader::OglShader():
    vertFilename(initData(&vertFilename, (std::string) "toonShading.vert", "vertFilename", "Set the vertex shader filename to load")),
    fragFilename(initData(&fragFilename, (std::string) "toonShading.frag", "fragFilename", "Set the fragment shader filename to load")),
    geoFilename(initData(&geoFilename, (std::string) "", "geoFilename", "Set the geometry shader filename to load")),
    hasGeometryShader(false)
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

void OglShader::initVisual()
{
    if (sofa::helper::gl::CShader::InitGLSL())
        std::cout << "GLSL OK" << std::endl;
    else    std::cout << "init GLSL failed" << std::endl;

    std::string file = std::string("shaders/") + vertFilename.getValue();

    if (!helper::system::DataRepository.findFile(file))
    {
        std::cerr << "OglShader : vertex shader file not found." << std::endl;
        return;
    }

    file = std::string("shaders/") + fragFilename.getValue();
    if (!helper::system::DataRepository.findFile(file))
    {
        std::cerr << "OglShader : fragment shader file not found." << std::endl;
        return;
    }

    file = std::string("shaders/") + geoFilename.getValue();
    if (geoFilename.getValue() == "" || !helper::system::DataRepository.findFile(file))
        m_shader.InitShaders(helper::system::DataRepository.getFile("shaders/" + vertFilename.getValue()),
                helper::system::DataRepository.getFile("shaders/" + fragFilename.getValue()));

    else
    {
        m_shader.InitShaders(helper::system::DataRepository.getFile("shaders/" + vertFilename.getValue()),
                helper::system::DataRepository.getFile("shaders/" + geoFilename.getValue()),
                helper::system::DataRepository.getFile("shaders/" + fragFilename.getValue()));

        hasGeometryShader = true;
    }

}

void OglShader::drawVisual()
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

void OglShader::updateVisual()
{

}

void OglShader::setTexture(const char* name, unsigned short unit)
{
    start();
    m_shader.SetInt(m_shader.GetVariable(name), unit);
    stop();
}
void OglShader::setInt(const char* name, unsigned int i)
{
    start();
    m_shader.SetInt(m_shader.GetVariable(name), i);
    stop();
}
void OglShader::setFloat(const char* name, float f1)
{
    start();
    m_shader.SetFloat(m_shader.GetVariable(name), f1);
    stop();
}
void OglShader::setFloat2(const char* name, float f1, float f2)
{
    start();
    m_shader.SetFloat2(m_shader.GetVariable(name), f1, f2);
    stop();
}
void OglShader::setFloat3(const char* name, float f1, float f2, float f3)
{
    start();
    m_shader.SetFloat3(m_shader.GetVariable(name), f1, f2, f3);
    stop();
}
void OglShader::setFloat4(const char* name, float f1, float f2, float f3, float f4)
{
    start();
    m_shader.SetFloat4(m_shader.GetVariable(name), f1, f2, f3, f4);
    stop();
}

GLint OglShader::getGeometryInputType()
{
    return m_shader.GetGeometryInputType();
}
void  OglShader::setGeometryInputType(GLint v)
{
    m_shader.SetGeometryInputType(v);
}

GLint OglShader::getGeometryOutputType()
{
    return m_shader.GetGeometryOutputType();
}
void  OglShader::setGeometryOutputType(GLint v)
{
    m_shader.SetGeometryOutputType(v);
}

GLint OglShader::getGeometryVerticesOut()
{
    return m_shader.GetGeometryVerticesOut();
}

void  OglShader::setGeometryVerticesOut(GLint v)
{
    m_shader.SetGeometryVerticesOut(v);
}

OglShaderElement::OglShaderElement()
    : id(initData(&id, (std::string) "id", "id", "Set an ID name"))
{

}

void OglShaderElement::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    shader = context->core::objectmodel::BaseContext::get<OglShader>();

    if (!shader)
    {
        std::cerr << "OglTexture: shader not found "<< std::endl;
        return;
    }
}

}//namespace visualmodel

} //namespace component

} //namespace sofa
