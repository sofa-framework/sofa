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
    turnOn(initData(&turnOn, (bool) true, "turnOn", "Turn On the shader?")),
    vertFilename(initData(&vertFilename, (std::string) "toonShading.vert", "vertFilename", "Set the vertex shader filename to load")),
    fragFilename(initData(&fragFilename, (std::string) "toonShading.frag", "fragFilename", "Set the fragment shader filename to load")),
    geoFilename(initData(&geoFilename, (std::string) "", "geoFilename", "Set the geometry shader filename to load")),
    geometryInputType(initData(&geometryInputType, (int) -1, "geometryInputType", "Set input types for the geometry shader")),
    geometryOutputType(initData(&geometryOutputType, (int) -1, "geometryOutputType", "Set output types for the geometry shader")),
    geometryVerticesOut(initData(&geometryVerticesOut, (int) -1, "geometryVerticesOut", "Set max number of vertices in output for the geometry shader")),
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

void OglShader::reinit()
{
    if (hasGeometryShader)
    {

    }
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
        if (geometryInputType.getValue() != -1)
            setGeometryInputType(geometryInputType.getValue());
        if (geometryOutputType.getValue() != -1)
            setGeometryOutputType(geometryOutputType.getValue());
#ifdef GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT
        GLint maxV;
        glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT, &maxV);
        if (geometryVerticesOut.getValue() == -1 || geometryVerticesOut.getValue() > maxV)
            geometryVerticesOut.setValue(3);
#endif
        if (geometryVerticesOut.getValue() != -1)
            setGeometryVerticesOut(geometryVerticesOut.getValue());

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
    if(turnOn.getValue())
        m_shader.TurnOff();
}

void OglShader::start()
{
    if(turnOn.getValue())
        m_shader.TurnOn();
}

void OglShader::updateVisual()
{

}

void OglShader::addDefineMacro(const std::string &name, const std::string &value)
{
    m_shader.AddDefineMacro(name, value);
}

void OglShader::setTexture(const char* name, unsigned short unit)
{
    start();
    m_shader.SetInt(m_shader.GetVariable(name), unit);
    stop();
}
void OglShader::setInt(const char* name, int i)
{
    start();
    m_shader.SetInt(m_shader.GetVariable(name), i);
    stop();
}

void OglShader::setInt2(const char* name, int i1, int i2)
{
    start();
    m_shader.SetInt2(m_shader.GetVariable(name), i1, i2);
    stop();
}
void OglShader::setInt3(const char* name, int i1, int i2, int i3)
{
    start();
    m_shader.SetInt3(m_shader.GetVariable(name), i1, i2, i3);
    stop();
}
void OglShader::setInt4(const char* name, int i1, int i2, int i3, int i4)
{
    start();
    m_shader.SetInt4(m_shader.GetVariable(name), i1, i2, i3, i4);
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

void OglShader::setIntVector(const char* name, int count, const GLint* i)
{
    start();
    m_shader.SetIntVector(m_shader.GetVariable(name), count, i);
    stop();
}
void OglShader::setIntVector2(const char* name, int count, const GLint* i)
{
    start();
    m_shader.SetIntVector2(m_shader.GetVariable(name), count, i);
    stop();
}
void OglShader::setIntVector3(const char* name, int count, const GLint* i)
{
    start();
    m_shader.SetIntVector3(m_shader.GetVariable(name), count, i);
    stop();
}
void OglShader::setIntVector4(const char* name, int count, const GLint* i)
{
    start();
    m_shader.SetIntVector4(m_shader.GetVariable(name), count, i);
    stop();
}

void OglShader::setFloatVector(const char* name, int count, const float* f)
{
    start();
    m_shader.SetFloatVector(m_shader.GetVariable(name), count, f);
    stop();
}
void OglShader::setFloatVector2(const char* name, int count, const float* f)
{
    start();
    m_shader.SetFloatVector2(m_shader.GetVariable(name), count, f);
    stop();
}
void OglShader::setFloatVector3(const char* name, int count, const float* f)
{
    start();
    m_shader.SetFloatVector3(m_shader.GetVariable(name), count, f);
    stop();
}
void OglShader::setFloatVector4(const char* name, int count, const float* f)
{
    start();
    m_shader.SetFloatVector4(m_shader.GetVariable(name), count, f);
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
        std::cerr << "OglShaderElement: shader not found "<< std::endl;
        return;
    }
}


}//namespace visualmodel

} //namespace component

} //namespace sofa
