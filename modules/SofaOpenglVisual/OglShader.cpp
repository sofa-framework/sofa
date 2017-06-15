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
#include <SofaOpenglVisual/OglShader.h>
#include <SofaOpenglVisual/CompositingVisualLoop.h>
#include <sofa/core/visual/VisualParams.h>
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
    passive(initData(&passive, (bool) false, "passive", "Will this shader be activated manually or automatically?")),
    vertFilename(initData(&vertFilename, helper::SVector<std::string>(1,"shaders/toonShading.vert"), "fileVertexShaders", "Set the vertex shader filename to load")),
    fragFilename(initData(&fragFilename, helper::SVector<std::string>(1,"shaders/toonShading.frag"), "fileFragmentShaders", "Set the fragment shader filename to load")),
#ifdef GL_GEOMETRY_SHADER_EXT
    geoFilename(initData(&geoFilename, "fileGeometryShaders", "Set the geometry shader filename to load")),
#endif
#ifdef GL_TESS_CONTROL_SHADER
    tessellationControlFilename(initData(&tessellationControlFilename, "fileTessellationControlShaders", "Set the tessellation control filename to load")),
#endif
#ifdef GL_TESS_EVALUATION_SHADER
    tessellationEvaluationFilename(initData(&tessellationEvaluationFilename, "fileTessellationEvaluationShaders", "Set the tessellation evaluation filename to load")),
#endif
#ifdef GL_GEOMETRY_SHADER_EXT
    geometryInputType(initData(&geometryInputType, (int) -1, "geometryInputType", "Set input types for the geometry shader")),
    geometryOutputType(initData(&geometryOutputType, (int) -1, "geometryOutputType", "Set output types for the geometry shader")),
    geometryVerticesOut(initData(&geometryVerticesOut, (int) -1, "geometryVerticesOut", "Set max number of vertices in output for the geometry shader")),
#endif
#ifdef GL_TESS_CONTROL_SHADER
    tessellationOuterLevel(initData(&tessellationOuterLevel,(GLfloat)1, "tessellationOuterLevel", "For tessellation without control shader: default outer level (edge subdivisions)")),
    tessellationInnerLevel(initData(&tessellationInnerLevel,(GLfloat)1, "tessellationInnerLevel", "For tessellation without control shader: default inner level (face subdivisions)")),
#endif

    indexActiveShader(initData(&indexActiveShader, (unsigned int) 0, "indexActiveShader", "Set current active shader")),
    backfaceWriting( initData(&backfaceWriting, (bool) false, "backfaceWriting", "it enables writing to gl_BackColor inside a GLSL vertex shader" ) ),
    clampVertexColor( initData(&clampVertexColor, (bool) true, "clampVertexColor", "clamp the vertex color between 0 and 1" ) )
{
#ifdef GL_TESS_CONTROL_SHADER
    addAlias(&tessellationOuterLevel,"tessellationLevel");
    addAlias(&tessellationInnerLevel,"tessellationLevel");
#endif
}

OglShader::~OglShader()
{
    if (!shaderVector.empty())
    {
        shaderVector[indexActiveShader.getValue()]->TurnOff();
        for (unsigned int i=0 ; i<shaderVector.size() ; i++)
        {
            if (shaderVector[i])
            {
                shaderVector[i]->Release();
                delete shaderVector[i];
            }
        }
    }
}

void OglShader::init()
{
    unsigned int nshaders = 0;
    nshaders = std::max(nshaders, (unsigned int)vertFilename.getValue().size());
    nshaders = std::max(nshaders, (unsigned int)fragFilename.getValue().size());
#ifdef GL_GEOMETRY_SHADER_EXT
    nshaders = std::max(nshaders, (unsigned int)geoFilename.getValue().size());
#endif
#ifdef GL_TESS_CONTROL_SHADER
    nshaders = std::max(nshaders, (unsigned int)tessellationControlFilename.getValue().size());
#endif
#ifdef GL_TESS_EVALUATION_SHADER
    nshaders = std::max(nshaders, (unsigned int)tessellationEvaluationFilename.getValue().size());
#endif

    sout << nshaders << " shader version(s)" << sendl;

    shaderVector.resize(nshaders);

    for (unsigned int i=0 ; i<nshaders ; i++)
    {
        shaderVector[i] = new sofa::helper::gl::GLSLShader();
        if (!vertFilename.getValue().empty())
        {
            shaderVector[i]->SetVertexShaderFileName(vertFilename.getFullPath(std::min(i,(unsigned int)vertFilename.getValue().size()-1)));
        }
        if (!fragFilename.getValue().empty())
        {
            shaderVector[i]->SetFragmentShaderFileName(fragFilename.getFullPath(std::min(i,(unsigned int)fragFilename.getValue().size()-1)));
        }
#ifdef GL_GEOMETRY_SHADER_EXT
        if (!geoFilename.getValue().empty())
        {
            shaderVector[i]->SetGeometryShaderFileName(geoFilename.getFullPath(std::min(i,(unsigned int)geoFilename.getValue().size()-1)));
        }
#endif
#ifdef GL_TESS_CONTROL_SHADER
        if (!tessellationControlFilename.getValue().empty())
        {
            shaderVector[i]->SetTessellationControlShaderFileName(tessellationControlFilename.getFullPath(std::min(i,(unsigned int)tessellationControlFilename.getValue().size()-1)));
        }
#endif
#ifdef GL_TESS_EVALUATION_SHADER
        if (!tessellationEvaluationFilename.getValue().empty())
        {
            shaderVector[i]->SetTessellationEvaluationShaderFileName(tessellationEvaluationFilename.getFullPath(std::min(i,(unsigned int)tessellationEvaluationFilename.getValue().size()-1)));
        }
#endif
    }
}

void OglShader::reinit()
{

}

void OglShader::initVisual()
{

    if (!sofa::helper::gl::GLSLShader::InitGLSL())
    {
        serr << "InitGLSL failed" << sendl;
        return;
    }
    unsigned int nshaders = (unsigned int)shaderVector.size();
#ifdef GL_GEOMETRY_SHADER_EXT
    if (!geoFilename.getValue().empty())
    {
        if (geometryInputType.getValue() != -1)
        {
            for (unsigned int i=0 ; i<nshaders ; i++)
                setGeometryInputType(i, geometryInputType.getValue());
        }
        if (geometryOutputType.getValue() != -1)
        {
            for (unsigned int i=0 ; i<nshaders ; i++)
                setGeometryOutputType(i, geometryOutputType.getValue());
        }
        GLint maxV;
        glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT, &maxV);
        if (geometryVerticesOut.getValue() < 0 || geometryVerticesOut.getValue() > maxV)
        {
            geometryVerticesOut.setValue(maxV);
        }
        if (geometryVerticesOut.getValue() >= 0)
        {
            for (unsigned int i=0 ; i<nshaders ; i++)
                setGeometryVerticesOut(i, geometryVerticesOut.getValue());
        }
    }
#endif
    for (unsigned int i=0 ; i<nshaders ; i++)
    {
        shaderVector[i]->InitShaders();
    }
}



void OglShader::parse(core::objectmodel::BaseObjectDescription* arg)
{
    Inherit1::parse(arg);


    // BACKWARD COMPATIBILITY oct 2016
    const char* fileVertexShader = arg->getAttribute("fileVertexShader");
    const char* fileVertexShaderAlias = arg->getAttribute("vertFilename");
    if( fileVertexShader || fileVertexShaderAlias )
    {
        serr<<helper::logging::Message::Deprecated<<"parse: You are using a deprecated Data<vector<string>> 'fileVertexShader' or 'vertFilename', please use the new Data<SVector<string>>'fileVertexShaders'"<<sendl;
        helper::vector<std::string> simplevector;
        std::istringstream( fileVertexShader ? fileVertexShader : fileVertexShaderAlias ) >> simplevector;
        vertFilename.setValue( simplevector );
    }
    const char* fileFragmentShader = arg->getAttribute("fileFragmentShader");
    const char* fileFragmentShaderAlias = arg->getAttribute("fragFilename");
    if( fileFragmentShader || fileFragmentShaderAlias )
    {
        serr<<helper::logging::Message::Deprecated<<"parse: You are using a deprecated Data<vector<string>> 'fileFragmentShader' or 'fragFilename', please use the new Data<SVector<string>>'fileFragmentShaders'"<<sendl;
        helper::vector<std::string> simplevector;
        std::istringstream( fileFragmentShader ? fileFragmentShader : fileFragmentShaderAlias ) >> simplevector;
        fragFilename.setValue( simplevector );
    }
#ifdef GL_GEOMETRY_SHADER_EXT
    const char* fileGeometryShader = arg->getAttribute("fileGeometryShader");
    const char* fileGeometryShaderAlias = arg->getAttribute("geoFilename");
    if( fileGeometryShader || fileGeometryShaderAlias )
    {
        serr<<helper::logging::Message::Deprecated<<"parse: You are using a deprecated Data<vector<string>> 'fileGeometryShader' or 'geoFilename', please use the new Data<SVector<string>>'fileGeometryShaders'"<<sendl;
        helper::vector<std::string> simplevector;
        std::istringstream( fileGeometryShader ? fileGeometryShader : fileGeometryShaderAlias ) >> simplevector;
        geoFilename.setValue( simplevector );
    }
#endif
#ifdef GL_TESS_CONTROL_SHADER
    const char* fileTessellationControlShader = arg->getAttribute("fileTessellationControlShader");
    if( fileTessellationControlShader )
    {
        serr<<helper::logging::Message::Deprecated<<"parse: You are using a deprecated Data<vector<string>> 'fileTessellationControlShader', please use the new Data<SVector<string>>'fileTessellationControlShaders'"<<sendl;
        helper::vector<std::string> simplevector;
        std::istringstream( fileTessellationControlShader ) >> simplevector;
        tessellationControlFilename.setValue( simplevector );
    }
#endif
#ifdef GL_TESS_EVALUATION_SHADER
    const char* fileTessellationEvaluationShader = arg->getAttribute("fileTessellationEvaluationShader");
    if( fileTessellationEvaluationShader )
    {
        serr<<helper::logging::Message::Deprecated<<"parse: You are using a deprecated Data<vector<string>> 'fileTessellationEvaluationShader', please use the new Data<SVector<string>>'fileTessellationEvaluationShaders'"<<sendl;
        helper::vector<std::string> simplevector;
        std::istringstream( fileTessellationEvaluationShader ) >> simplevector;
        tessellationEvaluationFilename.setValue( simplevector );
    }
#endif
}

void OglShader::drawVisual(const core::visual::VisualParams* )
{

}

void OglShader::stop()
{
    if(turnOn.getValue() && shaderVector[indexActiveShader.getValue()]->IsReady())
    {
        if ( backfaceWriting.getValue() )
            glDisable(GL_VERTEX_PROGRAM_TWO_SIDE);
        if ( !clampVertexColor.getValue() )
            glClampColorARB(GL_CLAMP_VERTEX_COLOR, GL_TRUE);
        shaderVector[indexActiveShader.getValue()]->TurnOff();
    }
}

void OglShader::start()
{
    if(turnOn.getValue() && shaderVector[indexActiveShader.getValue()]->IsReady())
    {
        shaderVector[indexActiveShader.getValue()]->TurnOn();

        if ( !clampVertexColor.getValue() )
            glClampColorARB(GL_CLAMP_VERTEX_COLOR, GL_FALSE);
        if ( backfaceWriting.getValue() )
            glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
#if defined(SOFA_HAVE_GLEW) && defined(GL_TESS_CONTROL_SHADER)
        if (shaderVector[indexActiveShader.getValue()]->GetTessellationEvaluationShaderID() && !shaderVector[indexActiveShader.getValue()]->GetTessellationControlShaderID() && GLEW_ARB_tessellation_shader)
        {
            helper::fixed_array<GLfloat,4> levels;
            levels.assign(tessellationOuterLevel.getValue());
            glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, levels.data());
            levels.assign(tessellationInnerLevel.getValue());
            glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, levels.data());
        }
#endif
    }
}

bool OglShader::isActive()
{
    return !passive.getValue();
}

void OglShader::updateVisual()
{

}

unsigned int OglShader::getNumberOfShaders()
{
    return (unsigned int)shaderVector.size();
}

unsigned int OglShader::getCurrentIndex()
{
    return indexActiveShader.getValue();
}

void OglShader::setCurrentIndex(const unsigned int index)
{
    if (index < shaderVector.size())
    {
        shaderVector[indexActiveShader.getValue()]->TurnOff();
        indexActiveShader.setValue(index);
        shaderVector[indexActiveShader.getValue()]->TurnOn();
    }
}

void OglShader::addDefineMacro(const unsigned int index, const std::string &name, const std::string &value)
{
    shaderVector[index]->AddDefineMacro(name, value);
}

void OglShader::setTexture(const unsigned int index, const char* name, unsigned short unit)
{
    start();
    shaderVector[index]->SetInt(shaderVector[index]->GetVariable(name), unit);
    stop();
}
void OglShader::setInt(const unsigned int index, const char* name, int i)
{
    start();
    shaderVector[index]->SetInt(shaderVector[index]->GetVariable(name), i);
    stop();
}

void OglShader::setInt2(const unsigned int index, const char* name, int i1, int i2)
{
    start();
    shaderVector[index]->SetInt2(shaderVector[index]->GetVariable(name), i1, i2);
    stop();
}
void OglShader::setInt3(const unsigned int index, const char* name, int i1, int i2, int i3)
{
    start();
    shaderVector[index]->SetInt3(shaderVector[index]->GetVariable(name), i1, i2, i3);
    stop();
}
void OglShader::setInt4(const unsigned int index, const char* name, int i1, int i2, int i3, int i4)
{
    start();
    shaderVector[index]->SetInt4(shaderVector[index]->GetVariable(name), i1, i2, i3, i4);
    stop();
}

void OglShader::setFloat(const unsigned int index, const char* name, float f1)
{
    start();
    shaderVector[index]->SetFloat(shaderVector[index]->GetVariable(name), f1);
    stop();
}
void OglShader::setFloat2(const unsigned int index, const char* name, float f1, float f2)
{
    start();
    shaderVector[index]->SetFloat2(shaderVector[index]->GetVariable(name), f1, f2);
    stop();
}
void OglShader::setFloat3(const unsigned int index, const char* name, float f1, float f2, float f3)
{
    start();
    shaderVector[index]->SetFloat3(shaderVector[index]->GetVariable(name), f1, f2, f3);
    stop();
}
void OglShader::setFloat4(const unsigned int index, const char* name, float f1, float f2, float f3, float f4)
{
    start();
    shaderVector[index]->SetFloat4(shaderVector[index]->GetVariable(name), f1, f2, f3, f4);
    stop();
}

void OglShader::setIntVector(const unsigned int index, const char* name, int count, const GLint* i)
{
    start();
    shaderVector[index]->SetIntVector(shaderVector[index]->GetVariable(name), count, i);
    stop();
}
void OglShader::setIntVector2(const unsigned int index, const char* name, int count, const GLint* i)
{
    start();
    shaderVector[index]->SetIntVector2(shaderVector[index]->GetVariable(name), count, i);
    stop();
}
void OglShader::setIntVector3(const unsigned int index, const char* name, int count, const GLint* i)
{
    start();
    shaderVector[index]->SetIntVector3(shaderVector[index]->GetVariable(name), count, i);
    stop();
}
void OglShader::setIntVector4(const unsigned int index, const char* name, int count, const GLint* i)
{
    start();
    shaderVector[index]->SetIntVector4(shaderVector[index]->GetVariable(name), count, i);
    stop();
}

void OglShader::setFloatVector(const unsigned int index, const char* name, int count, const float* f)
{
    start();
    shaderVector[index]->SetFloatVector(shaderVector[index]->GetVariable(name), count, f);
    stop();
}
void OglShader::setFloatVector2(const unsigned int index, const char* name, int count, const float* f)
{
    start();
    shaderVector[index]->SetFloatVector2(shaderVector[index]->GetVariable(name), count, f);
    stop();
}
void OglShader::setFloatVector3(const unsigned int index, const char* name, int count, const float* f)
{
    start();
    shaderVector[index]->SetFloatVector3(shaderVector[index]->GetVariable(name), count, f);
    stop();
}
void OglShader::setFloatVector4(const unsigned int index, const char* name, int count, const float* f)
{
    start();
    shaderVector[index]->SetFloatVector4(shaderVector[index]->GetVariable(name), count, f);
    stop();
}

void OglShader::setMatrix2(const unsigned int index, const char* name, int count, bool transpose, const float* f)
{
    start();
    shaderVector[index]->SetMatrix2(shaderVector[index]->GetVariable(name), count, transpose, f);
    stop();
}

void OglShader::setMatrix3(const unsigned int index, const char* name, int count, bool transpose, const float* f)
{
    start();
    shaderVector[index]->SetMatrix3(shaderVector[index]->GetVariable(name), count, transpose, f);
    stop();
}

void OglShader::setMatrix4(const unsigned int index, const char* name, int count, bool transpose, const float* f)
{
    start();
    shaderVector[index]->SetMatrix4(shaderVector[index]->GetVariable(name), count, transpose, f);
    stop();
}

void OglShader::setMatrix2x3(const unsigned int index, const char* name, int count, bool transpose, const float* f)
{
    start();
    shaderVector[index]->SetMatrix2x3(shaderVector[index]->GetVariable(name), count, transpose, f);
    stop();
}

void OglShader::setMatrix3x2(const unsigned int index, const char* name, int count, bool transpose, const float* f)
{
    start();
    shaderVector[index]->SetMatrix3x2(shaderVector[index]->GetVariable(name), count, transpose, f);
    stop();
}

void OglShader::setMatrix2x4(const unsigned int index, const char* name, int count, bool transpose, const float* f)
{
    start();
    shaderVector[index]->SetMatrix2x4(shaderVector[index]->GetVariable(name), count, transpose, f);
    stop();
}

void OglShader::setMatrix4x2(const unsigned int index, const char* name, int count, bool transpose, const float* f)
{
    start();
    shaderVector[index]->SetMatrix4x2(shaderVector[index]->GetVariable(name), count, transpose, f);
    stop();
}

void OglShader::setMatrix3x4(const unsigned int index, const char* name, int count, bool transpose, const float* f)
{
    start();
    shaderVector[index]->SetMatrix3x4(shaderVector[index]->GetVariable(name), count, transpose, f);
    stop();
}

void OglShader::setMatrix4x3(const unsigned int index, const char* name, int count, bool transpose, const float* f)
{
    start();
    shaderVector[index]->SetMatrix4x3(shaderVector[index]->GetVariable(name), count, transpose, f);
    stop();
}

GLint OglShader::getAttribute(const unsigned int index, const char* name)
{
    start();
    GLint res = shaderVector[index]->GetAttributeVariable(name);
    stop();

    return res;
}

GLint OglShader::getUniform(const unsigned int index, const char* name)
{
    start();
    GLint res = shaderVector[index]->GetVariable(name);
    stop();

    return res;
}

#ifdef GL_GEOMETRY_SHADER_EXT
GLint OglShader::getGeometryInputType(const unsigned int index)
{
    return shaderVector[index]->GetGeometryInputType();
}

void  OglShader::setGeometryInputType(const unsigned int index, GLint v)
{
    shaderVector[index]->SetGeometryInputType(v);
}

GLint OglShader::getGeometryOutputType(const unsigned int index)
{
    return shaderVector[index]->GetGeometryOutputType();
}

void  OglShader::setGeometryOutputType(const unsigned int index, GLint v)
{
    shaderVector[index]->SetGeometryOutputType(v);
}

GLint OglShader::getGeometryVerticesOut(const unsigned int index)
{
    return shaderVector[index]->GetGeometryVerticesOut();
}

void  OglShader::setGeometryVerticesOut(const unsigned int index, GLint v)
{
    shaderVector[index]->SetGeometryVerticesOut(v);
}
#endif

OglShaderElement::OglShaderElement()
    : id(initData(&id, std::string(""), "id", "Set an ID name"))
    , indexShader(initData(&indexShader, (unsigned int) 0, "indexShader", "Set the index of the desired shader you want to apply this parameter"))
{

}

void OglShaderElement::init()
{
    sofa::core::objectmodel::BaseContext* mycontext = this->getContext();

    if (id.getValue().empty())
        id.setValue(this->getName());

    /*when no multipass is active */
    sofa::component::visualmodel::CompositingVisualLoop* isMultipass=NULL;
    isMultipass= mycontext->core::objectmodel::BaseContext::get<sofa::component::visualmodel::CompositingVisualLoop>();
    if(isMultipass==NULL)
    {
        if ( OglShader* shader = mycontext->core::objectmodel::BaseContext::get<OglShader>(this->getTags()) )
        {
            shaders.insert( shader );

            msg_info() << this->id.getValue() << " set in " << shader->getName();
        }
        return;
    }

    sofa::core::objectmodel::TagSet::const_iterator begin = this->getTags().begin();
    sofa::core::objectmodel::TagSet::const_iterator end = this->getTags().end();
    sofa::core::objectmodel::TagSet::const_iterator it;
    helper::vector<OglShader*> gotShaders;

    for (it = begin; it != end; ++it)
    {
        mycontext->core::objectmodel::BaseContext::get<OglShader, helper::vector<OglShader*> >(&gotShaders, (*it));
        for(helper::vector<OglShader*>::iterator it2 = gotShaders.begin(); it2!= gotShaders.end(); ++it2) //merge into shaders vector
        {
            shaders.insert(*it2);
            //shaders.push_back(*it2);
        }
    }

    if (shaders.empty())
    {
        serr << this->getTypeName() <<" \"" << this->getName() << "\": no relevant shader found. please check tags validity"<< sendl;
        return;
    }
}


}//namespace visualmodel

} //namespace component

} //namespace sofa
