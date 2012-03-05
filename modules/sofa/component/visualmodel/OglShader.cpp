/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
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
#include <sofa/component/visualmodel/OglShader.h>
#include <sofa/component/visualmodel/CompositingVisualLoop.h>
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
    vertFilename(initData(&vertFilename, (std::string) "shaders/toonShading.vert", "vertFilename", "Set the vertex shader filename to load")),
    fragFilename(initData(&fragFilename, (std::string) "shaders/toonShading.frag", "fragFilename", "Set the fragment shader filename to load")),
    geoFilename(initData(&geoFilename, (std::string) "", "geoFilename", "Set the geometry shader filename to load")),
    geometryInputType(initData(&geometryInputType, (int) -1, "geometryInputType", "Set input types for the geometry shader")),
    geometryOutputType(initData(&geometryOutputType, (int) -1, "geometryOutputType", "Set output types for the geometry shader")),
    geometryVerticesOut(initData(&geometryVerticesOut, (int) -1, "geometryVerticesOut", "Set max number of vertices in output for the geometry shader")),
    indexActiveShader(initData(&indexActiveShader, (unsigned int) 0, "indexActiveShader", "Set current active shader")),
    backfaceWriting( initData(&backfaceWriting, (bool) false, "backfaceWriting", "it enables writing to gl_BackColor inside a GLSL vertex shader" ) ),
    clampVertexColor( initData(&clampVertexColor, (bool) true, "clampVertexColor", "clamp the vertex color between 0 and 1" ) ),
    hasGeometryShader(false)
{


}

OglShader::~OglShader()
{
    if (shaderVector.size() == 0) return;
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

void OglShader::init()
{
    ///Vertex filenames parsing
    std::string tempStr = vertFilename.getFullPath();
    std::string file;
    const std::string SEPARATOR = ";";
    size_t pos = 0;
    size_t oldPos = 0;

    pos = tempStr.find(SEPARATOR, oldPos);

    while (pos != std::string::npos)
    {
        file = tempStr.substr( oldPos, pos - oldPos);

        if (!helper::system::DataRepository.findFile(file))
        {
            serr << "OglShader : vertex shader file " << file <<" was not found." << sendl;
            return;
        }
        vertexFilenames.push_back( file );

        oldPos = pos + SEPARATOR.size();
        pos = tempStr.find(SEPARATOR, oldPos + SEPARATOR.size());
    }

    file = tempStr.substr( oldPos );

    if (!helper::system::DataRepository.findFile(file))
    {
        serr << "OglShader : vertex shader file " << file <<" was not found." << sendl;
        return;
    }
    vertexFilenames.push_back( file );

    ///Fragment filenames parsing
    pos = oldPos = 0;
    tempStr = fragFilename.getFullPath();

    pos = tempStr.find(SEPARATOR, oldPos);

    while (pos != std::string::npos)
    {
        file = tempStr.substr( oldPos, pos - oldPos );

        if (!helper::system::DataRepository.findFile(file))
        {
            serr << "OglShader : fragment shader file " << file <<" was not found." << sendl;
            return;
        }
        fragmentFilenames.push_back( file );

        oldPos = pos + SEPARATOR.size();
        pos = tempStr.find(SEPARATOR, oldPos + SEPARATOR.size());
    }

    file = tempStr.substr( oldPos );
    if (!helper::system::DataRepository.findFile(file))
    {
        serr << "OglShader : fragment shader file " << file <<" was not found." << sendl;
        return;
    }
    fragmentFilenames.push_back( file );


    ///Geometry filenames parsing
    pos = oldPos = 0;
    tempStr = geoFilename.getFullPath();

    if (geoFilename.getValue() == "" )
    {
        //shaderVector[i]->InitShaders(helper::system::DataRepository.getFile(vertFilename.getValue()),
        //		             helper::system::DataRepository.getFile(fragFilename.getValue()));

        if (fragmentFilenames.size() != vertexFilenames.size())
        {
            serr << "OglShader : The number of Vertex shaders is different from the number of Fragment Shaders." << sendl;
            return;
        }

        shaderVector.resize(vertexFilenames.size());

        for (unsigned int i=0 ; i<vertexFilenames.size() ; i++)
        {
            shaderVector[i] = new sofa::helper::gl::GLSLShader();
        }

    }
    else
    {
        pos = tempStr.find(SEPARATOR, oldPos);

        while (pos != std::string::npos)
        {
            pos = tempStr.find(SEPARATOR, oldPos);
            if (pos != std::string::npos)
            {
                file = tempStr.substr( oldPos, pos - oldPos );

                if (!helper::system::DataRepository.findFile(file))
                {
                    serr << "OglShader : geometry shader file " << file <<" was not found." << sendl;
                    return;
                }
                geometryFilenames.push_back( file );
            }
            oldPos = pos + SEPARATOR.size();
            pos = tempStr.find(SEPARATOR, oldPos + SEPARATOR.size());
        }

        file = tempStr.substr( oldPos );
        if (!helper::system::DataRepository.findFile(file))
        {
            serr << "OglShader : geometry shader file " << file <<" was not found." << sendl;
            return;
        }
        geometryFilenames.push_back( file );


        if (fragmentFilenames.size() != vertexFilenames.size() && geometryFilenames.size() !=  vertexFilenames.size())
        {
            serr << "OglShader : The number of indicated shaders is not coherent (not the same number for each triplet." << sendl;
            return;
        }

        shaderVector.resize(vertexFilenames.size());
        for (unsigned int i=0 ; i<vertexFilenames.size() ; i++)
            shaderVector[i] = new sofa::helper::gl::GLSLShader();

        hasGeometryShader = true;
    }

}

void OglShader::reinit()
{

}

void OglShader::initVisual()
{

    if (!sofa::helper::gl::GLSLShader::InitGLSL())
    {
        serr << "OglShader : InitGLSL failed" << sendl;
        return;
    }

    if (!hasGeometryShader)
    {
        for (unsigned int i=0 ; i<shaderVector.size() ; i++)
        {
            shaderVector[i]->InitShaders(helper::system::DataRepository.getFile(vertexFilenames[i]),
                    helper::system::DataRepository.getFile(fragmentFilenames[i]));
        }
    }
    else
    {
        if (geometryInputType.getValue() != -1)
        {
            for (unsigned int i=0 ; i<vertexFilenames.size() ; i++)
                setGeometryInputType(i, geometryInputType.getValue());
        }
        if (geometryOutputType.getValue() != -1)
        {
            for (unsigned int i=0 ; i<vertexFilenames.size() ; i++)
                setGeometryOutputType(i, geometryOutputType.getValue());
        }
#ifdef GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT
        GLint maxV;
        glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT, &maxV);
        if (geometryVerticesOut.getValue() < 0 || geometryVerticesOut.getValue() > maxV)
        {
            for (unsigned int i=0 ; i<vertexFilenames.size() ; i++)
                geometryVerticesOut.setValue(3);
        }
#endif
        if (geometryVerticesOut.getValue() >= 0)
        {
            for (unsigned int i=0 ; i<vertexFilenames.size() ; i++)
                setGeometryVerticesOut(i, geometryVerticesOut.getValue());
        }



        for (unsigned int i=0 ; i<vertexFilenames.size() ; i++)
        {
            shaderVector[i]->InitShaders(helper::system::DataRepository.getFile(vertexFilenames[i]),
                    helper::system::DataRepository.getFile(geometryFilenames[i]),
                    helper::system::DataRepository.getFile(fragmentFilenames[i]));
        }
    }

}

void OglShader::drawVisual(const core::visual::VisualParams* )
{

}

void OglShader::stop()
{
    if ( backfaceWriting.getValue() )
        glDisable(GL_VERTEX_PROGRAM_TWO_SIDE);

    if ( !clampVertexColor.getValue() )
        glClampColorARB(GL_CLAMP_VERTEX_COLOR, GL_TRUE);

    if(turnOn.getValue())
        shaderVector[indexActiveShader.getValue()]->TurnOff();
}

void OglShader::start()
{
    if(turnOn.getValue())
        shaderVector[indexActiveShader.getValue()]->TurnOn();

    if ( !clampVertexColor.getValue() )
        glClampColorARB(GL_CLAMP_VERTEX_COLOR, GL_FALSE);

    if ( backfaceWriting.getValue() )
        glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
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
    return shaderVector.size();
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

OglShaderElement::OglShaderElement()
    : id(initData(&id, std::string(""), "id", "Set an ID name"))
    , indexShader(initData(&indexShader, (unsigned int) 0, "indexShader", "Set the index of the desired shader you want to apply this parameter"))
{

}

void OglShaderElement::init()
{
    sofa::core::objectmodel::BaseContext* mycontext = this->getContext();

    /*when no multipass is active */
    sofa::component::visualmodel::CompositingVisualLoop* isMultipass=NULL;
    isMultipass= mycontext->core::objectmodel::BaseContext::get<sofa::component::visualmodel::CompositingVisualLoop>();
    if(isMultipass==NULL)
    {
        shaders.insert(mycontext->core::objectmodel::BaseContext::get<OglShader>());
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
    if (id.getValue().empty())
        id.setValue(this->getName());
}


}//namespace visualmodel

} //namespace component

} //namespace sofa
