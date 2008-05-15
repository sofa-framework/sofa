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
#ifndef SOFA_COMPONENT_OGLSHADER
#define SOFA_COMPONENT_OGLSHADER

#include <sofa/core/VisualModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/Shader.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/GLshader.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class OglShader : public core::Shader, public core::VisualModel
{
protected:
    Data<std::string> vertFilename;
    Data<std::string> fragFilename;
    Data<std::string> geoFilename;

    //TODO replace it?
    Data<int> geometryInputTypes;
    Data<int> geometryOutputTypes;
    Data<int> geometryVerticesOut;

    sofa::helper::gl::CShader m_shader;

    bool hasGeometryShader;

public:
    OglShader();
    virtual ~OglShader();

    void initVisual();
    void init();
    void reinit();
    void drawVisual();
    void updateVisual();

    void start();
    void stop();

    void setTexture(const char* name, unsigned short unit);

    void setInt(const char* name, int i);
    void setInt2(const char* name, int i1, int i2);
    void setInt3(const char* name, int i1, int i2, int i3);
    void setInt4(const char* name, int i1, int i2, int i3, int i4);

    void setFloat(const char* name, float f1);
    void setFloat2(const char* name, float f1, float f2);
    void setFloat3(const char* name, float f1, float f2, float f3);
    void setFloat4(const char* name, float f1, float f2, float f3, float f4);

    void setIntVector(const char* name, int count, const int* i);
    void setIntVector2(const char* name, int count, const int* i);
    void setIntVector3(const char* name, int count, const int* i);
    void setIntVector4(const char* name, int count, const int* i);

    void setFloatVector(const char* name, int count, const float* f);
    void setFloatVector2(const char* name, int count, const float* f);
    void setFloatVector3(const char* name, int count, const float* f);
    void setFloatVector4(const char* name, int count, const float* f);

    GLint getGeometryInputType() ;
    void  setGeometryInputType(GLint v) ;

    GLint getGeometryOutputType() ;
    void  setGeometryOutputType(GLint v) ;

    GLint getGeometryVerticesOut() ;
    void  setGeometryVerticesOut(GLint v);
};

class OglShaderElement : public core::ShaderElement
{
protected:
    Data<std::string> id;
    OglShader* shader;
public:
    OglShaderElement();
    virtual ~OglShaderElement() { };
    virtual void init();

    //virtual void setInShader(OglShader& s) = 0;
};

}//namespace visualmodel

} //namespace component

} //namespace sofa

#endif //SOFA_COMPONENT_OGLSHADER
