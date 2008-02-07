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

class OglShader : public core::Shader, public core::VisualModel
{
private:
    Data<std::string> vertFilename;
    Data<std::string> fragFilename;
    sofa::helper::gl::CShader m_shader;

public:
    OglShader();
    virtual ~OglShader();

    void initVisual();
    void init();
    void drawVisual();
    void updateVisual();

    void start();
    void stop();

};


} //namespace component

} //namespace sofa

#endif //SOFA_COMPONENT_OGLSHADER
