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
#ifndef SOFA_COMPONENT_VISUALMODEL_OGLVARIABLE_H
#define SOFA_COMPONENT_VISUALMODEL_OGLVARIABLE_H
#include "config.h"

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/gl/template.h>
#include <SofaOpenglVisual/OglShader.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

/**
 *  \brief Defines an uniform variable for a OglShader.
 *
 *  This is an abstract class which pass a value to an uniform
 *  variable defined into the shader.
 *  At the moment, following types are supported :
 *   - int, ivec2, ivec3, ivec4;
 *   - float, vec2, vec3, vec4;
 *   - int[], ivec2[], ivec3[], ivec4[];
 *   - float[], vec2[], vec3[], vec4[];
 */

template<class DataTypes>
class OglVariable : public core::visual::VisualModel, public OglShaderElement
{
public:
    SOFA_CLASS2(OglVariable, core::visual::VisualModel, OglShaderElement);

    Data< DataTypes > value;

protected:
    OglVariable(): value(initData(&value, DataTypes(), "value", "Set Uniform Value"))
    {
        addAlias(&value, "values"); // some variable types hold multiple values, so we authorize both names for this attribute
    }

    virtual ~OglVariable() {}
public:
    virtual void setValue( const DataTypes& v ) { value.setValue(v); }
    void init() override { OglShaderElement::init(); }
    void initVisual() override { core::visual::VisualModel::initVisual(); }
    void pushValue() { initVisual(); }
    void reinit() override { init();	initVisual(); }
	void updateVisual() override { initVisual(); }

    /// Returns the type of shader element (texture, macro, variable, or attribute)
    virtual ShaderElementType getSEType() const override { return core::visual::ShaderElement::SE_VARIABLE; }
    // Returns the value of the shader element
    virtual const core::objectmodel::BaseData* getSEValue() const override { return &value; }
    // Returns the value of the shader element
    virtual core::objectmodel::BaseData* getSEValue() override { return &value; }

};

/** SINGLE INT VARIABLE **/
class SOFA_OPENGL_VISUAL_API OglIntVariable : public OglVariable< int>
{
public:
    SOFA_CLASS(OglIntVariable, OglVariable< int>);

    OglIntVariable();
    virtual ~OglIntVariable() { }

    void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglInt2Variable : public OglVariable<defaulttype::Vec<2, int> >
{

public:
    SOFA_CLASS(OglInt2Variable, SOFA_TEMPLATE(OglVariable, SOFA_TEMPLATE2(defaulttype::Vec, 2, int)));

    OglInt2Variable();
    virtual ~OglInt2Variable() { }

    void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglInt3Variable : public OglVariable<defaulttype::Vec<3, int> >
{
public:
    SOFA_CLASS(OglInt3Variable, SOFA_TEMPLATE(OglVariable, SOFA_TEMPLATE2(defaulttype::Vec, 3, int)));

    OglInt3Variable();
    virtual ~OglInt3Variable() { }

    void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglInt4Variable : public OglVariable<defaulttype::Vec<4, int> >
{
public:
    SOFA_CLASS(OglInt4Variable, SOFA_TEMPLATE(OglVariable, SOFA_TEMPLATE2(defaulttype::Vec, 4, int)));

    OglInt4Variable();
    virtual ~OglInt4Variable() { }

    void initVisual() override;
};

/** SINGLE FLOAT VARIABLE **/

class SOFA_OPENGL_VISUAL_API OglFloatVariable : public OglVariable<float>
{
public:
    SOFA_CLASS(OglFloatVariable, OglVariable<float>);

    OglFloatVariable();
    virtual ~OglFloatVariable() { }

    void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglFloat2Variable : public OglVariable<defaulttype::Vec2f>
{
public:
    SOFA_CLASS(OglFloat2Variable, OglVariable<defaulttype::Vec2f>);

    OglFloat2Variable();
    virtual ~OglFloat2Variable() { }

    void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglFloat3Variable : public OglVariable<defaulttype::Vec3f>
{
public:
    SOFA_CLASS(OglFloat3Variable, OglVariable<defaulttype::Vec3f>);

    OglFloat3Variable();
    virtual ~OglFloat3Variable() { }

    void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglFloat4Variable : public OglVariable<defaulttype::Vec4f>
{
public:
    SOFA_CLASS(OglFloat4Variable, OglVariable<defaulttype::Vec4f>);

    OglFloat4Variable();
    virtual ~OglFloat4Variable() { }

    void initVisual() override;
};

/** INT VECTOR VARIABLE **/
class SOFA_OPENGL_VISUAL_API OglIntVectorVariable : public OglVariable<helper::vector<GLint> >
{
public:
    SOFA_CLASS(OglIntVectorVariable, OglVariable<helper::vector<GLint> >);

    OglIntVectorVariable();
    virtual ~OglIntVectorVariable() { }

    virtual void init() override;
    virtual void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglIntVector2Variable : public OglIntVectorVariable
{

public:
    SOFA_CLASS(OglIntVector2Variable, OglIntVectorVariable);

    OglIntVector2Variable();
    virtual ~OglIntVector2Variable() { }

    virtual void init() override;
    virtual void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglIntVector3Variable : public OglIntVectorVariable
{
public:
    SOFA_CLASS(OglIntVector3Variable, OglIntVectorVariable);

    OglIntVector3Variable();
    virtual ~OglIntVector3Variable() { }

    virtual void init() override;
    virtual void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglIntVector4Variable : public OglIntVectorVariable
{
public:
    SOFA_CLASS(OglIntVector4Variable, OglIntVectorVariable);

    OglIntVector4Variable();
    virtual ~OglIntVector4Variable() { }

    virtual void init() override;
    virtual void initVisual() override;
};

/** FLOAT VECTOR VARIABLE **/
class SOFA_OPENGL_VISUAL_API OglFloatVectorVariable : public OglVariable<helper::vector<float> >
{
public:
    SOFA_CLASS(OglFloatVectorVariable, OglVariable<helper::vector<float> >);

    OglFloatVectorVariable();
    virtual ~OglFloatVectorVariable() { }

    virtual void init() override;
    virtual void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglFloatVector2Variable : public OglVariable<helper::vector<defaulttype::Vec2f> >
{
public:
    SOFA_CLASS(OglFloatVector2Variable, OglVariable<helper::vector<defaulttype::Vec2f> >);

    OglFloatVector2Variable();
    virtual ~OglFloatVector2Variable() { }

    virtual void init() override;
    virtual void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglFloatVector3Variable : public OglVariable<helper::vector<defaulttype::Vec3f> >
{
public:
    SOFA_CLASS(OglFloatVector3Variable, OglVariable<helper::vector<defaulttype::Vec3f> >);

    OglFloatVector3Variable();
    virtual ~OglFloatVector3Variable() { }

    virtual void init() override;
    virtual void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglFloatVector4Variable : public OglVariable<helper::vector<defaulttype::Vec4f> >
{
public:
    SOFA_CLASS(OglFloatVector4Variable, OglVariable<helper::vector<defaulttype::Vec4f> >);

    OglFloatVector4Variable();
    virtual ~OglFloatVector4Variable() { }

    virtual void init() override;
    virtual void initVisual() override;
};

/** Matrix VARIABLE **/
class SOFA_OPENGL_VISUAL_API OglMatrix2Variable : public OglVariable<helper::vector<float> >
{
public:
    SOFA_CLASS(OglMatrix2Variable,OglVariable<helper::vector<float> >);

    Data<bool> transpose;

    OglMatrix2Variable();
    virtual ~OglMatrix2Variable() { }

    virtual void setTranspose( const bool& v ) { transpose.setValue(v); }

    virtual void init() override;
    virtual void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglMatrix3Variable : public OglMatrix2Variable
{
public:
    SOFA_CLASS(OglMatrix3Variable,OglMatrix2Variable);

    OglMatrix3Variable();
    virtual ~OglMatrix3Variable() { }

    virtual void init() override;
    virtual void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglMatrix4Variable : public OglMatrix2Variable
{
public:
    SOFA_CLASS(OglMatrix4Variable,OglMatrix2Variable);

    OglMatrix4Variable();
    virtual ~OglMatrix4Variable() { }

    virtual void init() override;
    virtual void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglMatrix2x3Variable : public OglMatrix2Variable
{
public:
    SOFA_CLASS(OglMatrix2x3Variable,OglMatrix2Variable);

    OglMatrix2x3Variable();
    virtual ~OglMatrix2x3Variable() { }

    virtual void init() override;
    virtual void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglMatrix3x2Variable : public OglMatrix2Variable
{
public:
    SOFA_CLASS(OglMatrix3x2Variable,OglMatrix2Variable);

    OglMatrix3x2Variable();
    virtual ~OglMatrix3x2Variable() { }

    virtual void init() override;
    virtual void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglMatrix2x4Variable : public OglMatrix2Variable
{
public:
    SOFA_CLASS(OglMatrix2x4Variable,OglMatrix2Variable);

    OglMatrix2x4Variable();
    virtual ~OglMatrix2x4Variable() { }

    virtual void init() override;
    virtual void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglMatrix4x2Variable : public OglMatrix2Variable
{
public:
    SOFA_CLASS(OglMatrix4x2Variable,OglMatrix2Variable);

    OglMatrix4x2Variable();
    virtual ~OglMatrix4x2Variable() { }

    virtual void init() override;
    virtual void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglMatrix3x4Variable : public OglMatrix2Variable
{
public:
    SOFA_CLASS(OglMatrix3x4Variable,OglMatrix2Variable);

    OglMatrix3x4Variable();
    virtual ~OglMatrix3x4Variable() { }

    virtual void init() override;
    virtual void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglMatrix4x3Variable : public OglMatrix2Variable
{
public:
    SOFA_CLASS(OglMatrix4x3Variable,OglMatrix2Variable);

    OglMatrix4x3Variable();
    virtual ~OglMatrix4x3Variable() { }

    virtual void init() override;
    virtual void initVisual() override;
};

class SOFA_OPENGL_VISUAL_API OglMatrix4VectorVariable : public OglVariable<helper::vector<defaulttype::Mat4x4f> >
{
public:
    SOFA_CLASS(OglMatrix4VectorVariable,OglVariable<helper::vector<defaulttype::Mat4x4f> >);

    OglMatrix4VectorVariable();
    virtual ~OglMatrix4VectorVariable() { }

    virtual void init() override;
    virtual void initVisual() override;

    Data<bool> transpose;
    virtual void setTranspose( const bool& v ) { transpose.setValue(v); }

};



} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_VISUALMODEL_OGLVARIABLE_H
