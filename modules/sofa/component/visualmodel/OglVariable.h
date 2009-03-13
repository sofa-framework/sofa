/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef OGLVARIABLE_H_
#define OGLVARIABLE_H_

#include <sofa/core/VisualModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/gl/template.h>
#include <sofa/component/visualmodel/OglShader.h>

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
 *  At the moment, following types have been supported :
 *   - int, ivec2, ivec3, ivec4;
 *   - float, vec2, vec3, vec4;
 *   - int[], ivec2[], ivec3[], ivec4[];
 *   - float[], vec2[], vec3[], vec4[];
 */

template<class DataTypes>
class SOFA_COMPONENT_VISUALMODEL_API OglVariable : public core::VisualModel, public OglShaderElement
{
protected:
    Data< DataTypes > value;
public:
    OglVariable():value(initData(&value, DataTypes(), "value", "Set Uniform Value")) { };
    virtual ~OglVariable() { };


    void setValue( const DataTypes& v) {value.setValue(v);}
    virtual void init() { OglShaderElement::init(); };
    virtual void initVisual() { }
    virtual void reinit() { init(); initVisual(); }
};

/** SINGLE INT VARIABLE **/
class SOFA_COMPONENT_VISUALMODEL_API OglIntVariable : public OglVariable< int>
{
public:
    OglIntVariable();
    virtual ~OglIntVariable() { };

    void initVisual();
};

class SOFA_COMPONENT_VISUALMODEL_API OglInt2Variable : public OglVariable<defaulttype::Vec<2, int> >
{

public:
    OglInt2Variable();
    virtual ~OglInt2Variable() { };

    void initVisual();
};

class SOFA_COMPONENT_VISUALMODEL_API OglInt3Variable : public OglVariable<defaulttype::Vec<3, int> >
{
public:
    OglInt3Variable();
    virtual ~OglInt3Variable() { };

    void initVisual();
};

class SOFA_COMPONENT_VISUALMODEL_API OglInt4Variable : public OglVariable<defaulttype::Vec<4, int> >
{
public:
    OglInt4Variable();
    virtual ~OglInt4Variable() { };

    void initVisual();
};

/** SINGLE FLOAT VARIABLE **/

class SOFA_COMPONENT_VISUALMODEL_API OglFloatVariable : public OglVariable<float>
{
public:
    OglFloatVariable();
    virtual ~OglFloatVariable() { };

    void initVisual();
};

class SOFA_COMPONENT_VISUALMODEL_API OglFloat2Variable : public OglVariable<defaulttype::Vec2f>
{
public:
    OglFloat2Variable();
    virtual ~OglFloat2Variable() { };

    void initVisual();
};

class SOFA_COMPONENT_VISUALMODEL_API OglFloat3Variable : public OglVariable<defaulttype::Vec3f>
{
public:
    OglFloat3Variable();
    virtual ~OglFloat3Variable() { };

    void initVisual();
};

class SOFA_COMPONENT_VISUALMODEL_API OglFloat4Variable : public OglVariable<defaulttype::Vec4f>
{
public:
    OglFloat4Variable();
    virtual ~OglFloat4Variable() { };

    void initVisual();
};

/** INT VECTOR VARIABLE **/
class SOFA_COMPONENT_VISUALMODEL_API OglIntVectorVariable : public OglVariable<helper::vector<GLint> >
{
public:
    OglIntVectorVariable();
    virtual ~OglIntVectorVariable() { };

    virtual void init();
    virtual void initVisual();
};

class SOFA_COMPONENT_VISUALMODEL_API OglIntVector2Variable : public OglIntVectorVariable
{

public:
    OglIntVector2Variable();
    virtual ~OglIntVector2Variable() { };

    virtual void init();
    virtual void initVisual();
};

class SOFA_COMPONENT_VISUALMODEL_API OglIntVector3Variable : public OglIntVectorVariable
{
public:
    OglIntVector3Variable();
    virtual ~OglIntVector3Variable() { };

    virtual void init();
    virtual void initVisual();
};

class SOFA_COMPONENT_VISUALMODEL_API OglIntVector4Variable : public OglIntVectorVariable
{
public:
    OglIntVector4Variable();
    virtual ~OglIntVector4Variable() { };

    virtual void init();
    virtual void initVisual();
};

/** FLOAT VECTOR VARIABLE **/
class SOFA_COMPONENT_VISUALMODEL_API OglFloatVectorVariable : public OglVariable<helper::vector<float> >
{
public:
    OglFloatVectorVariable();
    virtual ~OglFloatVectorVariable() { };

    virtual void init();
    virtual void initVisual();
};

class SOFA_COMPONENT_VISUALMODEL_API OglFloatVector2Variable : public OglFloatVectorVariable
{
public:
    OglFloatVector2Variable();
    virtual ~OglFloatVector2Variable() { };

    virtual void init();
    virtual void initVisual();
};

class SOFA_COMPONENT_VISUALMODEL_API OglFloatVector3Variable : public OglFloatVectorVariable
{
public:
    OglFloatVector3Variable();
    virtual ~OglFloatVector3Variable() { };

    virtual void init();
    virtual void initVisual();
};

class SOFA_COMPONENT_VISUALMODEL_API OglFloatVector4Variable : public OglFloatVectorVariable
{
public:
    OglFloatVector4Variable();
    virtual ~OglFloatVector4Variable() { };

    virtual void init();
    virtual void initVisual();
};

}

}

}

#endif /*OGLVARIABLE_H_*/
