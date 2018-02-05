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

#ifndef _OGL_ATTRIBUTE_H_
#define _OGL_ATTRIBUTE_H_
#include "config.h"

#include <sofa/core/visual/VisualModel.h>
#include <sofa/defaulttype/VecTypes.h>
#include <SofaOpenglVisual/OglModel.h>
#include <SofaOpenglVisual/OglShader.h>


namespace sofa
{

namespace component
{

namespace visualmodel
{

template< int size, unsigned int type, class DataTypes>
class OglAttribute: public core::visual::VisualModel, public OglShaderElement
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE3(OglAttribute, size, type, DataTypes), core::visual::VisualModel, OglShaderElement);
protected:
    OglAttribute();
    virtual ~OglAttribute();
public:
    virtual void init() override;

    virtual void initVisual() override;

    virtual void reinit() override;

    /// if attributes are not static, update the buffer
    void updateVisual() override;

    sofa::defaulttype::ResizableExtVector<DataTypes>* beginEdit();
    void endEdit();
    const sofa::defaulttype::ResizableExtVector<DataTypes>& getValue() const;
    void setValue( const sofa::defaulttype::ResizableExtVector<DataTypes>& value);
    void enable();
    void disable();
    virtual void bwdDraw(core::visual::VisualParams* ) override;
    virtual void fwdDraw(core::visual::VisualParams* ) override;

    void setUsage(unsigned int usage) { _usage = usage; }

    // handle topological changes
    virtual void handleTopologyChange() override;

    /// Returns the type of shader element (texture, macro, variable, or attribute)
    virtual ShaderElementType getSEType() const override { return core::visual::ShaderElement::SE_ATTRIBUTE; }
    // Returns the value of the shader element
    virtual const core::objectmodel::BaseData* getSEValue() const override { return &value; }
    // Returns the value of the shader element
    virtual core::objectmodel::BaseData* getSEValue() override { return &value; }
    // For attributes : return the number of values per vertex
    virtual int getSESizePerVertex() override { return size; }
    // Returns the total size of the values
    virtual int getSETotalSize() override;

protected:
    // attribute buffer object identity
    // to send data to the graphics card faster
    GLuint _abo;
    unsigned int _aboSize;
    bool _needUpdate;
    int _lastUpdateDataCounter;
    // memory index of the attribute into the graphics memory
    GLuint _index;

    unsigned int _usage;

    Data<sofa::defaulttype::ResizableExtVector<DataTypes> > value;
    Data<bool> handleDynamicTopology;

    sofa::core::topology::BaseMeshTopology* _topology;
};

/** FLOAT ATTRIBUTE **/
class SOFA_OPENGL_VISUAL_API OglFloatAttribute : public OglAttribute<1, GL_FLOAT, float>
{
public:
    SOFA_CLASS(OglFloatAttribute, SOFA_TEMPLATE3(OglAttribute, 1, GL_FLOAT, float));
    OglFloatAttribute() {};
    virtual ~OglFloatAttribute() { };

};

class SOFA_OPENGL_VISUAL_API OglFloat2Attribute : public OglAttribute<2, GL_FLOAT, sofa::defaulttype::Vec<2, float> >
{
public:
    SOFA_CLASS(OglFloat2Attribute, SOFA_TEMPLATE3(OglAttribute, 2, GL_FLOAT, SOFA_TEMPLATE2(sofa::defaulttype::Vec, 2, float)));
    OglFloat2Attribute() {};
    virtual ~OglFloat2Attribute() { };

};

class SOFA_OPENGL_VISUAL_API OglFloat3Attribute : public OglAttribute<3, GL_FLOAT, sofa::defaulttype::Vec<3, float> >
{
public:
    SOFA_CLASS(OglFloat3Attribute, SOFA_TEMPLATE3(OglAttribute, 3, GL_FLOAT, SOFA_TEMPLATE2(sofa::defaulttype::Vec, 3, float)));
    OglFloat3Attribute() {};
    virtual ~OglFloat3Attribute() { };

};

class SOFA_OPENGL_VISUAL_API OglFloat4Attribute : public OglAttribute<4, GL_FLOAT, sofa::defaulttype::Vec<4, float> >
{
public:
    SOFA_CLASS(OglFloat4Attribute, SOFA_TEMPLATE3(OglAttribute, 4, GL_FLOAT, SOFA_TEMPLATE2(sofa::defaulttype::Vec, 4, float)));
    OglFloat4Attribute() {};
    virtual ~OglFloat4Attribute() { };

};




/** INT ATTRIBUTE **/
class SOFA_OPENGL_VISUAL_API OglIntAttribute : public OglAttribute<1, GL_INT, int>
{
public:
    SOFA_CLASS(OglIntAttribute, SOFA_TEMPLATE3(OglAttribute, 1, GL_INT, int));
    OglIntAttribute() {};
    virtual ~OglIntAttribute() { };

};

class SOFA_OPENGL_VISUAL_API OglInt2Attribute : public OglAttribute<2, GL_INT, sofa::defaulttype::Vec<2, int> >
{
public:
    SOFA_CLASS(OglInt2Attribute, SOFA_TEMPLATE3(OglAttribute, 2, GL_INT, SOFA_TEMPLATE2(sofa::defaulttype::Vec, 2, int)));
    OglInt2Attribute() {};
    virtual ~OglInt2Attribute() { };

};

class SOFA_OPENGL_VISUAL_API OglInt3Attribute : public OglAttribute<3, GL_INT, sofa::defaulttype::Vec<3, int> >
{
public:
    SOFA_CLASS(OglInt3Attribute, SOFA_TEMPLATE3(OglAttribute, 3, GL_INT, SOFA_TEMPLATE2(sofa::defaulttype::Vec, 3, int)));
    OglInt3Attribute() {};
    virtual ~OglInt3Attribute() { };

};

class SOFA_OPENGL_VISUAL_API OglInt4Attribute : public OglAttribute<4, GL_INT, sofa::defaulttype::Vec<4, int> >
{
public:
    SOFA_CLASS(OglInt4Attribute, SOFA_TEMPLATE3(OglAttribute, 4, GL_INT, SOFA_TEMPLATE2(sofa::defaulttype::Vec, 4, int)));
    OglInt4Attribute() {};
    virtual ~OglInt4Attribute() { };

};




/** UNSIGNED INT ATTRIBUTE **/
class SOFA_OPENGL_VISUAL_API OglUIntAttribute : public OglAttribute<1, GL_UNSIGNED_INT, unsigned int>
{
public:
    SOFA_CLASS(OglUIntAttribute, SOFA_TEMPLATE3(OglAttribute, 1, GL_UNSIGNED_INT, unsigned int));
    OglUIntAttribute() {};
    virtual ~OglUIntAttribute() { };

};

class SOFA_OPENGL_VISUAL_API OglUInt2Attribute : public OglAttribute<2, GL_UNSIGNED_INT, sofa::defaulttype::Vec<2, unsigned int> >
{
public:
    SOFA_CLASS(OglUInt2Attribute, SOFA_TEMPLATE3(OglAttribute, 2, GL_UNSIGNED_INT, SOFA_TEMPLATE2(sofa::defaulttype::Vec, 2, unsigned int)));
    OglUInt2Attribute() {};
    virtual ~OglUInt2Attribute() { };

};

class SOFA_OPENGL_VISUAL_API OglUInt3Attribute : public OglAttribute<3, GL_UNSIGNED_INT, sofa::defaulttype::Vec<3, unsigned int> >
{
public:
    SOFA_CLASS(OglUInt3Attribute, SOFA_TEMPLATE3(OglAttribute, 3, GL_UNSIGNED_INT, SOFA_TEMPLATE2(sofa::defaulttype::Vec, 3, unsigned int)));
    OglUInt3Attribute() {};
    virtual ~OglUInt3Attribute() { };

};

class SOFA_OPENGL_VISUAL_API OglUInt4Attribute : public OglAttribute<4, GL_UNSIGNED_INT, sofa::defaulttype::Vec<4, unsigned int> >
{
public:
    SOFA_CLASS(OglUInt4Attribute, SOFA_TEMPLATE3(OglAttribute, 4, GL_UNSIGNED_INT, SOFA_TEMPLATE2(sofa::defaulttype::Vec, 4, unsigned int)));
    OglUInt4Attribute() {};
    virtual ~OglUInt4Attribute() { };

};

} // namespace visual

} // namespace component

} // namespace sofa

#endif
