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

#ifndef _OGL_ATTRIBUTE_H_
#define _OGL_ATTRIBUTE_H_

#include <sofa/core/VisualModel.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/visualmodel/OglModel.h>
#include <sofa/component/visualmodel/OglShader.h>


namespace sofa
{

namespace component
{

namespace visualmodel
{

template< int size, unsigned int type, class DataTypes>
class OglAttribute: public core::VisualModel, public OglShaderElement
{
public:
    OglAttribute();
    virtual ~OglAttribute();

    virtual void init();

    virtual void initVisual();

    virtual void reinit();

    // TODO
    // if attributes are not static, need to update buffer
    bool updateABO();

    ResizableExtVector<DataTypes>* beginEdit();
    void endEdit();
    const ResizableExtVector<DataTypes>& getValue() const;
    void setValue( const ResizableExtVector<DataTypes>& value);
    void enable();
    void disable();
    virtual void bwdDraw(Pass);
    virtual void fwdDraw(Pass);

    void setUsage(unsigned int usage) { _usage = usage; }

    // handle topological changes
    virtual void handleTopologyChange();

protected:
    // attribute buffer object identity
    // to send data to the graphics card faster
    GLuint _abo;
    // memory index of the attribute into the graphics memory
    GLuint _index;

    unsigned int _usage;

    Data<ResizableExtVector<DataTypes> > value;

    sofa::core::componentmodel::topology::BaseMeshTopology* _topology;
};


/** FLOAT ATTRIBUTE **/
class SOFA_COMPONENT_VISUALMODEL_API OglFloatAttribute : public OglAttribute<1, GL_FLOAT, float>
{
public:
    OglFloatAttribute() {};
    virtual ~OglFloatAttribute() { };

};

class SOFA_COMPONENT_VISUALMODEL_API OglFloat2Attribute : public OglAttribute<2, GL_FLOAT, Vec<2, float> >
{
public:
    OglFloat2Attribute() {};
    virtual ~OglFloat2Attribute() { };

};

class SOFA_COMPONENT_VISUALMODEL_API OglFloat3Attribute : public OglAttribute<3, GL_FLOAT, Vec<3, float> >
{
public:
    OglFloat3Attribute() {};
    virtual ~OglFloat3Attribute() { };

};

class SOFA_COMPONENT_VISUALMODEL_API OglFloat4Attribute : public OglAttribute<4, GL_FLOAT, Vec<4, float> >
{
public:
    OglFloat4Attribute() {};
    virtual ~OglFloat4Attribute() { };

};




/** INT ATTRIBUTE **/
class SOFA_COMPONENT_VISUALMODEL_API OglIntAttribute : public OglAttribute<1, GL_INT, int>
{
public:
    OglIntAttribute() {};
    virtual ~OglIntAttribute() { };

};

class SOFA_COMPONENT_VISUALMODEL_API OglInt2Attribute : public OglAttribute<2, GL_INT, Vec<2, int> >
{
public:
    OglInt2Attribute() {};
    virtual ~OglInt2Attribute() { };

};

class SOFA_COMPONENT_VISUALMODEL_API OglInt3Attribute : public OglAttribute<3, GL_INT, Vec<3, int> >
{
public:
    OglInt3Attribute() {};
    virtual ~OglInt3Attribute() { };

};

class SOFA_COMPONENT_VISUALMODEL_API OglInt4Attribute : public OglAttribute<4, GL_INT, Vec<4, int> >
{
public:
    OglInt4Attribute() {};
    virtual ~OglInt4Attribute() { };

};




/** UNSIGNED INT ATTRIBUTE **/
class SOFA_COMPONENT_VISUALMODEL_API OglUIntAttribute : public OglAttribute<1, GL_UNSIGNED_INT, unsigned int>
{
public:
    OglUIntAttribute() {};
    virtual ~OglUIntAttribute() { };

};

class SOFA_COMPONENT_VISUALMODEL_API OglUInt2Attribute : public OglAttribute<2, GL_UNSIGNED_INT, Vec<2, unsigned int> >
{
public:
    OglUInt2Attribute() {};
    virtual ~OglUInt2Attribute() { };

};

class SOFA_COMPONENT_VISUALMODEL_API OglUInt3Attribute : public OglAttribute<3, GL_UNSIGNED_INT, Vec<3, unsigned int> >
{
public:
    OglUInt3Attribute() {};
    virtual ~OglUInt3Attribute() { };

};

class SOFA_COMPONENT_VISUALMODEL_API OglUInt4Attribute : public OglAttribute<4, GL_UNSIGNED_INT, Vec<4, unsigned int> >
{
public:
    OglUInt4Attribute() {};
    virtual ~OglUInt4Attribute() { };

};

}

}

}

#endif
