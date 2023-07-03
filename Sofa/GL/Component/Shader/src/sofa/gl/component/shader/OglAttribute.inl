/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/gl/component/shader/OglAttribute.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/TopologyChange.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/topology/TopologyData.inl>

namespace sofa::gl::component::shader
{

template < int size, unsigned int type, class DataTypes>
OglAttribute< size, type, DataTypes>::OglAttribute() :
    OglShaderElement()
    , _abo ( GLuint(-1) ), _aboSize(0), _needUpdate(false), _lastUpdateDataCounter(-1)
    , _index ( GLuint(-1) ), _usage( GL_STATIC_DRAW)
    , value( initData(&value, "value", "internal Data"))
    , handleDynamicTopology( initData(&handleDynamicTopology, true, "handleDynamicTopology",
        "Activate handling of topological changes on the values of this attribute (resizes only)"))
{
    _topology = nullptr;
}

template < int size, unsigned int type, class DataTypes>
int OglAttribute< size, type, DataTypes >::getSETotalSize()
{
    const type::vector<DataTypes>& data = value.getValue();
    const unsigned int totalSize = data.size() *sizeof ( data[0] );
    return totalSize;
}

template < int size, unsigned int type, class DataTypes>
OglAttribute< size, type, DataTypes>::~OglAttribute()
{
    if (_abo != GLuint(-1) )
        glDeleteBuffersARB(1, &_abo);
}

template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::init()
{
    OglShaderElement::init();
    getContext()->get( _topology);
    value.getValue(); // make sure the data is updated

    if (_topology!= nullptr && handleDynamicTopology.getValue())
    {
        value.createTopologyHandler(_topology);
    }
}

template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::initVisual ()
{
    if ( _abo == GLuint(-1) ) glGenBuffers ( 1, &_abo );
    const type::vector<DataTypes>& data = value.getValue();
    unsigned int totalSize = data.size() *sizeof ( data[0] );
    _aboSize = totalSize;
    glBindBufferARB ( GL_ARRAY_BUFFER, _abo );
    glBufferDataARB ( GL_ARRAY_BUFFER,
            totalSize,
            nullptr,
            _usage );
    // Fill the buffer
    glBufferSubDataARB ( GL_ARRAY_BUFFER,
            0,
            totalSize,
            data.data() );
    _needUpdate = false;
    _lastUpdateDataCounter = value.getCounter();

    /*todo jeremy: add multi shaders management...temp solution for tonight...*/
    // todo(dmarchal)... I suspect that the Jeremy above is Jeremie Ringard that stop working on Sofa
    // since 2010.
    _index =  (*shaders.begin())->getAttribute ( indexShader.getValue(), id.getValue().c_str() ); //shader->getAttribute ( indexShader.getValue(), id.getValue().c_str() );
    if (_index == GLuint(-1) )
    {
        msg_error() << "Variable \"" << id.getValue() << "\" NOT FOUND in shader \"" << (*shaders.begin())->vertFilename.getValue() << "\"";
    }
    else
    {
        msg_info() << "Variable \"" << id.getValue() << "\" in shader \"" << (*shaders.begin())->vertFilename.getValue() << "\" with index: " << _index;
    }
    glBindBufferARB(GL_ARRAY_BUFFER,0);
}

template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::updateVisual()
{
     if ( _abo == GLuint(-1) )
         return; // initVisual not yet called
    const type::vector<DataTypes>& data = value.getValue();
    unsigned int totalSize = data.size() *sizeof ( data[0] );
     const int dataCounter = value.getCounter();
    if (!_needUpdate && totalSize == _aboSize && dataCounter == _lastUpdateDataCounter)
        return;

    glBindBufferARB ( GL_ARRAY_BUFFER, _abo );
    if (totalSize != _aboSize)
    {
        glBufferDataARB ( GL_ARRAY_BUFFER,
                totalSize,
                nullptr,
                _usage );
        _aboSize = totalSize;
    }
    // Fill the buffer
    glBufferSubDataARB ( GL_ARRAY_BUFFER,
            0,
            totalSize,
            (char*)data.data() );
    _needUpdate = false;
    _lastUpdateDataCounter = dataCounter;
    glBindBufferARB(GL_ARRAY_BUFFER,0);
}

template < int size, unsigned int type, class DataTypes>
type::vector<DataTypes>* OglAttribute< size, type, DataTypes>::beginEdit()
{
    return value.beginEdit();
}


template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::endEdit()
{
    value.endEdit();
}


template < int size, unsigned int type, class DataTypes>
const type::vector<DataTypes>& OglAttribute< size, type, DataTypes>::getValue() const
{
    return value.getValue();
}


template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::setValue ( const type::vector<DataTypes>& value )
{
    type::vector<DataTypes>& val = * ( this->value.beginEdit() );
    val = value;
    this->value.endEdit();
}


template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::enable()
{
    if (_index == GLuint(-1))
        return; // index not valid
    glBindBufferARB(GL_ARRAY_BUFFER, _abo);
    glEnableVertexAttribArrayARB(_index);

    //OS X does not support those functions in OpenGL Compatibility Mode
    ///TODO: if in the future Sofa works with OpenGL Core Mode,
    ///please remove these preprocessor instructions
#ifndef __APPLE__
    switch (type)
    {
    case GL_INT:
    case GL_UNSIGNED_INT:
    case GL_BYTE:
    case GL_UNSIGNED_BYTE:
    case GL_SHORT:
    case GL_UNSIGNED_SHORT:
        glVertexAttribIPointer(_index, size, type, 0, (char*)nullptr + 0);
        break;
    case GL_DOUBLE:
        glVertexAttribLPointer(_index, size, type, 0, (char*)nullptr + 0);
        break;
    default:
        glVertexAttribPointer(_index, size, type, GL_FALSE, 0, (char*)nullptr + 0);
    }
#else

    glVertexAttribPointerARB(_index, size, type, GL_FALSE, 0, (char*)nullptr + 0);

#endif // __APPLE__

    glBindBufferARB(GL_ARRAY_BUFFER, 0);
}


template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::disable()
{
    if (_index == GLuint(-1))
        return; // index not valid
    glDisableVertexAttribArrayARB ( _index );
    glBindBufferARB(GL_ARRAY_BUFFER,0);
}

template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::fwdDraw(core::visual::VisualParams*)
{
    enable();
}

template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::bwdDraw(core::visual::VisualParams*)
{
    disable();
}

template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::reinit()
{
    _needUpdate = true;
}

} // namespace sofa::gl::component::shader
