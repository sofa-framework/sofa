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
#ifndef _OGL_ATTRIBUTE_INL_
#define _OGL_ATTRIBUTE_INL_

#include <SofaOpenglVisual/OglAttribute.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/TopologyChange.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace visualmodel
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
    _topology = NULL;
}

template < int size, unsigned int type, class DataTypes>
int OglAttribute< size, type, DataTypes >::getSETotalSize()
{
    const sofa::defaulttype::ResizableExtVector<DataTypes>& data = value.getValue();
    unsigned int totalSize = data.size() *sizeof ( data[0] );
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
}

template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::initVisual ()
{
    if ( _abo == GLuint(-1) ) glGenBuffers ( 1, &_abo );
    const sofa::defaulttype::ResizableExtVector<DataTypes>& data = value.getValue();
    unsigned int totalSize = data.size() *sizeof ( data[0] );
    _aboSize = totalSize;
    glBindBufferARB ( GL_ARRAY_BUFFER, _abo );
    glBufferDataARB ( GL_ARRAY_BUFFER,
            totalSize,
            NULL,
            _usage );
    // Fill the buffer
    glBufferSubDataARB ( GL_ARRAY_BUFFER,
            0,
            totalSize,
            data.getData() );
    _needUpdate = false;
    _lastUpdateDataCounter = value.getCounter();

    /*todo jeremy: add multi shaders management...temp solution for tonight...*/
    // todo(dmarchal)... I suspect that the Jeremy above is Jeremie Ringard that stop working on Sofa
    // since 2010.
    _index =  (*shaders.begin())->getAttribute ( indexShader.getValue(), id.getValue().c_str() ); //shader->getAttribute ( indexShader.getValue(), id.getValue().c_str() );
    if (_index == GLuint(-1) )
    {
        serr << "Variable \""<<id.getValue()<<"\" NOT FOUND in shader \"" << (*shaders.begin())->vertFilename.getValue() << "\""<< sendl;
    }
    else
    {
        sout << "Variable \""<<id.getValue()<<"\" in shader \"" << (*shaders.begin())->vertFilename.getValue() << "\" with index: " << _index << sendl;
    }
    //enable();
    glBindBufferARB(GL_ARRAY_BUFFER,0);
}

template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::updateVisual()
{
     if ( _abo == GLuint(-1) )
         return; // initVisual not yet called
    const sofa::defaulttype::ResizableExtVector<DataTypes>& data = value.getValue();
    unsigned int totalSize = data.size() *sizeof ( data[0] );
    int dataCounter = value.getCounter();
    if (!_needUpdate && totalSize == _aboSize && dataCounter == _lastUpdateDataCounter)
        return;

    glBindBufferARB ( GL_ARRAY_BUFFER, _abo );
    if (totalSize != _aboSize)
    {
        glBufferDataARB ( GL_ARRAY_BUFFER,
                totalSize,
                NULL,
                _usage );
        _aboSize = totalSize;
    }
    // Fill the buffer
    glBufferSubDataARB ( GL_ARRAY_BUFFER,
            0,
            totalSize,
            (char*)data.getData() );
    _needUpdate = false;
    _lastUpdateDataCounter = dataCounter;
    //enable();
    glBindBufferARB(GL_ARRAY_BUFFER,0);
}

template < int size, unsigned int type, class DataTypes>
sofa::defaulttype::ResizableExtVector<DataTypes>* OglAttribute< size, type, DataTypes>::beginEdit()
{
    return value.beginEdit();
}


template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::endEdit()
{
    value.endEdit();
}


template < int size, unsigned int type, class DataTypes>
const sofa::defaulttype::ResizableExtVector<DataTypes>& OglAttribute< size, type, DataTypes>::getValue() const
{
    return value.getValue();
}


template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::setValue ( const sofa::defaulttype::ResizableExtVector<DataTypes>& value )
{
    sofa::defaulttype::ResizableExtVector<DataTypes>& val = * ( this->value.beginEdit() );
    val = value;
    this->value.endEdit();
}


template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::enable()
{
    if (_index == GLuint(-1))
        return; // index not valid
    glBindBufferARB(GL_ARRAY_BUFFER, _abo);
#ifndef PS3
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
        glVertexAttribIPointer(_index, size, type, 0, (char*)NULL + 0);
        break;
    case GL_DOUBLE:
        glVertexAttribLPointer(_index, size, type, 0, (char*)NULL + 0);
        break;
    default:
        glVertexAttribPointer(_index, size, type, GL_FALSE, 0, (char*)NULL + 0);
    }
#else

    glVertexAttribPointerARB(_index, size, type, GL_FALSE, 0, (char*)NULL + 0);

#endif // __APPLE__

#endif // PS3
    glBindBufferARB(GL_ARRAY_BUFFER, 0);
}


template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::disable()
{
    if (_index == GLuint(-1))
        return; // index not valid
#ifndef PS3
    glDisableVertexAttribArrayARB ( _index );
    glBindBufferARB(GL_ARRAY_BUFFER,0);
#endif
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

// Only resizing and renumbering is done. 'value' has to be set by external components.
template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::handleTopologyChange()
{
    if( _topology && handleDynamicTopology.getValue())
    {
        std::list<const sofa::core::topology::TopologyChange *>::const_iterator itBegin=_topology->beginChange();
        std::list<const sofa::core::topology::TopologyChange *>::const_iterator itEnd=_topology->endChange();

        while( itBegin != itEnd )
        {
            core::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();

            switch( changeType )
            {
            case core::topology::ENDING_EVENT:
            {
                //sout << "INFO_print : Vis - ENDING_EVENT" << sendl;
                _needUpdate = true;
                break;
            }

            case core::topology::TRIANGLESADDED:
            {
                //sout << "INFO_print : Vis - TRIANGLESADDED" << sendl;
                break;
            }

            case core::topology::QUADSADDED:
            {
                //sout << "INFO_print : Vis - QUADSADDED" << sendl;
                break;
            }

            case core::topology::TRIANGLESREMOVED:
            {
                //sout << "INFO_print : Vis - TRIANGLESREMOVED" << sendl;
                break;
            }

            case core::topology::QUADSREMOVED:
            {
                //sout << "INFO_print : Vis - QUADSREMOVED" << sendl;
                break;
            }

            case core::topology::POINTSADDED:
            {
                //sout << "INFO_print : Vis - POINTSADDED" << sendl;

                unsigned int nbPoints = ( static_cast< const sofa::core::topology::PointsAdded * >( *itBegin ) )->getNbAddedVertices();
                sofa::defaulttype::ResizableExtVector<DataTypes>& data = *value.beginEdit();
                data.resize( data.size() + nbPoints);
                value.endEdit();
                break;
            }

            // Case "POINTSREMOVED" added to propagate the treatment to the Visual Model

            case core::topology::POINTSREMOVED:
            {
                //sout << "INFO_print : Vis - POINTSREMOVED" << sendl;

                const sofa::helper::vector<unsigned int> tab = ( static_cast< const sofa::core::topology::PointsRemoved * >( *itBegin ) )->getArray();
                sofa::defaulttype::ResizableExtVector<DataTypes>& data = *value.beginEdit();
                unsigned int last = data.size();

                for ( unsigned int i = 0; i < tab.size(); ++i)
                {
                    last--;
                    data[tab[i]] = data[last];
                }
                data.resize( last);
                value.endEdit();

                break;
            }

            // Case "POINTSRENUMBERING" added to propagate the treatment to the Visual Model

            case core::topology::POINTSRENUMBERING:
            {
                //sout << "INFO_print : Vis - POINTSRENUMBERING" << sendl;

                const sofa::helper::vector<unsigned int> tab = ( static_cast< const sofa::core::topology::PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();
                sofa::defaulttype::ResizableExtVector<DataTypes>& data = *value.beginEdit();
                helper::vector<DataTypes> tmp;
                for ( unsigned int i = 0; i < tab.size(); ++i)
                {
                    tmp.push_back( data[tab[i]]);
                }
                for ( unsigned int i = 0; i < tab.size(); ++i)
                {
                    data[i] = tmp[i];
                }
                value.endEdit();

                break;
            }

            default:
                // Ignore events that are not Triangle  related.
                break;
            }; // switch( changeType )

            ++itBegin;
        } // while( changeIt != last; )
    }
}

} // namespace visual

} // namespace component

} // namespace sofa

#endif
