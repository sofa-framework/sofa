#ifndef _OGL_ATTRIBUTE_INL_
#define _OGL_ATTRIBUTE_INL_

#include <sofa/component/visualmodel/OglAttribute.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

template < int size, unsigned int type, class DataTypes>
OglAttribute< size, type, DataTypes>::OglAttribute() :
    OglShaderElement(),
    _abo ( -1 ),
    usage( GL_STATIC_DRAW)
{
}


template < int size, unsigned int type, class DataTypes>
OglAttribute< size, type, DataTypes>::~OglAttribute()
{
    glDeleteBuffers(1, &_abo);
}


template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::init()
{
    OglShaderElement::init();

    if ( ( int ) _abo == -1 ) glGenBuffers ( 1, &_abo );
}


template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::initVisual ()
{
    const DataTypes& data = value.getValue();
    unsigned int totalSize = data.size() *sizeof ( data[0] );

    glBindBuffer ( GL_ARRAY_BUFFER, _abo );

    glBufferData ( GL_ARRAY_BUFFER,
            totalSize,
            NULL,
            usage );

    // Fill the buffer
    glBufferSubData ( GL_ARRAY_BUFFER,
            0,
            totalSize,
            data.getData() );

    _index = shader->getAttribute ( 0, id.getValue().c_str() );

    enable();

    glBindBuffer(GL_ARRAY_BUFFER,0);

}


template < int size, unsigned int type, class DataTypes>
bool OglAttribute< size, type, DataTypes>::updateABO()
{
    GLvoid* attrib_bo = NULL;
    glBindBuffer(GL_ARRAY_BUFFER, _abo);
    attrib_bo = (GLvoid*) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

    if(attrib_bo == NULL)
    {
        std::cerr << "OglAttribute : Unknown error when updating attribute indices buffer "<< std::endl;
        return false;
    }
    const DataTypes& val = value.getValue();
    memcpy(attrib_bo, &(val[0]), val.size()*sizeof(val[0]));
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return true;
}



template < int size, unsigned int type, class DataTypes>
DataTypes* OglAttribute< size, type, DataTypes>::beginEdit()
{
    return value.beginEdit();
}


template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::endEdit()
{
    value.endEdit();
}


template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::setValue ( const DataTypes& value )
{
    DataTypes& val = * ( this->value.beginEdit() );
    val = value;
    this->value.endEdit();
}


template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::enable()
{
    glEnableVertexAttribArray ( _index );
    glVertexAttribPointer ( _index, size, type, GL_FALSE, 0, ( char* ) NULL + 0 );
}


template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::disable()
{
    glDisableVertexAttribArray ( _index );
}



template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::draw()
{
    glEnableVertexAttribArray ( _index );
    glVertexAttribPointer ( _index, size, type, GL_FALSE, 0, (void*)(&value.getValue()[0]));
}


template < int size, unsigned int type, class DataTypes>
void OglAttribute< size, type, DataTypes>::reinit()
{
    init();
    initVisual();
}

}

}

}

#endif
