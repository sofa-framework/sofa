/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_HELPER_GL_BASICSHAPESGL_INL
#define SOFA_HELPER_GL_BASICSHAPESGL_INL

#include <sofa/helper/gl/BasicShapesGL.h>

namespace sofa
{

namespace helper
{

namespace gl
{

template<class VertexType>
BasicShapesGL_Sphere<VertexType>::BasicShapesGL_Sphere()
{
}

template<class VertexType>
BasicShapesGL_Sphere<VertexType>::~BasicShapesGL_Sphere()
{
    glDeleteBuffers(1, &m_VBO);
    glDeleteBuffers(1, &m_IBO);

}

// http://stackoverflow.com/questions/5988686/creating-a-3d-sphere-in-opengl-using-visual-c
template<class VertexType>
void BasicShapesGL_Sphere<VertexType>::init(const unsigned int rings, const unsigned int sectors)
{
    glGenBuffers(1, &m_VBO);
    glGenBuffers(1, &m_IBO);

    //int rings = 32;
    //int sectors = 16;
    float radius = 1.0;

    float const R = 1. / (float)(rings - 1);
    float const S = 1. / (float)(sectors - 1);
    int r, s;

    std::vector<GLfloat> vertices;
    std::vector<GLfloat> normals;
    std::vector<GLfloat> texcoords;
    std::vector<GLushort> indices;

    vertices.resize(rings * sectors * 3);
    normals.resize(rings * sectors * 3);
    texcoords.resize(rings * sectors * 2);
    indices.resize(rings * sectors * 4);
    std::vector<GLfloat>::iterator v = vertices.begin();
    std::vector<GLfloat>::iterator n = normals.begin();
    std::vector<GLfloat>::iterator t = texcoords.begin();
    std::vector<GLushort>::iterator i = indices.begin();

    for (r = 0; r < rings; r++)
    {
        for (s = 0; s < sectors; s++)
        {
            float const y = sin(-M_PI_2 + M_PI * r * R);
            float const x = cos(2 * M_PI * s * S) * sin(M_PI * r * R);
            float const z = sin(2 * M_PI * s * S) * sin(M_PI * r * R);

            *t++ = s*S;
            *t++ = r*R;

            *v++ = x * radius;
            *v++ = y * radius;
            *v++ = z * radius;

            *n++ = x;
            *n++ = y;
            *n++ = z;
        }
    }

    for (r = 0; r < rings - 1; r++)
    {
        for (s = 0; s < sectors - 1; s++)
        {
            *i++ = r * sectors + s;
            *i++ = r * sectors + (s + 1);
            *i++ = (r + 1) * sectors + (s + 1);
            *i++ = (r + 1) * sectors + s;
        }
    }
    
    //Generate PositionVBO
    m_verticesBufferSize = (vertices.size()*sizeof(vertices[0]));
    m_normalsBufferSize = (normals.size()*sizeof(normals[0]));
    m_texcoordsBufferSize = (texcoords.size()*sizeof(texcoords[0]));
    unsigned int totalSize = m_verticesBufferSize + m_normalsBufferSize + m_texcoordsBufferSize;

    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferDataARB(GL_ARRAY_BUFFER,
        totalSize,
        NULL,
        GL_DYNAMIC_DRAW);

    glBufferSubDataARB(GL_ARRAY_BUFFER,
        0,
        m_verticesBufferSize,
        &(vertices[0]));
    glBufferSubDataARB(GL_ARRAY_BUFFER,
        m_verticesBufferSize,
        m_normalsBufferSize,
        &(normals[0]));
    glBufferSubDataARB(GL_ARRAY_BUFFER,
        m_verticesBufferSize + m_normalsBufferSize,
        m_texcoordsBufferSize,
        &(texcoords[0]));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    //IBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(indices[0]), &(indices[0]), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    m_indicesSize = indices.size();

}

template<class VertexType>
void BasicShapesGL_Sphere<VertexType>::internalDraw(const VertexType& center, const float& radius)
{
    glPushMatrix();
    glTranslatef(center[0], center[1], center[2]);
    glScalef(radius, radius, radius);

    glDrawElements(GL_QUADS, m_indicesSize, GL_UNSIGNED_SHORT, (void*)0);

    glPopMatrix();
}
template<class VertexType>
void BasicShapesGL_Sphere<VertexType>::draw(const VertexType& center, const float& radius)
{
    glMatrixMode(GL_MODELVIEW);

    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glVertexPointer(3, GL_FLOAT, 0, (char*)NULL + 0);
    glNormalPointer(GL_FLOAT, 0, (char*)NULL + m_verticesBufferSize);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IBO);

    internalDraw(center, radius);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

template<class VertexType>
void BasicShapesGL_Sphere<VertexType>::draw(const helper::vector<VertexType>& centers, const float& radius)
{
    glMatrixMode(GL_MODELVIEW);

    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glVertexPointer(3, GL_FLOAT, 0, (char*)NULL + 0);
    glNormalPointer(GL_FLOAT, 0, (char*)NULL + m_verticesBufferSize);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IBO);

    //assert on size ?
    for (unsigned int c = 0; c < centers.size(); c++)
    {
        internalDraw(centers[c], radius);
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

template<class VertexType>
void BasicShapesGL_Sphere<VertexType>::draw(const helper::vector<VertexType>& centers, const std::vector<float>& radius)
{
    glMatrixMode(GL_MODELVIEW);

    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glVertexPointer(3, GL_FLOAT, 0, (char*)NULL + 0);
    glNormalPointer(GL_FLOAT, 0, (char*)NULL + m_verticesBufferSize);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IBO);

    //assert on size ?
    for (unsigned int c = 0; c < centers.size(); c++)
    {
        internalDraw(centers[c], radius[c]);
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}


} //gl
} //helper
} //sofa


#endif // SOFA_HELPER_GL_BASICSHAPESGL_INL