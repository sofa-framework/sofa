/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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

#include <sofa/helper/gl/shaders/generateSphere.cppglsl>

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
    typename std::map<SphereDescription, GLBuffers>::const_iterator it;
    for (it = m_mapBuffers.begin(); it != m_mapBuffers.end(); ++it)
    {
        const GLBuffers& buffer = it->second;
        glDeleteBuffers(1, &buffer.VBO);
        glDeleteBuffers(1, &buffer.IBO);
    }
}

// http://stackoverflow.com/questions/5988686/creating-a-3d-sphere-in-opengl-using-visual-c
template<class VertexType>
void BasicShapesGL_Sphere<VertexType>::generateBuffer(const SphereDescription &desc, GLBuffers &buffer)
{
    glGenBuffers(1, &buffer.VBO);
    glGenBuffers(1, &buffer.IBO);

    float radius = 1.0;

    float const R = 1. / (float)(desc.rings - 1);
    float const S = 1. / (float)(desc.sectors - 1);
    unsigned int r, s;

    std::vector<GLfloat> vertices;
    std::vector<GLfloat> normals;
    std::vector<GLfloat> texcoords;
    std::vector<GLuint> indices;

    vertices.resize(desc.rings * desc.sectors * 3);
    normals.resize(desc.rings * desc.sectors * 3);
    texcoords.resize(desc.rings * desc.sectors * 2);
    indices.resize(desc.rings * desc.sectors * 4);
    std::vector<GLfloat>::iterator v = vertices.begin();
    std::vector<GLfloat>::iterator n = normals.begin();
    std::vector<GLfloat>::iterator t = texcoords.begin();
    std::vector<GLuint>::iterator i = indices.begin();

    for (r = 0; r < desc.rings; r++)
    {
        for (s = 0; s < desc.sectors; s++)
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

    for (r = 0; r < desc.rings - 1; r++)
    {
        for (s = 0; s < desc.sectors - 1; s++)
        {
            *i++ = r * desc.sectors + s;
            *i++ = r * desc.sectors + (s + 1);
            *i++ = (r + 1) * desc.sectors + (s + 1);
            *i++ = (r + 1) * desc.sectors + s;
        }
    }

    //Generate PositionVBO
    buffer.verticesBufferSize = (vertices.size()*sizeof(vertices[0]));
    buffer.normalsBufferSize = (normals.size()*sizeof(normals[0]));
    buffer.texcoordsBufferSize = (texcoords.size()*sizeof(texcoords[0]));
    buffer.totalSize = buffer.verticesBufferSize + buffer.normalsBufferSize + buffer.texcoordsBufferSize;

    glBindBuffer(GL_ARRAY_BUFFER, buffer.VBO);
    glBufferDataARB(GL_ARRAY_BUFFER,
        buffer.totalSize,
        NULL,
        GL_DYNAMIC_DRAW);

    glBufferSubDataARB(GL_ARRAY_BUFFER,
        0,
        buffer.verticesBufferSize,
        &(vertices[0]));
    glBufferSubDataARB(GL_ARRAY_BUFFER,
        buffer.verticesBufferSize,
        buffer.normalsBufferSize,
        &(normals[0]));
    glBufferSubDataARB(GL_ARRAY_BUFFER,
        buffer.verticesBufferSize + buffer.normalsBufferSize,
        buffer.texcoordsBufferSize,
        &(texcoords[0]));

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //IBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer.IBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(indices[0]), &(indices[0]), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    buffer.indicesSize = indices.size();

}

template<class VertexType>
void BasicShapesGL_Sphere<VertexType>::internalDraw(const GLBuffers &buffer, const VertexType& center, const float& radius)
{
    glPushMatrix();
    glTranslatef(center[0], center[1], center[2]);
    glScalef(radius, radius, radius);

    glDrawElements(GL_QUADS, buffer.indicesSize, GL_UNSIGNED_INT, (void*)0);

    glPopMatrix();
}

template<class VertexType>
void BasicShapesGL_Sphere<VertexType>::beforeDraw(const GLBuffers& buffer)
{
    glMatrixMode(GL_MODELVIEW);

    glBindBuffer(GL_ARRAY_BUFFER, buffer.VBO);
    glVertexPointer(3, GL_FLOAT, 0, (char*)NULL + 0);
    glNormalPointer(GL_FLOAT, 0, (char*)NULL + buffer.verticesBufferSize);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer.IBO);
}

template<class VertexType>
void BasicShapesGL_Sphere<VertexType>::afterDraw(const GLBuffers &/* buffer */)
{
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

template<class VertexType>
void BasicShapesGL_Sphere<VertexType>::checkBuffers(const SphereDescription& desc)
{
    if (m_mapBuffers.find(desc) == m_mapBuffers.end())
    {
        GLBuffers glbuffer;
        generateBuffer(desc, glbuffer);
        m_mapBuffers[desc] = glbuffer;
    }

}

template<class VertexType>
void BasicShapesGL_Sphere<VertexType>::draw(const VertexType& center, const float& radius, const unsigned int rings, const unsigned int sectors)
{
    SphereDescription desc(rings, sectors);
    checkBuffers(desc);

    beforeDraw(m_mapBuffers[desc]);
    internalDraw(m_mapBuffers[desc], center, radius);
    afterDraw(m_mapBuffers[desc]);

}

template<class VertexType>
void BasicShapesGL_Sphere<VertexType>::draw(const helper::vector<VertexType>& centers, const float& radius, const unsigned int rings, const unsigned int sectors)
{
    SphereDescription desc(rings, sectors);
    checkBuffers(desc);

    beforeDraw(m_mapBuffers[desc]);

    for (unsigned int c = 0; c < centers.size(); c++)
    {
        internalDraw(m_mapBuffers[desc], centers[c], radius);
    }

    afterDraw(m_mapBuffers[desc]);
}

template<class VertexType>
void BasicShapesGL_Sphere<VertexType>::draw(const helper::vector<VertexType>& centers, const std::vector<float>& radius, const unsigned int rings, const unsigned int sectors)
{
    SphereDescription desc(rings, sectors);
    checkBuffers(desc);

    beforeDraw(m_mapBuffers[desc]);

    //assert on size ?
    for (unsigned int c = 0; c < centers.size(); c++)
    {
        internalDraw(m_mapBuffers[desc], centers[c], radius[c]);
    }

    afterDraw(m_mapBuffers[desc]);
}

//////////////////////////////////////////////////////////////////////
template<class VertexType>
BasicShapesGL_FakeSphere<VertexType>::BasicShapesGL_FakeSphere()
    : m_shader(NULL)
    , b_isInit(false)
{
}

template<class VertexType>
BasicShapesGL_FakeSphere<VertexType>::~BasicShapesGL_FakeSphere()
{
    glDeleteBuffers(1, &m_buffer.VBO);
    glDeleteBuffers(1, &m_buffer.IBO);
    glDeleteBuffers(1, &m_radiusBuffer.VBO);
}

///
template<class VertexType>
void BasicShapesGL_FakeSphere<VertexType>::init()
{
    if (!b_isInit)
    {
        if (!sofa::helper::gl::GLSLShader::InitGLSL())
        {
            msg_info("BasicShapesGL") << "InitGLSL failed" ;
            return;
        }
        std::string vertexShaderContent = sofa::helper::gl::generateSphereVS;
        std::string fragmentShaderContent = sofa::helper::gl::generateSphereFS;

        m_shader = new GLSLShader();
        m_shader->SetVertexShaderFromString(vertexShaderContent);
        m_shader->SetFragmentShaderFromString(fragmentShaderContent);
        m_shader->InitShaders();
        b_isInit = true;

        m_shader->TurnOn();
        m_radiusBuffer.location = m_shader->GetAttributeVariable("a_radius");
        m_shader->TurnOff();

        glGenBuffers(1, &m_buffer.VBO);
        glGenBuffers(1, &m_buffer.IBO);
        glGenBuffers(1, &m_radiusBuffer.VBO);
    }

}

// http://stackoverflow.com/questions/5988686/creating-a-3d-sphere-in-opengl-using-visual-c
template<class VertexType>
void BasicShapesGL_FakeSphere<VertexType>::generateBuffer(const std::vector<VertexType>& positions, const std::vector<float>& radii)
{
    //    const float radius = 1;

    std::vector<GLfloat> vertices;
    std::vector<GLfloat> texcoords;
    std::vector<GLuint> indices;
    std::vector<GLfloat> glradii;

    vertices.resize(positions.size() * 4 * 3);
    indices.resize(positions.size() * 4);
    texcoords.resize(positions.size() * 4 * 2);
    glradii.resize(positions.size() * 4);
    std::vector<GLfloat>::iterator v = vertices.begin();
    std::vector<GLuint>::iterator i = indices.begin();
    std::vector<GLfloat>::iterator t = texcoords.begin();
    std::vector<GLfloat>::iterator r = glradii.begin();

    //assert positions size and radius ?
    for (unsigned int p = 0; p<positions.size(); p++)
    {
        const VertexType& vertex = positions[p];
        //Should try to avoid this test...
        const float& radius = (p < radii.size()) ? radii[p] : radii[0];

        *v++ = vertex[0];
        *v++ = vertex[1];
        *v++ = vertex[2];

        *v++ = vertex[0];
        *v++ = vertex[1];
        *v++ = vertex[2];

        *v++ = vertex[0];
        *v++ = vertex[1];
        *v++ = vertex[2];

        *v++ = vertex[0];
        *v++ = vertex[1];
        *v++ = vertex[2];

        *t++ = -1.0;
        *t++ = -1.0;

        *t++ = 1.0;
        *t++ = -1.0;

        *t++ = 1.0;
        *t++ = 1.0;

        *t++ = -1.0;
        *t++ = 1.0;

        *i++ = 4 * p + 0;
        *i++ = 4 * p + 1;
        *i++ = 4 * p + 2;
        *i++ = 4 * p + 3;

        *r++ = radius;
        *r++ = radius;
        *r++ = radius;
        *r++ = radius;
    }

    //Generate PositionVBO
    m_buffer.verticesBufferSize = (vertices.size()*sizeof(vertices[0]));
    m_buffer.normalsBufferSize = 0;
    m_buffer.texcoordsBufferSize = (texcoords.size()*sizeof(texcoords[0]));
    m_buffer.totalSize = m_buffer.verticesBufferSize + m_buffer.normalsBufferSize
        + m_buffer.texcoordsBufferSize;

    glBindBuffer(GL_ARRAY_BUFFER, m_buffer.VBO);
    glBufferData(GL_ARRAY_BUFFER,
        m_buffer.totalSize,
        NULL,
        GL_DYNAMIC_DRAW);
    glBufferSubDataARB(GL_ARRAY_BUFFER,
        0,
        m_buffer.verticesBufferSize,
        &(vertices[0]));
    glBufferSubData(GL_ARRAY_BUFFER,
        m_buffer.verticesBufferSize,
        m_buffer.texcoordsBufferSize,
        &(texcoords[0]));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    //Radius buffer
    m_radiusBuffer.bufferSize = glradii.size()*sizeof(glradii[0]);
    glBindBuffer(GL_ARRAY_BUFFER, m_radiusBuffer.VBO);
    glBufferData(GL_ARRAY_BUFFER,
        m_radiusBuffer.bufferSize,
        NULL,
        GL_DYNAMIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER,
        0,
        m_radiusBuffer.bufferSize,
        &(glradii[0]));
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //IBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buffer.IBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(indices[0]), &(indices[0]), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    m_buffer.indicesSize = indices.size();

}

template<class VertexType>
void BasicShapesGL_FakeSphere<VertexType>::beforeDraw()
{
    glMatrixMode(GL_MODELVIEW);

    glBindBuffer(GL_ARRAY_BUFFER, m_radiusBuffer.VBO);
    glVertexAttribPointer(m_radiusBuffer.location, 1, GL_FLOAT, GL_FALSE, 0, (char*)NULL + 0);
    glEnableVertexAttribArray(m_radiusBuffer.location);

    glBindBuffer(GL_ARRAY_BUFFER, m_buffer.VBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buffer.IBO);
    glVertexPointer(3, GL_FLOAT, 0, (char*)NULL + 0);
    glTexCoordPointer(2, GL_FLOAT, 0, (char*)NULL + m_buffer.verticesBufferSize);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

}

template<class VertexType>
void BasicShapesGL_FakeSphere<VertexType>::internalDraw()
{
    m_shader->TurnOn();

    glPushMatrix();
    glDrawElements(GL_QUADS, m_buffer.indicesSize, GL_UNSIGNED_INT, (void*)0);
    glPopMatrix();
    m_shader->TurnOff();
}

template<class VertexType>
void BasicShapesGL_FakeSphere<VertexType>::afterDraw()
{
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableVertexAttribArray(m_radiusBuffer.location);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

}

template<class VertexType>
void BasicShapesGL_FakeSphere<VertexType>::draw(const VertexType& center, const float& radius)
{
    init();
    std::vector<VertexType> centers;
    centers.push_back(center);
    std::vector<float> radii;
    radii.push_back(radius);
    generateBuffer(centers, radii);

    beforeDraw();
    internalDraw();
    afterDraw();

}

template<class VertexType>
void BasicShapesGL_FakeSphere<VertexType>::draw(const helper::vector<VertexType>& centers, const float& radius)
{
    init();
    std::vector<float> radii;
    radii.push_back(radius);
    generateBuffer(centers, radii);

    beforeDraw();
    internalDraw();
    afterDraw();
}

template<class VertexType>
void BasicShapesGL_FakeSphere<VertexType>::draw(const helper::vector<VertexType>& centers, const std::vector<float>& radii)
{
    init();

    generateBuffer(centers, radii);

    beforeDraw();

    internalDraw();

    afterDraw();
}


} // namespace gl

} // namespace helper

} // namespace sofa


#endif // SOFA_HELPER_GL_BASICSHAPESGL_INL
