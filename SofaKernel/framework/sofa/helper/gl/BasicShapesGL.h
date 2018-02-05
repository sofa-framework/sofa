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
#ifndef SOFA_HELPER_GL_BASICSHAPESGL_H
#define SOFA_HELPER_GL_BASICSHAPESGL_H

#ifndef SOFA_NO_OPENGL

#include <sofa/helper/system/gl.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/gl/GLSLShader.h>
#include <cmath>
#include <map>

namespace sofa
{

namespace helper
{

namespace gl
{

class BasicShapesGL
{
public:
    struct GLBuffers
    {
        GLuint VBO, IBO;
        GLuint verticesBufferSize, normalsBufferSize, texcoordsBufferSize, totalSize;
        GLuint indicesSize;
    };
    struct CustomGLBuffer
    {
        GLuint VBO;
        GLuint bufferSize;
        GLint location;
    };
};

template<class VertexType>
class BasicShapesGL_Sphere : public BasicShapesGL
{
    typedef BasicShapesGL Inherit;
public:
    struct SphereDescription
    {
        SphereDescription(unsigned int r, unsigned int s) : rings(r), sectors(s) {}

        bool operator< (const SphereDescription& d) const
            {
                return this->rings < d.rings || (this->rings == d.rings && this->sectors < d.sectors);
            }

        unsigned int rings;
        unsigned int sectors;
    };

    BasicShapesGL_Sphere();
    virtual ~BasicShapesGL_Sphere();

    void init() {}

    void draw(const VertexType& center, const float& radius, const unsigned int rings = 32, const unsigned int sectors = 16);
    void draw(const helper::vector<VertexType>& centers, const float& radius, const unsigned int rings = 32, const unsigned int sectors = 16);
    void draw(const helper::vector<VertexType>& centers, const std::vector<float>& radius, const unsigned int rings = 32, const unsigned int sectors = 16);

private:
    void generateBuffer(const SphereDescription& desc, GLBuffers& buffer);
    void checkBuffers(const SphereDescription& desc);

    void beforeDraw(const GLBuffers &buffer);
    void internalDraw(const GLBuffers &buffer, const VertexType& center, const float& radius);
    void afterDraw(const GLBuffers &buffer);

    std::map<SphereDescription, GLBuffers> m_mapBuffers;

};


template<class VertexType>
class BasicShapesGL_FakeSphere : public BasicShapesGL
{
    typedef BasicShapesGL Inherit;
public:
    BasicShapesGL_FakeSphere();
    virtual ~BasicShapesGL_FakeSphere();

    void init();

    void draw(const VertexType& center, const float& radius);
    void draw(const helper::vector<VertexType>& centers, const float& radius);
    void draw(const helper::vector<VertexType>& centers, const std::vector<float>& radii);

private:
    void generateBuffer(const std::vector<VertexType> &positions, const std::vector<float>& radii);

    GLBuffers m_buffer;
    CustomGLBuffer m_radiusBuffer;

    GLSLShader* m_shader;
    bool b_isInit;

    void beforeDraw();
    void internalDraw();
    void afterDraw();

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_HELPER_GL_BASICSHAPESGL_CPP)
extern template class SOFA_HELPER_API BasicShapesGL_Sphere<helper::fixed_array< float, 3 > >;
extern template class SOFA_HELPER_API BasicShapesGL_Sphere<helper::fixed_array< double, 3 > >;
extern template class SOFA_HELPER_API BasicShapesGL_FakeSphere<helper::fixed_array< float, 3 > >;
extern template class SOFA_HELPER_API BasicShapesGL_FakeSphere<helper::fixed_array< double, 3 > >;
#endif // defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_HELPER_GL_BASICSHAPESGL_CPP)

} //gl

} //helper

} //sofa

#endif /* SOFA_NO_OPENGL */

#endif // SOFA_HELPER_GL_BASICSHAPESGL_H
