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
#include <sofa/gl/gl.h>
#include <sofa/type/fixed_array.h>
#include <sofa/type/vector.h>
#include <sofa/gl/GLSLShader.h>
#include <cmath>
#include <map>

namespace sofa::gl
{

class BasicShapesGL
{
public:
    struct GLBuffers
    {
        GLuint VBO, IBO;
        GLint64 verticesBufferSize, normalsBufferSize, texcoordsBufferSize, totalSize;
        GLint64 indicesSize;
    };
    struct CustomGLBuffer
    {
        GLuint VBO;
        GLint64 bufferSize;
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
    void draw(const type::vector<VertexType>& centers, const float& radius, const unsigned int rings = 32, const unsigned int sectors = 16);
    void draw(const type::vector<VertexType>& centers, const std::vector<float>& radius, const unsigned int rings = 32, const unsigned int sectors = 16);

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
    void draw(const type::vector<VertexType>& centers, const float& radius);
    void draw(const type::vector<VertexType>& centers, const std::vector<float>& radii);

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

#if !defined(SOFA_HELPER_GL_BASICSHAPESGL_CPP)
extern template class SOFA_GL_API BasicShapesGL_Sphere<type::fixed_array< float, 3 > >;
extern template class SOFA_GL_API BasicShapesGL_Sphere<type::fixed_array< double, 3 > >;
extern template class SOFA_GL_API BasicShapesGL_FakeSphere<type::fixed_array< float, 3 > >;
extern template class SOFA_GL_API BasicShapesGL_FakeSphere<type::fixed_array< double, 3 > >;
#endif //  !defined(SOFA_HELPER_GL_BASICSHAPESGL_CPP)

} // namespace sofa::gl
