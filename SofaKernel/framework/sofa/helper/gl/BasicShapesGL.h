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
#ifndef SOFA_HELPER_GL_BASICSHAPESGL_H
#define SOFA_HELPER_GL_BASICSHAPESGL_H

#ifndef SOFA_NO_OPENGL

#include <sofa/helper/system/gl.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/vector.h>
#include <cmath>

namespace sofa
{

namespace helper
{

namespace gl
{

template<class VertexType>
class BasicShapesGL_Sphere
{
public:
    //typedef helper::fixed_array<SReal, 3> Vector3;

    GLuint m_VBO, m_IBO;
    GLuint m_normalsBufferSize, m_verticesBufferSize, m_texcoordsBufferSize, m_indicesSize;

    BasicShapesGL_Sphere();
    virtual ~BasicShapesGL_Sphere();

    void init(const unsigned int rings, const unsigned int sectors);
    void draw(const VertexType& center, const float& radius);
    void draw(const helper::vector<VertexType>& centers, const float& radius);
    void draw(const helper::vector<VertexType>& centers, const std::vector<float>& radius);

private:
    void internalDraw(const VertexType& center, const float& radius);


};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_HELPER_GL_BASICSHAPESGL_CPP)
extern template class SOFA_HELPER_API BasicShapesGL_Sphere<helper::fixed_array< float, 3 > >;
extern template class SOFA_HELPER_API BasicShapesGL_Sphere<helper::fixed_array< double, 3 > >;
#endif // defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_HELPER_GL_BASICSHAPESGL_CPP)

} //gl

} //helper

} //sofa

#endif /* SOFA_NO_OPENGL */

#endif // SOFA_HELPER_GL_BASICSHAPESGL_H
