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
#include <sofa/helper/config.h>

#if __has_include(<sofa/gl/template.h>)
#include <sofa/gl/template.h>
#define GL_TEMPLATE_ENABLE_WRAPPER

SOFA_DEPRECATED_HEADER(v21.06, "sofa/gl/template.h")

#else
#error "OpenGL headers have been moved to Sofa.GL; you will need to link against this library if you need OpenGL, and include <sofa/gl/template.h> instead of this one."
#endif

#ifdef GL_TEMPLATE_ENABLE_WRAPPER
namespace sofa::helper::gl
{
    
template<int N>
inline void glVertexNv(const float* /*p*/)
{
}

template<>
inline void glVertexNv<3>(const float* p)
{
    sofa::gl::glVertexNv<3>(p);
}

template<>
inline void glVertexNv<2>(const float* p)
{
    sofa::gl::glVertexNv<2>(p);
}

template<>
inline void glVertexNv<1>(const float* p)
{
    sofa::gl::glVertexNv<1>(p);
}

template<int N>
inline void glVertexNv(const double* p)
{
    sofa::gl::glVertexNv<N>(p);
}

template<>
inline void glVertexNv<2>(const double* p)
{
    sofa::gl::glVertexNv<2>(p);
}

template<>
inline void glVertexNv<1>(const double* p)
{
    sofa::gl::glVertexNv<1>(p);
}

template<class Coord>
inline void glVertexT(const Coord& c)
{
    sofa::gl::glVertexT<Coord>(c);
}

template<>
inline void glVertexT<double>(const double& c)
{
    sofa::gl::glVertexT<double>(c);
}

template<>
inline void glVertexT<float>(const float& c)
{
    sofa::gl::glVertexT<float>(c);
}


////////////////////////////////////////

template<int N>
inline void glTexCoordNv(const float* /*p*/)
{
}

template<>
inline void glTexCoordNv<3>(const float* p)
{
    sofa::gl::glTexCoordNv<3>(p);
}

template<>
inline void glTexCoordNv<2>(const float* p)
{
    sofa::gl::glTexCoordNv<2>(p);
}

template<>
inline void glTexCoordNv<1>(const float* p)
{
    sofa::gl::glTexCoordNv<1>(p);
}

template<int N>
inline void glTexCoordNv(const double* p)
{
    sofa::gl::glTexCoordNv<N>(p);
}

template<>
inline void glTexCoordNv<2>(const double* p)
{
    sofa::gl::glTexCoordNv<2>(p);
}

template<>
inline void glTexCoordNv<1>(const double* p)
{
    sofa::gl::glTexCoordNv<1>(p);
}

template<class Coord>
inline void glTexCoordT(const Coord& c)
{
    sofa::gl::glTexCoordT<Coord>(c);
}

template<>
inline void glTexCoordT<double>(const double& c)
{
    sofa::gl::glTexCoordT<double>(c);
}

template<>
inline void glTexCoordT<float>(const float& c)
{
    sofa::gl::glTexCoordT<float>(c);
}



///////////////////////////////////////

template<int N>
inline void glNormalNv(const float* p)
{
    sofa::gl::glNormalNv<N>(p);
}

template<>
inline void glNormalNv<2>(const float* p)
{
    sofa::gl::glNormalNv<2>(p);
}

template<>
inline void glNormalNv<1>(const float* p)
{
    sofa::gl::glNormalNv<1>(p);
}

template<int N>
inline void glNormalNv(const double* p)
{
    sofa::gl::glNormalNv<N>(p);
}

template<>
inline void glNormalNv<2>(const double* p)
{
    sofa::gl::glNormalNv<2>(p);
}

template<>
inline void glNormalNv<1>(const double* p)
{
    sofa::gl::glNormalNv<1>(p);
}

template<class Coord>
inline void glNormalT(const Coord& c)
{
    sofa::gl::glNormalNv<Coord::static_size>(c.ptr());
}

template<>
inline void glNormalT<double>(const double& c)
{
    sofa::gl::glNormalT<double>(c);
}

template<>
inline void glNormalT<float>(const float& c)
{
    sofa::gl::glNormalT<float>(c);
}
////////
inline void glTranslate(const float& c1, const float& c2, const float& c3)
{
    sofa::gl::glTranslate(c1, c2, c3);
}

inline void glTranslate(const double& c1, const double& c2, const double& c3)
{
    sofa::gl::glTranslate(c1, c2, c3);
}

template<int N>
inline void glTranslateNv(const float* p)
{
    sofa::gl::glTranslateNv<N>(p);
}

template<>
inline void glTranslateNv<2>(const float* p)
{
    sofa::gl::glTranslateNv<2>(p);
}

template<>
inline void glTranslateNv<1>(const float* p)
{
    sofa::gl::glTranslateNv<1>(p);
}

template<int N>
inline void glTranslateNv(const double* p)
{
    sofa::gl::glTranslateNv<N>(p);
}

template<>
inline void glTranslateNv<2>(const double* p)
{
    sofa::gl::glTranslateNv<2>(p);
}

template<>
inline void glTranslateNv<1>(const double* p)
{
    sofa::gl::glTranslateNv<1>(p);
}

template<class Coord>
inline void glTranslateT(const Coord& c)
{
    sofa::gl::glTranslateT<Coord>(c);
}

template<>
inline void glTranslateT<double>(const double& c)
{
    sofa::gl::glTranslateT<double>(c);
}

template<>
inline void glTranslateT<float>(const float& c)
{
    sofa::gl::glTranslateT<float>(c);
}


////////////




inline void glScale(const float& c1, const float& c2, const float& c3)
{
    sofa::gl::glScale(c1, c2, c3);
}

inline void glScale(const double& c1, const double& c2, const double& c3)
{
    sofa::gl::glScale(c1, c2, c3);
}

inline void glRotate(const GLfloat &value, const float& c1, const float& c2, const float& c3)
{
    sofa::gl::glRotate(value, c1, c2, c3);
}

inline void glRotate(const GLdouble &value, const double& c1, const double& c2, const double& c3)
{
    sofa::gl::glRotate(value, c1, c2, c3);
}

inline void glMultMatrix(const float* p)
{
    sofa::gl::glMultMatrix(p);
}

inline void glMultMatrix(const double* p)
{
    sofa::gl::glMultMatrix(p);
}

}

#endif // GL_TEMPLATE_ENABLE_WRAPPER

#undef GL_TEMPLATE_ENABLE_WRAPPER
