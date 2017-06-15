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
#ifndef SOFA_HELPER_GL_TEMPLATE_H
#define SOFA_HELPER_GL_TEMPLATE_H

#ifndef SOFA_NO_OPENGL

#include <sofa/helper/system/gl.h>
#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

namespace gl
{

template<int N>
inline void glVertexNv(const float* /*p*/)
{
}

template<>
inline void glVertexNv<3>(const float* p)
{
	glVertex3f(p[0],p[1],p[2]);
}

template<>
inline void glVertexNv<2>(const float* p)
{
    glVertex2f(p[0],p[1]);
}

template<>
inline void glVertexNv<1>(const float* p)
{
    glVertex2f(p[0],0.0f);
}

template<int N>
inline void glVertexNv(const double* p)
{
    glVertex3d(p[0],p[1],p[2]);
}

template<>
inline void glVertexNv<2>(const double* p)
{
    glVertex2d(p[0],p[1]);
}

template<>
inline void glVertexNv<1>(const double* p)
{
    glVertex2d(p[0],0.0);
}

template<class Coord>
inline void glVertexT(const Coord& c)
{
    glVertexNv<Coord::spatial_dimensions>(c.ptr());
}


template<>
inline void glVertexT<double>(const double& c)
{
    glVertex3d(c,0.0,0.0);
}

template<>
inline void glVertexT<float>(const float& c)
{
    glVertex3f(c,0.0f,0.0f);
}


////////////////////////////////////////

template<int N>
inline void glTexCoordNv(const float* /*p*/)
{
}

template<>
inline void glTexCoordNv<3>(const float* p)
{
    glTexCoord3f(p[0],p[1],p[2]);
}

template<>
inline void glTexCoordNv<2>(const float* p)
{
    glTexCoord2f(p[0],p[1]);
}

template<>
inline void glTexCoordNv<1>(const float* p)
{
    glTexCoord2f(p[0],0.0f);
}

template<int N>
inline void glTexCoordNv(const double* p)
{
    glTexCoord3d(p[0],p[1],p[2]);
}

template<>
inline void glTexCoordNv<2>(const double* p)
{
    glTexCoord2d(p[0],p[1]);
}

template<>
inline void glTexCoordNv<1>(const double* p)
{
    glTexCoord2d(p[0],0.0);
}

template<class Coord>
inline void glTexCoordT(const Coord& c)
{
    glTexCoordNv<Coord::static_size>(c.ptr());
}

template<>
inline void glTexCoordT<double>(const double& c)
{
    glTexCoord3d(c,0.0,0.0);
}

template<>
inline void glTexCoordT<float>(const float& c)
{
    glTexCoord3f(c,0.0f,0.0f);
}



///////////////////////////////////////

template<int N>
inline void glNormalNv(const float* p)
{
    glNormal3f(p[0],p[1],p[2]);
}

template<>
inline void glNormalNv<2>(const float* p)
{
    glNormal3f(p[0],p[1],0.0f);
}

template<>
inline void glNormalNv<1>(const float* p)
{
    glNormal3f(p[0],0.0f,0.0f);
}

template<int N>
inline void glNormalNv(const double* p)
{
    glNormal3d(p[0],p[1],p[2]);
}

template<>
inline void glNormalNv<2>(const double* p)
{
    glNormal3d(p[0],p[1],0.0);
}

template<>
inline void glNormalNv<1>(const double* p)
{
    glNormal3d(p[0],0.0,0.0);
}

template<class Coord>
inline void glNormalT(const Coord& c)
{
    glNormalNv<Coord::static_size>(c.ptr());
}

template<>
inline void glNormalT<double>(const double& c)
{
    glNormal3d(c,0.0,0.0);
}

template<>
inline void glNormalT<float>(const float& c)
{
    glNormal3f(c,0.0f,0.0f);
}
////////
inline void glTranslate(const float& c1, const float& c2, const float& c3)
{
    glTranslatef(c1, c2, c3);
}

inline void glTranslate(const double& c1, const double& c2, const double& c3)
{
    glTranslated(c1, c2, c3);
}

template<int N>
inline void glTranslateNv(const float* p)
{
    glTranslatef(p[0],p[1],p[2]);
}

template<>
inline void glTranslateNv<2>(const float* p)
{
    glTranslatef(p[0],p[1],0.0f);
}

template<>
inline void glTranslateNv<1>(const float* p)
{
    glTranslatef(p[0],0.0f,0.0f);
}

template<int N>
inline void glTranslateNv(const double* p)
{
    glTranslated(p[0],p[1],p[2]);
}

template<>
inline void glTranslateNv<2>(const double* p)
{
    glTranslated(p[0],p[1],0.0);
}

template<>
inline void glTranslateNv<1>(const double* p)
{
    glTranslated(p[0],0.0,0.0);
}

template<class Coord>
inline void glTranslateT(const Coord& c)
{
	//
    //glTranslateNv<Coord::spatial_dimensions>(c.ptr());
	glTranslateNv<Coord::static_size>(c.data());
}

template<>
inline void glTranslateT<double>(const double& c)
{
    glTranslated(c,0.0,0.0);
}

template<>
inline void glTranslateT<float>(const float& c)
{
    glTranslatef(c,0.0f,0.0f);
}


////////////




inline void glScale(const float& c1, const float& c2, const float& c3)
{
    glScalef(c1, c2, c3);
}

inline void glScale(const double& c1, const double& c2, const double& c3)
{
    glScaled(c1, c2, c3);
}

inline void glRotate(const GLfloat &value, const float& c1, const float& c2, const float& c3)
{
    glRotatef(value, c1, c2, c3);
}

inline void glRotate(const GLdouble &value, const double& c1, const double& c2, const double& c3)
{
    glRotated(value, c1, c2, c3);
}

inline void glMultMatrix(const float* p)
{
    glMultMatrixf(p);
}

inline void glMultMatrix(const double* p)
{
#ifdef PS3
	float f[16];
	
	for(int i=0; i<16; i++)
	{
		f[i] = (float) p[i];
	}
	glMultMatrixf(f);
#else
    glMultMatrixd(p);
#endif
}



} // namespace gl

} // namespace helper

} // namespace sofa

#endif /* SOFA_NO_OPENGL */

#endif
