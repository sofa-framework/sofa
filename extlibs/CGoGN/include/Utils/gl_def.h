/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#ifndef __CGOGN_GL_DEF_H_
#define __CGOGN_GL_DEF_H_

#ifdef USE_VRJUGGLER
#include <vrj/Draw/OGL/GlApp.h>
#include <vrj/Draw/OGL/GlContextData.h>
#endif

#include <GL/glew.h>

#include "Utils/cgognStream.h"

namespace CGoGN
{

#ifdef USE_VRJUGGLER

#include <vrj/Draw/OGL/GlApp.h>
#include <vrj/Draw/OGL/GlContextData.h>
typedef vrj::opengl::ContextData<GLint> CGoGNGLint;
typedef vrj::opengl::ContextData<GLuint> CGoGNGLuint;
typedef vrj::opengl::ContextData<GLhandleARB> CGoGNGLhandleARB;
typedef vrj::opengl::ContextData<GLenum> CGoGNGLenum;
typedef vrj::opengl::ContextData<GLenum*> CGoGNGLenumTable;
#else

template <typename T>
class FalsePtr
{
        T m_v;
public:
        FalsePtr() :m_v(T(0)) {}
        FalsePtr(const T& v) : m_v(v) {}
        T& operator*() { return m_v; }
        const T& operator*() const { return m_v; }
};

typedef FalsePtr<GLint> CGoGNGLint;
typedef FalsePtr<GLuint> CGoGNGLuint;
typedef FalsePtr<GLuint> CGoGNGLhandleARB;
typedef FalsePtr<GLenum> CGoGNGLenum;
typedef FalsePtr<GLenum*> CGoGNGLenumTable;

#endif


#ifdef MAC_OSX

inline void glCheckErrors()
{
	GLenum glError = glGetError();
	if (glError != GL_NO_ERROR)
		CGoGNerr<<"GL error: " << glError << CGoGNendl;
}

inline void glCheckErrors(const std::string& message)
{
	GLenum glError = glGetError();
	if (glError != GL_NO_ERROR)
		CGoGNerr<< message <<" : " << glError << CGoGNendl;
}

#else
inline void glCheckErrors()
{
	GLenum glError = glGetError();
	if (glError != GL_NO_ERROR)
		CGoGNerr<<"GL error: " << gluErrorString(glError) << CGoGNendl;
}

inline void glCheckErrors(const std::string& message)
{
	GLenum glError = glGetError();
	if (glError != GL_NO_ERROR)
		CGoGNerr<< message <<" : " << gluErrorString(glError) << CGoGNendl;
}
#endif



}


#endif /* GL_DEF_H_ */
