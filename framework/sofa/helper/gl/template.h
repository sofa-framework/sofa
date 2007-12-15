/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
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
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_HELPER_GL_TEMPLATE_H
#define SOFA_HELPER_GL_TEMPLATE_H

#include <sofa/helper/system/gl.h>

namespace sofa
{

namespace helper
{

namespace gl
{

template<int N>
inline void glVertexNv(const float* p)
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
    glVertexNv<Coord::static_size>(c.ptr());
}

template<>
inline void glVertexT<double>(const double& c)
{
    glVertex3d(c,0,0);
}

template<>
inline void glVertexT<float>(const float& c)
{
    glVertex3d(c,0,0);
}

} // namespace gl

} // namespace helper

} // namespace sofa

#endif
