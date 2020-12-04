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

#if __has_include(<sofa/gl/BasicShapes.h>)
#include <sofa/gl/BasicShapes.h>
#define GL_BASICSHAPES_ENABLE_WRAPPER

SOFA_DEPRECATED_HEADER(v21.06, "sofa/gl/BasicShapes.h")

#else
#error "OpenGL headers have been moved to Sofa.GL; you will need to link against this library if you need OpenGL, and include <sofa/gl/BasicShapes.h> instead of this one."
#endif

#ifdef GL_BASICSHAPES_ENABLE_WRAPPER
#include <sofa/helper/fixed_array.h>

namespace sofa::helper::gl
{
template <typename V>
void drawCone(const V& p1, const V& p2, const float& radius1, const float& radius2, const int subd = 8)
{
    sofa::gl::drawCone<V>(p1, p2, radius1, radius2, subd);
}


template <typename V>
void drawCylinder(const V& p1, const V& p2, const float& rad, const int subd = 8)
{
    sofa::gl::drawCylinder<V>(p1, p2, rad, subd);
}


template <typename V>
void drawArrow(const V& p1, const V& p2, const float& rad, const int subd = 8)
{
    sofa::gl::drawArrow<V>(p1, p2, rad, subd);
}


template <typename V>
void drawSphere(const V& center, const float& rad, const int subd1 = 8, const int subd2 = 8)
{
    sofa::gl::drawSphere<V>(center, rad, subd1, subd2);
}

template <typename V>
void drawEllipsoid(const V& center, const float& radx, const float& rady, const float& radz, const int subd1 = 8, const int subd2 = 8)
{
    sofa::gl::drawEllipsoid<V>(center, radx, rady, subd1, subd2);
}

template <typename V>
void drawWireSphere(const V& center, const float& rad, const int subd1 = 8, const int subd2 = 8)
{
    sofa::gl::drawWireSphere<V>(center, rad, subd1, subd2);
}

template <typename V>
void drawTorus(const float* coordinateMatrix, const float& bodyRad = 0.0, const float& rad = 1.0, const int precision = 20,
    const V& color = sofa::helper::fixed_array<int, 3>(255, 215, 180))
{
    sofa::gl::drawTorus<V>(coordinateMatrix, bodyRad, rad, precision, color);
}

template <typename V>
void drawEmptyParallelepiped(const V& vert1, const V& vert2, const V& vert3, const V& vert4, const V& vecFromFaceToOppositeFace, const float& rad = 1.0, const int precision = 8,
    const V& color = sofa::helper::fixed_array<int, 3>(255, 0, 0))
{
    sofa::gl::drawEmptyParallelepiped<V>(vert1, vert2, vert3, vert4, vecFromFaceToOppositeFace, rad, precision, color);
}

} // namespace sofa::helper::gl

#endif // GL_BASICSHAPES_ENABLE_WRAPPER

#undef GL_BASICSHAPES_ENABLE_WRAPPER
