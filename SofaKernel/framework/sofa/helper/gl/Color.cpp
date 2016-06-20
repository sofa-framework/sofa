/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/gl/Color.h>
#include <sofa/helper/system/gl.h>
#include <cmath>

namespace sofa
{

namespace helper
{

namespace gl
{
Color::Color() {}

Color::~Color() {}

void Color::setHSVA( float h, float s, float v, float a )
{
    float rgba[4];
    getHSVA( rgba, h, s, v, a);
    glColor4fv(rgba);
}

void Color::getHSVA( float* rgba, float h, float s, float v, float a )
{
    // H [0, 360] S, V and A [0.0, 1.0].
    int i = (int)floor(h/60.0f) % 6;
    float f = h/60.0f - floor(h/60.0f);
    float p = v * (float)(1 - s);
    float q = v * (float)(1 - s * f);
    float t = v * (float)(1 - (1 - f) * s);
    rgba[3]=a;
    switch (i)
    {
    case 0: rgba[0]=v; rgba[1]=t; rgba[2]=p;
        break;
    case 1: rgba[0]=q; rgba[1]=v; rgba[2]=p;
        break;
    case 2: rgba[0]=p; rgba[1]=v; rgba[2]=t;
        break;
    case 3: rgba[0]=p; rgba[1]=q; rgba[2]=v;
        break;
    case 4: rgba[0]=t; rgba[1]=p; rgba[2]=v;
        break;
    case 5: rgba[0]=v; rgba[1]=p; rgba[2]=q;
    }
}


} // namespace gl

} // namespace helper

} // namespace sofa

