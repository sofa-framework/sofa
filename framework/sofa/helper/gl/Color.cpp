/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/gl/Color.h>
#include <sofa/helper/system/gl.h>
#include <math.h>

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
    // H [0, 360] S, V and A [0.0, 1.0].
    int i = (int)floor(h/60.0f) % 6;
    float f = h/60.0f - floor(h/60.0f);
    float p = v * (float)(1 - s);
    float q = v * (float)(1 - s * f);
    float t = v * (float)(1 - (1 - f) * s);
    float rgb[4]= {0,0,0,a};
    switch (i)
    {
    case 0: rgb[0]=v; rgb[1]=t; rgb[2]=p;
        break;
    case 1: rgb[0]=q; rgb[1]=v; rgb[2]=p;
        break;
    case 2: rgb[0]=p; rgb[1]=v; rgb[2]=t;
        break;
    case 3: rgb[0]=p; rgb[1]=q; rgb[2]=v;
        break;
    case 4: rgb[0]=t; rgb[1]=p; rgb[2]=v;
        break;
    case 5: rgb[0]=v; rgb[1]=p; rgb[2]=q;

    }
    glColor4fv(rgb);
}


} // namespace gl

} // namespace helper

} // namespace sofa

