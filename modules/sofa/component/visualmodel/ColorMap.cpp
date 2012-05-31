/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_COMPONENT_VISUALMODEL_COLORMAP_CPP

#include <sofa/component/visualmodel/ColorMap.h>
#include <sofa/helper/rmath.h>
#include <string>
#include <iostream>

namespace sofa
{

namespace component
{

namespace visualmodel
{

enum { NDefaultColorMapEntries = 64 };
static ColorMap::Color DefaultColorMapEntries[NDefaultColorMapEntries] =
{
    ColorMap::Color( 0.0f,        0.0f,       0.5625f, 1.0f ),
    ColorMap::Color( 0.0f,        0.0f,        0.625f, 1.0f ),
    ColorMap::Color( 0.0f,        0.0f,       0.6875f, 1.0f ),
    ColorMap::Color( 0.0f,        0.0f,         0.75f, 1.0f ),
    ColorMap::Color( 0.0f,        0.0f,       0.8125f, 1.0f ),
    ColorMap::Color( 0.0f,        0.0f,        0.875f, 1.0f ),
    ColorMap::Color( 0.0f,        0.0f,       0.9375f, 1.0f ),
    ColorMap::Color( 0.0f,        0.0f,          1.0f, 1.0f ),
    ColorMap::Color( 0.0f,     0.0625f,          1.0f, 1.0f ),
    ColorMap::Color( 0.0f,      0.125f,          1.0f, 1.0f ),
    ColorMap::Color( 0.0f,     0.1875f,          1.0f, 1.0f ),
    ColorMap::Color( 0.0f,       0.25f,          1.0f, 1.0f ),
    ColorMap::Color( 0.0f,     0.3125f,          1.0f, 1.0f ),
    ColorMap::Color( 0.0f,      0.375f,          1.0f, 1.0f ),
    ColorMap::Color( 0.0f,     0.4375f,          1.0f, 1.0f ),
    ColorMap::Color( 0.0f,        0.5f,          1.0f, 1.0f ),
    ColorMap::Color( 0.0f,     0.5625f,          1.0f, 1.0f ),
    ColorMap::Color( 0.0f,      0.625f,          1.0f, 1.0f ),
    ColorMap::Color( 0.0f,     0.6875f,          1.0f, 1.0f ),
    ColorMap::Color( 0.0f,       0.75f,          1.0f, 1.0f ),
    ColorMap::Color( 0.0f,     0.8125f,          1.0f, 1.0f ),
    ColorMap::Color( 0.0f,     0.875f,           1.0f, 1.0f ),
    ColorMap::Color( 0.0f,     0.9375f,          1.0f, 1.0f ),
    ColorMap::Color( 0.0f,        1.0f,          1.0f, 1.0f ),
    ColorMap::Color( 0.0625f,     1.0f,          1.0f, 1.0f ),
    ColorMap::Color( 0.125f,      1.0f,       0.9375f, 1.0f ),
    ColorMap::Color( 0.1875f,     1.0f,        0.875f, 1.0f ),
    ColorMap::Color( 0.25f,       1.0f,       0.8125f, 1.0f ),
    ColorMap::Color( 0.3125f,     1.0f,         0.75f, 1.0f ),
    ColorMap::Color( 0.375f,      1.0f,       0.6875f, 1.0f ),
    ColorMap::Color( 0.4375f,     1.0f,        0.625f, 1.0f ),
    ColorMap::Color( 0.5f,        1.0f,       0.5625f, 1.0f ),
    ColorMap::Color( 0.5625f,     1.0f,          0.5f, 1.0f ),
    ColorMap::Color( 0.625f,      1.0f,       0.4375f, 1.0f ),
    ColorMap::Color( 0.6875f,     1.0f,        0.375f, 1.0f ),
    ColorMap::Color( 0.75f,       1.0f,       0.3125f, 1.0f ),
    ColorMap::Color( 0.8125f,     1.0f,         0.25f, 1.0f ),
    ColorMap::Color( 0.875f,      1.0f,       0.1875f, 1.0f ),
    ColorMap::Color( 0.9375f,     1.0f,        0.125f, 1.0f ),
    ColorMap::Color( 1.0f,        1.0f,       0.0625f, 1.0f ),
    ColorMap::Color( 1.0f,        1.0f,          0.0f, 1.0f ),
    ColorMap::Color( 1.0f,       0.9375f,        0.0f, 1.0f ),
    ColorMap::Color( 1.0f,        0.875f,        0.0f, 1.0f ),
    ColorMap::Color( 1.0f,       0.8125f,        0.0f, 1.0f ),
    ColorMap::Color( 1.0f,         0.75f,        0.0f, 1.0f ),
    ColorMap::Color( 1.0f,       0.6875f,        0.0f, 1.0f ),
    ColorMap::Color( 1.0f,        0.625f,        0.0f, 1.0f ),
    ColorMap::Color( 1.0f,       0.5625f,        0.0f, 1.0f ),
    ColorMap::Color( 1.0f,          0.5f,        0.0f, 1.0f ),
    ColorMap::Color( 1.0f,       0.4375f,        0.0f, 1.0f ),
    ColorMap::Color( 1.0f,        0.375f,        0.0f, 1.0f ),
    ColorMap::Color( 1.0f,       0.3125f,        0.0f, 1.0f ),
    ColorMap::Color( 1.0f,         0.25f,        0.0f, 1.0f ),
    ColorMap::Color( 1.0f,       0.1875f,        0.0f, 1.0f ),
    ColorMap::Color( 1.0f,        0.125f,        0.0f, 1.0f ),
    ColorMap::Color( 1.0f,       0.0625f,        0.0f, 1.0f ),
    ColorMap::Color( 1.0f,          0.0f,        0.0f, 1.0f ),
    ColorMap::Color( 0.9375f,       0.0f,        0.0f, 1.0f ),
    ColorMap::Color( 0.875f,        0.0f,        0.0f, 1.0f ),
    ColorMap::Color( 0.8125f,       0.0f,        0.0f, 1.0f ),
    ColorMap::Color( 0.75f,         0.0f,        0.0f, 1.0f ),
    ColorMap::Color( 0.6875f,       0.0f,        0.0f, 1.0f ),
    ColorMap::Color( 0.625f,        0.0f,        0.0f, 1.0f ),
    ColorMap::Color( 0.5625f,       0.0f,        0.0f, 1.0f )
};

ColorMap::ColorMap(const std::string& name)
    : name(name)
{
    entries.insert(entries.end(), DefaultColorMapEntries, DefaultColorMapEntries+NDefaultColorMapEntries);
}

ColorMap* ColorMap::getDefault()
{
    static ColorMap defaultColorMap("default");
    return &defaultColorMap;
}

} // namespace visualmodel

} // namespace component

} // namespace sofa
