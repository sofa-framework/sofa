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

#include <sofa/helper/ColorMap.h>
#include <sofa/helper/logging/Messaging.h>
#include <string>
#include <iostream>

namespace sofa
{

namespace helper
{

//enum { NDefaultColorMapEntries = 64 };
//static ColorMap::Color DefaultColorMapEntries[NDefaultColorMapEntries] =
//{
//    ColorMap::Color( 0.0f,        0.0f,       0.5625f, 1.0f ),
//    ColorMap::Color( 0.0f,        0.0f,        0.625f, 1.0f ),
//    ColorMap::Color( 0.0f,        0.0f,       0.6875f, 1.0f ),
//    ColorMap::Color( 0.0f,        0.0f,         0.75f, 1.0f ),
//    ColorMap::Color( 0.0f,        0.0f,       0.8125f, 1.0f ),
//    ColorMap::Color( 0.0f,        0.0f,        0.875f, 1.0f ),
//    ColorMap::Color( 0.0f,        0.0f,       0.9375f, 1.0f ),
//    ColorMap::Color( 0.0f,        0.0f,          1.0f, 1.0f ),
//    ColorMap::Color( 0.0f,     0.0625f,          1.0f, 1.0f ),
//    ColorMap::Color( 0.0f,      0.125f,          1.0f, 1.0f ),
//    ColorMap::Color( 0.0f,     0.1875f,          1.0f, 1.0f ),
//    ColorMap::Color( 0.0f,       0.25f,          1.0f, 1.0f ),
//    ColorMap::Color( 0.0f,     0.3125f,          1.0f, 1.0f ),
//    ColorMap::Color( 0.0f,      0.375f,          1.0f, 1.0f ),
//    ColorMap::Color( 0.0f,     0.4375f,          1.0f, 1.0f ),
//    ColorMap::Color( 0.0f,        0.5f,          1.0f, 1.0f ),
//    ColorMap::Color( 0.0f,     0.5625f,          1.0f, 1.0f ),
//    ColorMap::Color( 0.0f,      0.625f,          1.0f, 1.0f ),
//    ColorMap::Color( 0.0f,     0.6875f,          1.0f, 1.0f ),
//    ColorMap::Color( 0.0f,       0.75f,          1.0f, 1.0f ),
//    ColorMap::Color( 0.0f,     0.8125f,          1.0f, 1.0f ),
//    ColorMap::Color( 0.0f,     0.875f,           1.0f, 1.0f ),
//    ColorMap::Color( 0.0f,     0.9375f,          1.0f, 1.0f ),
//    ColorMap::Color( 0.0f,        1.0f,          1.0f, 1.0f ),
//    ColorMap::Color( 0.0625f,     1.0f,          1.0f, 1.0f ),
//    ColorMap::Color( 0.125f,      1.0f,       0.9375f, 1.0f ),
//    ColorMap::Color( 0.1875f,     1.0f,        0.875f, 1.0f ),
//    ColorMap::Color( 0.25f,       1.0f,       0.8125f, 1.0f ),
//    ColorMap::Color( 0.3125f,     1.0f,         0.75f, 1.0f ),
//    ColorMap::Color( 0.375f,      1.0f,       0.6875f, 1.0f ),
//    ColorMap::Color( 0.4375f,     1.0f,        0.625f, 1.0f ),
//    ColorMap::Color( 0.5f,        1.0f,       0.5625f, 1.0f ),
//    ColorMap::Color( 0.5625f,     1.0f,          0.5f, 1.0f ),
//    ColorMap::Color( 0.625f,      1.0f,       0.4375f, 1.0f ),
//    ColorMap::Color( 0.6875f,     1.0f,        0.375f, 1.0f ),
//    ColorMap::Color( 0.75f,       1.0f,       0.3125f, 1.0f ),
//    ColorMap::Color( 0.8125f,     1.0f,         0.25f, 1.0f ),
//    ColorMap::Color( 0.875f,      1.0f,       0.1875f, 1.0f ),
//    ColorMap::Color( 0.9375f,     1.0f,        0.125f, 1.0f ),
//    ColorMap::Color( 1.0f,        1.0f,       0.0625f, 1.0f ),
//    ColorMap::Color( 1.0f,        1.0f,          0.0f, 1.0f ),
//    ColorMap::Color( 1.0f,       0.9375f,        0.0f, 1.0f ),
//    ColorMap::Color( 1.0f,        0.875f,        0.0f, 1.0f ),
//    ColorMap::Color( 1.0f,       0.8125f,        0.0f, 1.0f ),
//    ColorMap::Color( 1.0f,         0.75f,        0.0f, 1.0f ),
//    ColorMap::Color( 1.0f,       0.6875f,        0.0f, 1.0f ),
//    ColorMap::Color( 1.0f,        0.625f,        0.0f, 1.0f ),
//    ColorMap::Color( 1.0f,       0.5625f,        0.0f, 1.0f ),
//    ColorMap::Color( 1.0f,          0.5f,        0.0f, 1.0f ),
//    ColorMap::Color( 1.0f,       0.4375f,        0.0f, 1.0f ),
//    ColorMap::Color( 1.0f,        0.375f,        0.0f, 1.0f ),
//    ColorMap::Color( 1.0f,       0.3125f,        0.0f, 1.0f ),
//    ColorMap::Color( 1.0f,         0.25f,        0.0f, 1.0f ),
//    ColorMap::Color( 1.0f,       0.1875f,        0.0f, 1.0f ),
//    ColorMap::Color( 1.0f,        0.125f,        0.0f, 1.0f ),
//    ColorMap::Color( 1.0f,       0.0625f,        0.0f, 1.0f ),
//    ColorMap::Color( 1.0f,          0.0f,        0.0f, 1.0f ),
//    ColorMap::Color( 0.9375f,       0.0f,        0.0f, 1.0f ),
//    ColorMap::Color( 0.875f,        0.0f,        0.0f, 1.0f ),
//    ColorMap::Color( 0.8125f,       0.0f,        0.0f, 1.0f ),
//    ColorMap::Color( 0.75f,         0.0f,        0.0f, 1.0f ),
//    ColorMap::Color( 0.6875f,       0.0f,        0.0f, 1.0f ),
//    ColorMap::Color( 0.625f,        0.0f,        0.0f, 1.0f ),
//    ColorMap::Color( 0.5625f,       0.0f,        0.0f, 1.0f )
//};

enum { NDefaultColorMapSchemes = 20 };
static std::string DefaultColorSchemes[NDefaultColorMapSchemes] =
{
    "Red to Blue",  // HSV space
    "Blue to Red",  // HSV space
    "HSV",          // HSV space
    "Red",          // RGB space
    "Green",        // RGB space
    "Blue",         // RGB space
    "Yellow to Cyan",// HSV space
    "Cyan to Yellow",// HSV space
    "Red to Yellow",// RGB space
    "Yellow to Red",// RGB space
    "Yellow to Green",// RGB space
    "Green to Yellow",// RGB space
    "Green to Cyan",// RGB space
    "Cyan to Green",// RGB space
    "Cyan to Blue",// RGB space
    "Blue to Cyan",// RGB space
    "BlueInv",// HSV space
    "GreenInv",// HSV space
    "RedInv",// HSV space
    "Custom"// TODO: Custom colors
};

ColorMap* ColorMap::getDefault()
{
    static ColorMap* defaultColorMap;
    if (defaultColorMap == nullptr) {
        defaultColorMap = new ColorMap();
        std::string tmp("");
        defaultColorMap->init();
    }
    return defaultColorMap;
}

ColorMap::ColorMap(unsigned int paletteSize, const std::string& colorScheme)
: m_paletteSize(paletteSize)
, m_colorScheme(colorScheme)
{
    init();
}

ColorMap::~ColorMap()
{

}

void ColorMap::init()
{
    reinit();
}


void ColorMap::reinit()
{
    entries.clear();

    unsigned int nColors = m_paletteSize;
    if (nColors < 2) {
        msg_warning("ColorMap") << "Palette size must be greater than or equal to 2." << msgendl
                                << "Using the default value of '2'. ";
        m_paletteSize = 2;
        nColors = 2;
    }

    const std::string scheme = m_colorScheme;
    if (scheme == "Custom") {
        // TODO
    } else if (scheme == "Red to Blue") {
        // List the colors
        const float step = (2.0f/3.0f)/(nColors-1);
        for (unsigned int i=0; i<nColors; i++)
        {
            const auto h = i * step * 360;
            entries.push_back(type::RGBAColor::fromHSVA(h, 1.0f, 1.0f, 1.0f));
        }
    } else if (scheme == "Blue to Red") {
        // List the colors
        const float step = (2.0f/3.0f)/(nColors-1);
        for (unsigned int i=0; i<nColors; i++)
        {
            const auto h = (2.0f / 3.0f - i * step) * 360;
            entries.push_back(type::RGBAColor::fromHSVA(h, 1.0f, 1.0f, 1.0f));
        }
    } else if (scheme == "Yellow to Cyan") {
        // List the colors
        const float step = (0.5f - 1.0f/6.0f)/(nColors-1);
        for (unsigned int i=0; i<nColors; i++)
        {
            const auto h = (1.0f / 6.0f + i * step) * 360;
            entries.push_back(type::RGBAColor::fromHSVA(h, 1.0f, 1.0f, 1.0f));
        }
    } else if (scheme == "Cyan to Yellow") {
        // List the colors
        const float step = (0.5f - 1.0f/6.0f)/(nColors-1);
        for (unsigned int i=0; i<nColors; i++)
        {
            const auto h = (0.5f - i * step) * 360;
            entries.push_back(type::RGBAColor::fromHSVA(h, 1.0f, 1.0f, 1.0f));
        }
    } else if (scheme == "Red to Yellow") {
        const float step = 1.0f/(nColors);
        for (unsigned int i=0; i<nColors; i++)
        {
            entries.push_back(type::RGBAColor(1.0f, i*step, 0.0f, 1.0f));
        }
    } else if (scheme == "Yellow to Red") {
        const float step = 1.0f/(nColors);
        for (unsigned int i=0; i<nColors; i++)
        {
            entries.push_back(type::RGBAColor(1.0f, 1.0f-i*step, 0.0f, 1.0f));
        }
    } else if (scheme == "Yellow to Green") {
        const float step = 1.0f/(nColors);
        for (unsigned int i=0; i<nColors; i++)
        {
            entries.push_back(type::RGBAColor(1.0f-i*step, 1.0f, 0.0f, 1.0f));
        }
    } else if (scheme == "Green to Yellow") {
        const float step = 1.0f/(nColors);
        for (unsigned int i=0; i<nColors; i++)
        {
            entries.push_back(type::RGBAColor(i*step, 1.0f, 0.0f, 1.0f));
        }
    } else if (scheme == "Green to Cyan") {
        const float step = 1.0f/(nColors);
        for (unsigned int i=0; i<nColors; i++)
        {
            entries.push_back(type::RGBAColor(0.0f, 1.0f, i*step, 1.0f));
        }
    } else if (scheme == "Cyan to Green") {
        const float step = 1.0f/(nColors);
        for (unsigned int i=0; i<nColors; i++)
        {
            entries.push_back(type::RGBAColor(0.0f, 1.0f, 1.0f-i*step, 1.0f));
        }
    } else if (scheme == "Cyan to Blue") {
        const float step = 1.0f/(nColors);
        for (unsigned int i=0; i<nColors; i++)
        {
            entries.push_back(type::RGBAColor(0.0f, 1.0f-i*step, 1.0f, 1.0f));
        }
    } else if (scheme == "Blue to Cyan") {
        const float step = 1.0f/(nColors);
        for (unsigned int i=0; i<nColors; i++)
        {
            entries.push_back(type::RGBAColor(0.0f, i*step, 1.0f, 1.0f));
        }
    } else if (scheme == "Red") {
        const float step = 1.4f/(nColors);
        for (unsigned int i=0; i<nColors/2; i++)
        {
            entries.push_back(type::RGBAColor(0.3f + i*step, 0.0f, 0.0f, 1.0f));
        }
        for (unsigned int i=0; i<(nColors - nColors/2); i++)
        {
            entries.push_back(type::RGBAColor(1.0f, i*step, i*step, 1.0f));
        }
    } else if (scheme == "RedInv") {
        const float step = 1.4f/(nColors);
        for (unsigned int i=0; i<(nColors - nColors/2); i++)
        {
            entries.push_back(type::RGBAColor(1.0f, 0.7f-i*step, 0.7f-i*step, 1.0f));
        }
        for (unsigned int i=0; i<nColors/2; i++)
        {
            entries.push_back(type::RGBAColor(1.0f-i*step, 0.0f, 0.0f, 1.0f));
        }
    } else if (scheme == "Green") {
        const float step = 1.4f/(nColors);
        for (unsigned int i=0; i<nColors/2; i++)
        {
            entries.push_back(type::RGBAColor(0.0f, 0.3f + i*step, 0.0f, 1.0f));
        }
        for (unsigned int i=0; i<(nColors - nColors/2); i++)
        {
            entries.push_back(type::RGBAColor(i*step, 1.0f, i*step, 1.0f));
        }
    } else if (scheme == "GreenInv") {
        const float step = 1.4f/(nColors);
        for (unsigned int i=0; i<(nColors - nColors/2); i++)
        {
            entries.push_back(type::RGBAColor(0.7f-i*step, 1.0f, 0.7f-i*step, 1.0f));
        }
        for (unsigned int i=0; i<nColors/2; i++)
        {
            entries.push_back(type::RGBAColor(0.0f, 1.0f - i*step, 0.0f, 1.0f));
        }
    } else if (scheme == "Blue") {
        const float step = 1.4f/(nColors);
        for (unsigned int i=0; i<nColors/2; i++)
        {
            entries.push_back(type::RGBAColor(0.0f, 0.0f, 0.3f + i*step, 1.0f));
        }
        for (unsigned int i=0; i<(nColors - nColors/2); i++)
        {
            entries.push_back(type::RGBAColor(i*step, i*step, 1.0f, 1.0f));
        }
    } else if (scheme == "BlueInv") {
        const float step = 1.4f/(nColors);
        for (unsigned int i=0; i<(nColors - nColors/2); i++)
        {
            entries.push_back(type::RGBAColor(0.7f-i*step, 0.7f-i*step, 1.0f, 1.0f));
        }
        for (unsigned int i=0; i<nColors/2; i++)
        {
            entries.push_back(type::RGBAColor(0.0f, 0.0f, 1.0f - i*step, 1.0f));
        }
    } else {
        // HSV is the default
        if (scheme != "HSV") {
            msg_warning("ColorMap") << "Invalid color scheme selected: " << scheme ;
        }

        // List the colors
        const float step = 1.0f/(nColors-1);
        for (unsigned int i=0; i<nColors; i++)
        {
            const auto h = i * step * 360;
            entries.emplace_back(type::RGBAColor::fromHSVA(h,1.f,1.f, 1.0f));
        }
    }
}

// Color space conversion routines

// Hue/Saturation/Value -> Red/Green/Blue
// h,s,v ∈ [0,1]
// r,g,b ∈ [0,1]
// Ref: Alvy Ray Smith, Color Gamut Transform Pairs, SIGGRAPH '78
type::Vec3f ColorMap::hsv2rgb(const type::Vec3f&hsv)
{
    auto rgba = type::RGBAColor::fromHSVA( hsv[0] * 360, hsv[1],hsv[2], 1.0f );

    return { rgba[0], rgba[1], rgba[2] };
}


} // namespace component

} // namespace sofa
