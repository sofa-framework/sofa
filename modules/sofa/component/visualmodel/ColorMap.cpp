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

#include <sofa/component/visualmodel/ColorMap.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>

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

SOFA_DECL_CLASS(ColorMap)

int ColorMapClass = core::RegisterObject("Provides color palette and support for conversion of numbers to colors.")
        .add< ColorMap >()
        ;

ColorMap::ColorMap()
: f_paletteSize(initData(&f_paletteSize, (unsigned int)256, "paletteSize", "How many colors to use"))
, f_colorScheme(initData(&f_colorScheme, "colorScheme", "Color scheme to use"))
{
    f_colorScheme.beginEdit()->setNames(6,
        "Red to Blue",  // HSV space
        "Blue to Red",  // HSV space
        "HSV",          // HSV space
        "Red",          // RGB space
        "Green",        // RGB space
        "Blue",         // RGB space
        "Custom"        // TODO: Custom colors
        );
    f_colorScheme.beginEdit()->setSelectedItem("HSV");
    f_colorScheme.endEdit();

}

// For backward compatibility only
// TODO: remove this later
void ColorMap::initOld(const std::string &data)
{
    if (data == "") {
        entries.insert(entries.end(), DefaultColorMapEntries, DefaultColorMapEntries+NDefaultColorMapEntries);
        return;
    }

    std::istringstream is(data);
    is >> *this;

    return;
}

void ColorMap::init()
{
    // Prepare texture for legend
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_1D, texture);
    //glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    reinit();
}


void ColorMap::reinit()
{
    entries.clear();

    unsigned int nColors = f_paletteSize.getValue();
    if (nColors < 2) {
        serr << "Pallette size has to be equal or greater than 2.";
        *f_paletteSize.beginEdit() = 2;
        f_paletteSize.endEdit();
        nColors = 2;
    }

    std::string scheme = f_colorScheme.getValue().getSelectedItem();
    if (scheme == "Custom") {
        // TODO
    } else if (scheme == "Red to Blue") {
        // List the colors
        float step = (2.0f/3.0f)/(nColors-1);
        for (unsigned int i=0; i<nColors; i++)
        {
            entries.push_back(Color(
                    hsv2rgb(Color3(i*step, 1.0, 1.0)),
                    1.0 // alpha
                    ));
        }

    } else if (scheme == "Blue to Red") {
        // List the colors
        float step = (2.0f/3.0f)/(nColors-1);
        for (unsigned int i=0; i<nColors; i++)
        {
            entries.push_back(Color(
                    hsv2rgb(Color3(2.0f/3.0f - i*step, 1.0, 1.0)),
                    1.0 // alpha
                    ));
        }

    } else if (scheme == "Red") {
        float step = 1.4f/(nColors);
        for (unsigned int i=0; i<nColors/2; i++)
        {
            entries.push_back(Color(
                    0.3f + i*step, 0.0, 0.0,
                    1.0 // alpha
                    ));
        }
        for (unsigned int i=0; i<(nColors - nColors/2); i++)
        {
            entries.push_back(Color(
                    1.0, i*step, i*step,
                    1.0 // alpha
                    ));
        }


    } else if (scheme == "Green") {
        float step = 1.4f/(nColors);
        for (unsigned int i=0; i<nColors/2; i++)
        {
            entries.push_back(Color(
                    0.0, 0.3f + i*step, 0.0,
                    1.0 // alpha
                    ));
        }
        for (unsigned int i=0; i<(nColors - nColors/2); i++)
        {
            entries.push_back(Color(
                    i*step, 1.0, i*step,
                    1.0 // alpha
                    ));
        }


    } else if (scheme == "Blue") {
        float step = 1.4f/(nColors);
        for (unsigned int i=0; i<nColors/2; i++)
        {
            entries.push_back(Color(
                    0.0, 0.0, 0.3f + i*step,
                    1.0 // alpha
                    ));
        }
        for (unsigned int i=0; i<(nColors - nColors/2); i++)
        {
            entries.push_back(Color(
                    i*step, i*step, 1.0,
                    1.0 // alpha
                    ));
        }


    } else {
        // HSV is the default
        if (scheme != "HSV") {
            serr << "Invalid color scheme selected: " << scheme << sendl;
        }

        // List the colors
        float step = 1.0f/(nColors-1);
        for (unsigned int i=0; i<nColors; i++)
        {
            entries.push_back(Color(
                    hsv2rgb(Color3(i*step,1,1)),
                    1.0 // alpha
                    ));
        }
    }

    prepareLegend();
}

ColorMap* ColorMap::getDefault()
{
    static ColorMap::SPtr defaultColorMap;
    if (defaultColorMap == NULL) {
        defaultColorMap = sofa::core::objectmodel::New< ColorMap >();
        std::string tmp("");
        defaultColorMap->initOld(tmp); // TODO: replace initOld() with init()
    }
    return defaultColorMap.get();
}

void ColorMap::prepareLegend()
{
    int width = getNbColors();
    unsigned char *data = new unsigned char[ width * 3 ];

    for (int i=0; i<width; i++) {
        Color c = getColor(i);
        data[i*3+0] = (unsigned char)(c[0]*255);
        data[i*3+1] = (unsigned char)(c[1]*255);
        data[i*3+2] = (unsigned char)(c[2]*255);
    }

    glBindTexture(GL_TEXTURE_1D, texture);

    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, width, 0, GL_RGB, GL_UNSIGNED_BYTE,
        data);

    delete data;
}

void ColorMap::drawVisual(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowVisualModels()) return;

    //
    // Draw legend
    //
    // TODO: move the code to DrawTool
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);
    const int vWidth = viewport[2];
    const int vHeight = viewport[3];

    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_1D);
    glDisable(GL_LIGHTING);
    glDisable(GL_BLEND);

    // Setup orthogonal projection
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, vWidth, vHeight, 0.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glBindTexture(GL_TEXTURE_1D, texture);

    //glBlendFunc(GL_ONE, GL_ONE);
    glColor3f(1.0f, 1.0f, 1.0f);

    glBegin(GL_QUADS);

    glTexCoord1f(1.0);
    glVertex3f(20, 30, 0.0);

    glTexCoord1f(1.0);
    glVertex3f(30, 30, 0.0);

    glTexCoord1f(0.0);
    glVertex3f(30, 130, 0.0);

    glTexCoord1f(0.0);
    glVertex3f(20, 130, 0.0);

    glEnd();

    // Restore projection matrix
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    // Restore model view matrix
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    // Restore state
    glPopAttrib();

    //
    // Maximum & minimum
    //

    std::ostringstream smin, smax;
    smin << min;
    smax << max;

    vparams->drawTool()->writeOverlayText(
        10, 10, 14,  // x, y, size
        Color(1.0f, 1.0f, 1.0f, 1.0f),
        smax.str().c_str());

    vparams->drawTool()->writeOverlayText(
        10, 135, 14,  // x, y, size
        Color(1.0f, 1.0f, 1.0f, 1.0f),
        smin.str().c_str());
}


// Color space conversion routines

// Hue/Saturation/Value -> Red/Green/Blue
// h,s,v ∈ [0,1]
// r,g,b ∈ [0,1]
// Ref: Alvy Ray Smith, Color Gamut Transform Pairs, SIGGRAPH '78
ColorMap::Color3 ColorMap::hsv2rgb(const Color3 &hsv)
{
    Color3 rgb(0.0, 0.0, 0.0);

    float i, f;
    f = modff(hsv[0] * 6.0f, &i);

    float x = hsv[2] * (1.0f - hsv[1]),
           y = hsv[2] * (1.0f - hsv[1] * f),
           z = hsv[2] * (1.0f - hsv[1] * (1.0f - f));

    switch ((int)i % 6) {
        case 0: rgb = Color3(hsv[2],      z,      x); break;
        case 1: rgb = Color3(     y, hsv[2],      x); break;
        case 2: rgb = Color3(     x, hsv[2],      z); break;
        case 3: rgb = Color3(     x,      y, hsv[2]); break;
        case 4: rgb = Color3(     z,      x, hsv[2]); break;
        case 5: rgb = Color3(hsv[2],      x,      y); break;
    }

    return rgb;
}

} // namespace visualmodel

} // namespace component

} // namespace sofa
