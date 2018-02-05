/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#include <SofaOpenglVisual/OglColorMap.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>

#ifdef SOFA_HAVE_GLEW
#include <sofa/helper/gl/GLSLShader.h>
#endif // SOFA_HAVE_GLEW

#include <string>
#include <iostream>

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(OglColorMap)

int OglColorMapClass = core::RegisterObject("Provides color palette and support for conversion of numbers to colors.")
        .add< OglColorMap >()
        .addAlias("ColorMap")
        ;

OglColorMap::OglColorMap()
: f_paletteSize(initData(&f_paletteSize, (unsigned int)256, "paletteSize", "How many colors to use"))
, f_colorScheme(initData(&f_colorScheme, "colorScheme", "Color scheme to use"))
, f_showLegend(initData(&f_showLegend, false, "showLegend", "Activate rendering of color scale legend on the side"))
, f_legendOffset(initData(&f_legendOffset, defaulttype::Vec2f(10.0f,5.0f),"legendOffset", "Draw the legend on screen with an x,y offset"))
, f_legendTitle(initData(&f_legendTitle,"legendTitle", "Add a title to the legend"))
, d_min(initData(&d_min,0.0f,"min","min value for drawing the legend without the need to actually use the range with getEvaluator method wich sets the min"))
, d_max(initData(&d_max,0.0f,"max","max value for drawing the legend without the need to actually use the range with getEvaluator method wich sets the max"))
, d_legendRangeScale(initData(&d_legendRangeScale,1.f,"legendRangeScale","to change the unit of the min/max value of the legend"))
, texture(0)
{
   f_colorScheme.beginEdit()->setNames(19,
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
        );
    f_colorScheme.beginEdit()->setSelectedItem("HSV");
    f_colorScheme.endEdit();

}

OglColorMap::~OglColorMap() {
    // Some components may use OglColorMap internally, in which case an OpenGL
    // context might not exist.  That's why this 'if' is here, to avoid calling
    // an OpenGL function in a destructor unless strictly necessary.
    if (texture != 0)
        glDeleteTextures(1, &texture);
}

// For backward compatibility only
// TODO: remove this later
//void OglColorMap::initOld(const std::string &data)
//{
//    if (data == "") {
//        entries.insert(entries.end(), DefaultOglColorMapEntries, DefaultOglColorMapEntries+NDefaultOglColorMapEntries);
//        return;
//    }
//
//    std::istringstream is(data);
//    is >> *this;
//
//    return;
//}

void OglColorMap::init()
{
    reinit();
}


void OglColorMap::reinit()
{
    m_colorMap.setPaletteSize(f_paletteSize.getValue());
    m_colorMap.setColorScheme(f_colorScheme.getValue().getSelectedItem());
    m_colorMap.reinit();
    }

OglColorMap* OglColorMap::getDefault()
{
    static OglColorMap::SPtr defaultOglColorMap;
    if (defaultOglColorMap == NULL) {
        defaultOglColorMap = sofa::core::objectmodel::New< OglColorMap >();
        std::string tmp("");
        //defaultOglColorMap->initOld(tmp); // TODO: replace initOld() with init()
        defaultOglColorMap->init();
    }
    return defaultOglColorMap.get();
}

void OglColorMap::drawVisual(const core::visual::VisualParams* vparams)
{
    if( !vparams->displayFlags().getShowVisual() ) return;

    if (!f_showLegend.getValue()) return;

    // Prepare texture for legend
    // crashes on mac in batch mode (no GL context)
    if (vparams->isSupported(core::visual::API_OpenGL)
        && !texture)
    {
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_1D, texture);
        //glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        //glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

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

        delete[] data;
    }



    //
    // Draw legend
    //
    // TODO: move the code to DrawTool


    const std::string& legendTitle = f_legendTitle.getValue();
    int yoffset = legendTitle.empty() ? 0 : 25;


    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);
    const int vWidth = viewport[2];
    const int vHeight = viewport[3];

    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_BLEND);

    // disable clipping planes
	for(int i = 0; i < GL_MAX_CLIP_PLANES; ++i)
		glDisable(GL_CLIP_PLANE0+i);

    // Setup orthogonal projection
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, vWidth, vHeight, 0.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

#ifdef SOFA_HAVE_GLEW
    for(int i = 0; i < 8; ++i)
    {
        glActiveTexture(GL_TEXTURE0 + i);
        glDisable(GL_TEXTURE_2D);
    }

    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_1D);
    glBindTexture(GL_TEXTURE_1D, texture);

    //glBlendFunc(GL_ONE, GL_ONE);
    glColor3f(1.0f, 1.0f, 1.0f);

    GLhandleARB currentShader = sofa::helper::gl::GLSLShader::GetActiveShaderProgram();
    sofa::helper::gl::GLSLShader::SetActiveShaderProgram(0);
#endif // SOFA_HAVE_GLEW

    glBegin(GL_QUADS);

    glTexCoord1f(1.0);
    glVertex3f(20.0f+f_legendOffset.getValue().x(), yoffset+20.0f+f_legendOffset.getValue().y(), 0.0f);

    glTexCoord1f(1.0);
    glVertex3f(10.0f+f_legendOffset.getValue().x(), yoffset+20.0f+f_legendOffset.getValue().y(), 0.0f);

    glTexCoord1f(0.0);
    glVertex3f(10.0f+f_legendOffset.getValue().x(), yoffset+120.0f+f_legendOffset.getValue().y(), 0.0f);

    glTexCoord1f(0.0);
    glVertex3f(20.0f+f_legendOffset.getValue().x(), yoffset+120.0f+f_legendOffset.getValue().y(), 0.0f);

    glEnd();

    glDisable(GL_TEXTURE_1D);

    // Restore model view matrix
    glPopMatrix(); // GL_MODELVIEW

    // Restore projection matrix
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);

    // Adjust the text color according to the background luminance
    GLfloat bgcol[4];
    glGetFloatv(GL_COLOR_CLEAR_VALUE,bgcol);

    Color textcolor(1.0f, 1.0f, 1.0f, 1.0f);
    static const sofa::defaulttype::Vec3f luminanceMatrix(0.212f, 0.715f, 0.072f);
    float backgroundLuminance = sofa::defaulttype::Vec3f(bgcol[0], bgcol[1], bgcol[2]) * luminanceMatrix;
    if(backgroundLuminance > 0.5f)
        textcolor = Color(0.0f, 0.0f, 0.0f, 1.0f);

    if( !legendTitle.empty() )
    {
        vparams->drawTool()->writeOverlayText((int)f_legendOffset.getValue().x(), // x
                                              (int)f_legendOffset.getValue().y(), // y
                                              11u, // size
                                              textcolor,
                                              legendTitle.c_str());
    }



    // Maximum & minimum
    std::ostringstream smin, smax;
    smin << d_min.getValue() * d_legendRangeScale.getValue();
    smax << d_max.getValue() * d_legendRangeScale.getValue();



    vparams->drawTool()->writeOverlayText((int)f_legendOffset.getValue().x(), // x
                                          yoffset + (int)f_legendOffset.getValue().y(), // y
                                          12u, // size
                                          textcolor,
                                          smax.str().c_str());

    vparams->drawTool()->writeOverlayText((int)f_legendOffset.getValue().x(), // x
                                          yoffset + 120 + (int)f_legendOffset.getValue().y(), // y
                                          12u, // size
                                          textcolor,
                                          smin.str().c_str());

#ifdef SOFA_HAVE_GLEW
    sofa::helper::gl::GLSLShader::SetActiveShaderProgram(currentShader);
#endif // SOFA_HAVE_GLEW

    // Restore state
    glPopAttrib();
}


} // namespace visualmodel

} // namespace component

} // namespace sofa
