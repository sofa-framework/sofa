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
#ifndef SOFA_HELPER_GL_TEXT_H
#define SOFA_HELPER_GL_TEXT_H

#ifndef SOFA_NO_OPENGL

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>

#include <string>
#include <sstream>

#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glu.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/gl/Texture.h>

#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

namespace gl
{


/**
 * This class, called GlText, allows to render text in OpenGL, always facing the camera
 * in 2D (screen) or in 3D (world coordinates)
*/

class SOFA_HELPER_API GlText
{
public:
    typedef sofa::helper::fixed_array<float, 3> Vector3;
    typedef sofa::helper::fixed_array<float, 2> Vector2;

    /// Constructor
    GlText ();
    /// Constructor with specified text
    GlText ( const std::string& text );
    /// Constructor with specified text and position
    GlText ( const std::string& text, const defaulttype::Vector3& position );
    /// Constructor with specified text, position and scale
    GlText ( const std::string& text, const defaulttype::Vector3& position, const double& scale );
    /// Destructor
    ~GlText();

    /// Update the text to render
    void setText ( const std::string& text );
    /// Update the text to render
    template <typename T>
    void setText ( const T& text );
    /// Update the position used to render the text
    void update ( const defaulttype::Vector3& position );
    /// Update the scale used to render the text
    void update ( const double& scale );

    /// Render the text at the defined position and scale.
    void draw();

    ///// Render the text at the current position with no scale
    //template <typename T>
    //static void draw ( const T& text );
    ///// Render the text at the defined position with no scale
    //template <typename T>
    //static void draw ( const T& text, const defaulttype::Vector3& position );

    /// Render the text at the defined position and scale
    template <typename T>
    static void draw ( const T& text, const defaulttype::Vector3& position = defaulttype::Vector3(0.0,0.0,0.0), const double& scale = 1.0);
    
    static void textureDraw_Overlay(const char* text, const double scale = 1.0);
    static void textureDraw_Indices(const helper::vector<defaulttype::Vector3>& positions, const double& scale);

private:
    static void initTexture();

    static const std::string ASCII_TEXTURE_PATH;

    static sofa::helper::io::Image *s_asciiImage;
    static sofa::helper::gl::Texture* s_asciiTexture;

    double scale;
    std::string text;
    defaulttype::Vector3 position;
};

} // namespace gl

} // namespace helper

} // namespace sofa

#endif /* SOFA_NO_OPENGL */

#endif
