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
#ifndef SOFA_HELPER_GL_TEXT_H
#define SOFA_HELPER_GL_TEXT_H

#ifndef SOFA_NO_OPENGL

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>

#include <string>
#include <sstream>

#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glu.h>
#include <sofa/helper/system/glut.h>

#include <sofa/SofaFramework.h>

namespace sofa
{

namespace helper
{

namespace gl
{


/**
 * This class, called GlText, allow to render text at a 3D position, facing the camera
*/

class SOFA_HELPER_API GlText
{
public:
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

    /// Render the text at the current position with no scale
    template <typename T>
    static void draw ( const T& text );
    /// Render the text at the defined position with no scale
    template <typename T>
    static void draw ( const T& text, const defaulttype::Vector3& position );
    /// Render the text at the defined position and scale
    template <typename T>
    static void draw ( const T& text, const defaulttype::Vector3& position, const double& scale );

private:
    double scale;
    std::string text;
    defaulttype::Vector3 position;
};

} // namespace gl

} // namespace helper

} // namespace sofa

#endif /* SOFA_NO_OPENGL */

#endif
