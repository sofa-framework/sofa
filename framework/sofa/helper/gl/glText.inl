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
#ifndef SOFA_HELPER_GL_GLTEXT_INL
#define SOFA_HELPER_GL_GLTEXT_INL

#ifndef SOFA_NO_OPENGL

#include <sofa/helper/gl/glText.h>

#include <sofa/helper/system/gl.h>

#include <cassert>
#include <algorithm>
#include <iostream>


namespace sofa
{

namespace helper
{

namespace gl
{

template <typename T>
void GlText::setText ( const T& text )
{
    std::ostringstream oss;
    oss << text;
    this->text = oss.str();
}



template <typename T>
void GlText::draw ( const T& text )
{
#ifndef PS3
    defaulttype::Mat<4,4, GLfloat> modelviewM;
    glDisable ( GL_LIGHTING );

    std::ostringstream oss;
    oss << text;
    std::string tmp = oss.str();
    const char* s = tmp.c_str();
    glPushMatrix();

    // Makes text always face the viewer by removing the scene rotation
    // get the current modelview matrix
    glGetFloatv ( GL_MODELVIEW_MATRIX , modelviewM.ptr() );
    modelviewM.transpose();

    defaulttype::Vec3d temp = modelviewM.transform ( defaulttype::Vec3d() );

    glLoadIdentity();
    glTranslatef ( temp[0], temp[1], temp[2] );

    while ( *s )
    {
        glutStrokeCharacter ( GLUT_STROKE_ROMAN, *s );
        s++;
    }

    glPopMatrix();
#endif
}


template <typename T>
void GlText::draw ( const T& text, const defaulttype::Vector3& position )
{
#ifndef PS3
    defaulttype::Mat<4,4, GLfloat> modelviewM;
    glDisable ( GL_LIGHTING );

    std::ostringstream oss;
    oss << text;
    std::string tmp = oss.str();
    const char* s = tmp.c_str();
    glPushMatrix();

    glTranslatef ( position[0],  position[1],  position[2]);

    // Makes text always face the viewer by removing the scene rotation
    // get the current modelview matrix
    glGetFloatv ( GL_MODELVIEW_MATRIX , modelviewM.ptr() );
    modelviewM.transpose();

    defaulttype::Vec3d temp ( position[0],  position[1],  position[2]);
    temp = modelviewM.transform ( temp );

    glLoadIdentity();
    glTranslatef ( temp[0], temp[1], temp[2] );

    while ( *s )
    {
        glutStrokeCharacter ( GLUT_STROKE_ROMAN, *s );
        s++;
    }

    glPopMatrix();
#endif
}



template <typename T>
void GlText::draw ( const T& text, const defaulttype::Vector3& position, const double& scale )
{
#ifndef PS3
    defaulttype::Mat<4,4, GLfloat> modelviewM;
    glDisable ( GL_LIGHTING );

    std::ostringstream oss;
    oss << text;
    std::string tmp = oss.str();
    const char* s = tmp.c_str();
    glPushMatrix();

    // Makes text always face the viewer by removing the scene rotation
    // get the current modelview matrix
    glGetFloatv ( GL_MODELVIEW_MATRIX , modelviewM.ptr() );
    modelviewM.transpose();

    defaulttype::Vec3d temp ( position[0],  position[1],  position[2] );
    temp = modelviewM.transform ( temp );

    glLoadIdentity();
    glTranslatef ( (float)temp[0], (float)temp[1], (float)temp[2] );
    glScalef ( (float)scale, (float)scale, (float)scale );

    while ( *s )
    {
        glutStrokeCharacter ( GLUT_STROKE_ROMAN, *s );
        s++;
    }

    glPopMatrix();
#endif
}

} // namespace gl

} // namespace helper

} // namespace sofa

#endif /* SOFA_NO_OPENGL */

#endif
