/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Authors: The SOFA Team   (see Authors.txt)                                  *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/gl/glText.h>

#include <sofa/helper/system/gl.h>

#include <assert.h>
#include <algorithm>
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;


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
    Mat<4,4, GLfloat> modelviewM;
    glColor3f ( 1.0,1.0,1.0 );
    glDisable ( GL_LIGHTING );

    std::ostringstream oss;
    oss << text;
    string tmp = oss.str();
    const char* s = tmp.c_str();
    glPushMatrix();

    // Makes text always face the viewer by removing the scene rotation
    // get the current modelview matrix
    glGetFloatv ( GL_MODELVIEW_MATRIX , modelviewM.ptr() );
    modelviewM.transpose();

    Vec3d temp = modelviewM.transform ( Vec3d() );

    glLoadIdentity();
    glTranslatef ( temp[0], temp[1], temp[2] );

    while ( *s )
    {
        glutStrokeCharacter ( GLUT_STROKE_ROMAN, *s );
        s++;
    }

    glPopMatrix();
}


template <typename T>
void GlText::draw ( const T& text, const Vector3& position )
{
    Mat<4,4, GLfloat> modelviewM;
    glColor3f ( 1.0,1.0,1.0 );
    glDisable ( GL_LIGHTING );

    std::ostringstream oss;
    oss << text;
    string tmp = oss.str();
    const char* s = tmp.c_str();
    glPushMatrix();

    glTranslatef ( position[0],  position[1],  position[2]);

    // Makes text always face the viewer by removing the scene rotation
    // get the current modelview matrix
    glGetFloatv ( GL_MODELVIEW_MATRIX , modelviewM.ptr() );
    modelviewM.transpose();

    Vec3d temp ( position[0],  position[1],  position[2]);
    temp = modelviewM.transform ( temp );

    glLoadIdentity();
    glTranslatef ( temp[0], temp[1], temp[2] );

    while ( *s )
    {
        glutStrokeCharacter ( GLUT_STROKE_ROMAN, *s );
        s++;
    }

    glPopMatrix();
}



template <typename T>
void GlText::draw ( const T& text, const Vector3& position, const double& scale )
{
    Mat<4,4, GLfloat> modelviewM;
    glColor3f ( 1.0,1.0,1.0 );
    glDisable ( GL_LIGHTING );

    std::ostringstream oss;
    oss << text;
    string tmp = oss.str();
    const char* s = tmp.c_str();
    glPushMatrix();

    glTranslatef ( position[0],  position[1],  position[2]);
    glScalef ( scale,scale,scale );

    // Makes text always face the viewer by removing the scene rotation
    // get the current modelview matrix
    glGetFloatv ( GL_MODELVIEW_MATRIX , modelviewM.ptr() );
    modelviewM.transpose();

    Vec3d temp ( position[0],  position[1],  position[2]);
    temp = modelviewM.transform ( temp );

    glLoadIdentity();
    glTranslatef ( temp[0], temp[1], temp[2] );
    glScalef ( scale,scale,scale );

    while ( *s )
    {
        glutStrokeCharacter ( GLUT_STROKE_ROMAN, *s );
        s++;
    }

    glPopMatrix();
}

} // namespace gl

} // namespace helper

} // namespace sofa
