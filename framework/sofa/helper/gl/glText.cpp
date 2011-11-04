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
#include <sofa/helper/gl/glText.inl>

namespace sofa
{

namespace helper
{

namespace gl
{

GlText::GlText()
{
}

GlText::GlText ( const string& text )
{
    this->text = text;
}

GlText::GlText ( const string& text, const Vector3& position )
{
    this->text = text;
    this->position = position;
}

GlText::GlText ( const string& text, const Vector3& position, const double& scale )
{
    this->text = text;
    this->position = position;
    this->scale = scale;
}

GlText::~GlText()
{
}


void GlText::setText ( const string& text )
{
    this->text = text;
}

void GlText::update ( const Vector3& position )
{
    this->position = position;
}

void GlText::update ( const double& scale )
{
    this->scale = scale;
}


void GlText::draw()
{
    Mat<4,4, GLfloat> modelviewM;
    glDisable ( GL_LIGHTING );

    const char* s = text.c_str();
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
