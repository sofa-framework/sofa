/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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


template <typename T>
void GlText::textureDraw(const T& text, const defaulttype::Vector3& position, const double& scale)
{
    defaulttype::Mat<4, 4, GLfloat> modelviewM;

    const unsigned int nb_char_width = 16;
    const unsigned int nb_char_height = 16;
    const float worldSize = 1.0;

    std::ostringstream oss;
    oss << text;
    std::string tmp = oss.str();
    unsigned int length = tmp.size();

    typedef sofa::helper::fixed_array<float, 3> Vector3;
    typedef sofa::helper::fixed_array<float, 2> Vector2;

    std::vector<Vector3> vertices;
    std::vector<Vector2> UVs;

    glDisable(GL_LIGHTING);
    glPushMatrix();

    // Makes text always face the viewer by removing the scene rotation
    // get the current modelview matrix
    glGetFloatv(GL_MODELVIEW_MATRIX, modelviewM.ptr());
    modelviewM.transpose();

    defaulttype::Vec3d temp(position[0], position[1], position[2]);
    temp = modelviewM.transform(temp);

    glLoadIdentity();
    glTranslatef((float)temp[0], (float)temp[1], (float)temp[2]);
    glScalef((float)scale, (float)scale, (float)scale);
    glRotatef(180.0, 1, 0, 0);

    for (unsigned int i = 0; i<length; i++)
    {
        Vector3 vertex_up_left = Vector3(i*worldSize, worldSize, 0.0);
        Vector3 vertex_up_right = Vector3(i*worldSize + worldSize, worldSize, 0.0);
        Vector3 vertex_down_right = Vector3(i*worldSize + worldSize, 0.0, 0.0);
        Vector3 vertex_down_left = Vector3(i*worldSize, 0.0, 0.0);

        vertices.push_back(vertex_up_left);
        vertices.push_back(vertex_down_left);
        vertices.push_back(vertex_up_right);

        vertices.push_back(vertex_down_right);
        vertices.push_back(vertex_up_right);
        vertices.push_back(vertex_down_left);

        char character = text[i] - 32 ;
        float uv_x = (character % nb_char_width) / (float)nb_char_width;
        float uv_y = 1.0 - ( (character / nb_char_height) / (float)nb_char_height );

        Vector2 uv_up_left = Vector2(uv_x, (uv_y - (1.0f / (float)nb_char_height)));
        Vector2 uv_up_right = Vector2(uv_x + (1.0f / (float)nb_char_width), (uv_y - (1.0f / (float)nb_char_height)));
        Vector2 uv_down_right = Vector2(uv_x + (1.0f / (float)nb_char_width), uv_y);
        Vector2 uv_down_left = Vector2(uv_x, uv_y);

        UVs.push_back(uv_up_left);
        UVs.push_back(uv_down_left);
        UVs.push_back(uv_up_right);

        UVs.push_back(uv_down_right);
        UVs.push_back(uv_up_right);
        UVs.push_back(uv_down_left);
    }

    glPushAttrib(GL_TEXTURE_BIT);
    glEnable(GL_TEXTURE_2D);

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glEnable(GL_ALPHA_TEST);
    glAlphaFunc(GL_GREATER, 0.0);
    m_tex->init();
    m_tex->bind();

    glBegin(GL_TRIANGLES);
    for (unsigned int i = 0; i < vertices.size() ; i++)
    {
        glColor4f(1.0, 1.0, 1.0, 0.0);
        glTexCoord2fv(UVs[i].data());
        glVertex3fv(vertices[i].data());
    }
    glEnd();

    m_tex->unbind();
    glDisable(GL_ALPHA_TEST);
    glPopAttrib();

    glPopMatrix();

    glEnable(GL_LIGHTING);
}

} // namespace gl

} // namespace helper

} // namespace sofa

#endif /* SOFA_NO_OPENGL */

#endif
