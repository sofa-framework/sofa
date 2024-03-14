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
#pragma once
#include <sofa/gl/glText.h>

#include <sofa/gl/gl.h>

#include <cassert>
#include <algorithm>
#include <iostream>

namespace sofa::gl
{

namespace {
    using sofa::type::Vec3f;
    using sofa::type::Vec2f;
}

template <typename T>
void GlText::setText ( const T& text )
{
    std::ostringstream oss;
    oss << text;
    this->text = oss.str();
}

template <typename T>
void GlText::draw(const T& text, const type::Vec3& position, const double& scale)
{
    if (!s_asciiTexture)
    {
        GlText::initTexture();
        s_asciiTexture->init();
    }
    type::Mat<4, 4, GLfloat> modelviewM;

    const unsigned int nb_char_width = 16;
    const unsigned int nb_char_height = 16;
    const float worldHeight = 1.0;
    const float worldWidth = 0.5;

    std::ostringstream oss;
    oss << text;
    const std::string str = oss.str();
    const std::size_t length = str.size();

    std::vector<type::Vec3f> vertices;
    std::vector<type::Vec2f> UVs;

    glPushAttrib(GL_TEXTURE_BIT);
    glEnable(GL_TEXTURE_2D);

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glEnable(GL_ALPHA_TEST);
    glAlphaFunc(GL_GREATER, 0.0);


    s_asciiTexture->bind();

    glDisable(GL_LIGHTING);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    // Makes text always face the viewer by removing the scene rotation
    // get the current modelview matrix
    glGetFloatv(GL_MODELVIEW_MATRIX, modelviewM.ptr());
    modelviewM.transpose();

    type::Vec3d temp(position[0], position[1], position[2]);
    temp = modelviewM.transform(temp);

    glLoadIdentity();
    glTranslatef((float)temp[0], (float)temp[1], (float)temp[2]);
    glScalef((float)scale, (float)scale, (float)scale);
    glRotatef(180.0, 1, 0, 0);

	for (std::size_t j = 0; j < length; j++)
    {
        Vec3f vertex_up_left = Vec3f(j*worldWidth, worldHeight, 0.0);
        Vec3f vertex_up_right = Vec3f(j*worldWidth + worldWidth, worldHeight, 0.0);
        Vec3f vertex_down_right = Vec3f(j*worldWidth + worldWidth, 0.0, 0.0);
        Vec3f vertex_down_left = Vec3f(j*worldWidth, 0.0, 0.0);

        vertices.push_back(vertex_up_left);
        vertices.push_back(vertex_down_left);
        vertices.push_back(vertex_up_right);

        vertices.push_back(vertex_down_right);
        vertices.push_back(vertex_up_right);
        vertices.push_back(vertex_down_left);

        const char character = str[j] - 32;

        float uv_x = (character % nb_char_width) / (float)nb_char_width;
        float uv_y = 1.0f - ((character / nb_char_height) / (float)nb_char_height);

        Vec2f uv_up_left = Vec2f(uv_x, (uv_y - (1.0f / (float)nb_char_height)));
        Vec2f uv_up_right = Vec2f(uv_x + (1.0f / (float)nb_char_width), (uv_y - (1.0f / (float)nb_char_height)));
        Vec2f uv_down_right = Vec2f(uv_x + (1.0f / (float)nb_char_width), uv_y);
        Vec2f uv_down_left = Vec2f(uv_x, uv_y);

        UVs.push_back(uv_up_left);
        UVs.push_back(uv_down_left);
        UVs.push_back(uv_up_right);

        UVs.push_back(uv_down_right);
        UVs.push_back(uv_up_right);
        UVs.push_back(uv_down_left);
    }

    glBegin(GL_TRIANGLES);
	for (std::size_t j = 0; j < vertices.size(); j++)
    {
        glTexCoord2fv(UVs[j].data());
        glVertex3fv(vertices[j].data());
    }
    glEnd();

    glPopMatrix();

    s_asciiTexture->unbind();
    glDisable(GL_ALPHA_TEST);
    glPopAttrib();


    glEnable(GL_LIGHTING);
}

} // namespace sofa::gl
