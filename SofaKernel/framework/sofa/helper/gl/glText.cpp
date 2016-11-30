/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
using namespace sofa::defaulttype;
using std::string;

SOFA_HELPER_API const std::string GlText::ASCII_TEXTURE_PATH("textures/texture_ascii_smooth.png");
SOFA_HELPER_API sofa::helper::io::Image *GlText::s_asciiImage = NULL;
SOFA_HELPER_API sofa::helper::gl::Texture* GlText::s_asciiTexture = NULL;

void GlText::initTexture()
{
    if (s_asciiImage == NULL)
    {
        s_asciiImage = helper::io::Image::Create(ASCII_TEXTURE_PATH);
    }
    if (s_asciiTexture == NULL && s_asciiImage != NULL)
    {
        s_asciiTexture = new sofa::helper::gl::Texture(s_asciiImage, false, true, false );
    }
}

GlText::GlText()
{
}

GlText::GlText ( const string& text )
{
    this->text = text;
}

GlText::GlText ( const string& text, const defaulttype::Vector3& position )
{
    this->text = text;
    this->position = position;
}

GlText::GlText ( const string& text, const defaulttype::Vector3& position, const double& scale )
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

void GlText::update ( const defaulttype::Vector3& position )
{
    this->position = position;
}

void GlText::update ( const double& scale )
{
    this->scale = scale;
}


void GlText::draw()
{
#ifndef PS3
    Mat<4,4, GLfloat> modelviewM;
    glDisable ( GL_LIGHTING );

    const char* s = text.c_str();

    GlText::draw(s, position, scale);

#endif
}

void GlText::textureDraw_Overlay(const char* text, float scale)
{
    if (!s_asciiTexture)
    {
        GlText::initTexture();
        s_asciiTexture->init();
    }
    static const unsigned int nb_char_width = 16;
    static const unsigned int nb_char_height = 16;
    static const float worldHeight = 1.0f;
    static const float worldWidth = 0.50f;

    std::vector<Vector3> vertices;
    std::vector<Vector2> UVs;

    std::ostringstream oss;
    oss << text;
    std::string str = oss.str();
    unsigned int length = str.size();

    glPushAttrib(GL_TEXTURE_BIT);
    glEnable(GL_TEXTURE_2D);

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE); // multiply tex color with glColor
    //glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE); // only tex color (no glColor)
    glEnable(GL_ALPHA_TEST);
    glAlphaFunc(GL_GREATER, 0.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s_asciiTexture->bind();

    glScalef((float)scale, (float)scale, (float)scale);

    for (unsigned int j = 0; j < length; j++)
    {
        Vector3 vertex_up_left = Vector3(j*worldWidth, worldHeight, 0.0);
        Vector3 vertex_up_right = Vector3(j*worldWidth + worldWidth, worldHeight, 0.0);
        Vector3 vertex_down_right = Vector3(j*worldWidth + worldWidth, 0.0, 0.0);
        Vector3 vertex_down_left = Vector3(j*worldWidth, 0.0, 0.0);

        vertices.push_back(vertex_up_left);
        vertices.push_back(vertex_down_left);
        vertices.push_back(vertex_down_right);
        vertices.push_back(vertex_up_right);

        char character = str[j] - 32;

        float uv_x = (character % nb_char_width) / (float)nb_char_width;
        float uv_y = 1.0f - ((character / nb_char_height) / (float)nb_char_height);

        Vector2 uv_up_left = Vector2(uv_x, (uv_y - (1.0f / (float)nb_char_height)));
        Vector2 uv_up_right = Vector2(uv_x + (1.0f / (float)nb_char_width), (uv_y - (1.0f / (float)nb_char_height)));
        Vector2 uv_down_right = Vector2(uv_x + (1.0f / (float)nb_char_width), uv_y);
        Vector2 uv_down_left = Vector2(uv_x, uv_y);

        UVs.push_back(uv_up_left);
        UVs.push_back(uv_down_left);
        UVs.push_back(uv_down_right);
        UVs.push_back(uv_up_right);
    }

    glBegin(GL_QUADS);
    for (unsigned int j = 0; j < vertices.size(); j++)
    {
        glTexCoord2fv(UVs[j].data());
        glVertex3fv(vertices[j].data());
    }
    glEnd();

    s_asciiTexture->unbind();
    glDisable(GL_ALPHA_TEST);
    glPopAttrib();

}



} // namespace gl

} // namespace helper

} // namespace sofa
