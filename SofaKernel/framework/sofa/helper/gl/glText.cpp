/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

void GlText::textureDraw_Overlay(const char* text, const double scale)
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

void GlText::textureDraw_Indices(const helper::vector<defaulttype::Vector3>& positions, const double& scale)
{
    if (!s_asciiTexture)
    {
        GlText::initTexture();
        s_asciiTexture->init();
    }
    defaulttype::Mat<4, 4, GLfloat> modelviewM;

    static const unsigned int nb_char_width = 16;
    static const unsigned int nb_char_height = 16;
    static const float worldHeight = 1.0;
    static const float worldWidth = 0.5;

    glPushAttrib(GL_TEXTURE_BIT);
    glEnable(GL_TEXTURE_2D);

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE); // multiply tex color with glColor
    //glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE); // only tex color (no glColor)
    glEnable(GL_ALPHA_TEST);
    glAlphaFunc(GL_GREATER, 0.0);

    s_asciiTexture->bind();

    for (unsigned int i = 0; i < positions.size(); i++)
    {
        std::ostringstream oss;
        oss << i;
        std::string str = oss.str();
        unsigned int length = str.size();

        std::vector<Vector3> vertices;
        std::vector<Vector2> UVs;

        glDisable(GL_LIGHTING);

        glPushMatrix();

        // Makes text always face the viewer by removing the scene rotation
        // get the current modelview matrix
        glGetFloatv(GL_MODELVIEW_MATRIX, modelviewM.ptr());
        modelviewM.transpose();

        defaulttype::Vec3d temp(positions[i][0], positions[i][1], positions[i][2]);
        temp = modelviewM.transform(temp);

        glLoadIdentity();
        //translate a little bit to center the text on the position (instead of starting from a top-left position)
        glTranslatef((float)temp[0] - (worldWidth*length*scale)*0.5, (float)temp[1] + worldHeight*scale*0.5, (float)temp[2]);
        glScalef((float)scale, (float)scale, (float)scale);
        glRotatef(180.0, 1, 0, 0);
        for (unsigned int j = 0; j < length; j++)
        {
            Vector3 vertex_up_left = Vector3(j*worldWidth, worldHeight, 0.0f);
            Vector3 vertex_up_right = Vector3(j*worldWidth + worldWidth, worldHeight, 0.0f);
            Vector3 vertex_down_right = Vector3(j*worldWidth + worldWidth, 0.0f, 0.0f);
            Vector3 vertex_down_left = Vector3(j*worldWidth, 0.0f, 0.0f);

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

        glPopMatrix();
    }

    s_asciiTexture->unbind();
    glDisable(GL_ALPHA_TEST);
    glPopAttrib();


    glEnable(GL_LIGHTING);
}



} // namespace gl

} // namespace helper

} // namespace sofa
