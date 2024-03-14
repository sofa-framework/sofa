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
#include <sofa/gl/glText.inl>

namespace sofa::gl
{
using namespace sofa::type;
using std::string;

SOFA_GL_API const std::string GlText::ASCII_TEXTURE_PATH("textures/texture_ascii_smooth.png");
SOFA_GL_API sofa::helper::io::Image *GlText::s_asciiImage = nullptr;
SOFA_GL_API sofa::gl::Texture* GlText::s_asciiTexture = nullptr;

void GlText::initTexture()
{
    if (s_asciiImage == nullptr)
    {
        s_asciiImage = helper::io::Image::Create(ASCII_TEXTURE_PATH);
    }
    if (s_asciiTexture == nullptr && s_asciiImage != nullptr)
    {
        s_asciiTexture = new sofa::gl::Texture(s_asciiImage, false, true, false );
    }
}

GlText::GlText()
{
}

GlText::GlText ( const string& text )
{
    this->text = text;
}

GlText::GlText ( const string& text, const type::Vec3& position )
{
    this->text = text;
    this->position = position;
}

GlText::GlText ( const string& text, const type::Vec3& position, const double& scale )
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

void GlText::update ( const type::Vec3& position )
{
    this->position = position;
}

void GlText::update ( const double& scale )
{
    this->scale = scale;
}


void GlText::draw()
{
    glDisable ( GL_LIGHTING );

    const char* s = text.c_str();

    GlText::draw(s, position, scale);
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

    std::vector<Vec3f> vertices;
    std::vector<Vec2f> UVs;

    std::ostringstream oss;
    oss << text;
    const std::string str = oss.str();
    const std::size_t length = str.size();

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

    for (std::size_t j = 0; j < length; j++)
    {
        Vec3f vertex_up_left = Vec3f(j*worldWidth, worldHeight, 0.f);
        Vec3f vertex_up_right = Vec3f(j*worldWidth + worldWidth, worldHeight, 0.f);
        Vec3f vertex_down_right = Vec3f(j*worldWidth + worldWidth, 0.f, 0.f);
        Vec3f vertex_down_left = Vec3f(j*worldWidth, 0.f, 0.f);

        vertices.push_back(vertex_up_left);
        vertices.push_back(vertex_down_left);
        vertices.push_back(vertex_down_right);
        vertices.push_back(vertex_up_right);

        const char character = str[j] - 32;

        float uv_x = (character % nb_char_width) / (float)nb_char_width;
        float uv_y = 1.0f - ((character / nb_char_height) / (float)nb_char_height);

        Vec2f uv_up_left = Vec2f(uv_x, (uv_y - (1.0f / (float)nb_char_height)));
        Vec2f uv_up_right = Vec2f(uv_x + (1.0f / (float)nb_char_width), (uv_y - (1.0f / (float)nb_char_height)));
        Vec2f uv_down_right = Vec2f(uv_x + (1.0f / (float)nb_char_width), uv_y);
        Vec2f uv_down_left = Vec2f(uv_x, uv_y);

        UVs.push_back(uv_up_left);
        UVs.push_back(uv_down_left);
        UVs.push_back(uv_down_right);
        UVs.push_back(uv_up_right);
    }

    glBegin(GL_QUADS);
    for (std::size_t j = 0; j < vertices.size(); j++)
    {
        glTexCoord2fv(UVs[j].data());
        glVertex3fv(vertices[j].data());
    }
    glEnd();

    s_asciiTexture->unbind();
    glDisable(GL_ALPHA_TEST);
    glPopAttrib();

}

void GlText::textureDraw_Indices(const type::vector<type::Vec3>& positions, const float& scale)
{
    if (!s_asciiTexture)
    {
        GlText::initTexture();
        s_asciiTexture->init();
    }
    type::Mat<4, 4, GLfloat> modelviewM;

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

    for (std::size_t i = 0; i < positions.size(); i++)
    {
        std::ostringstream oss;
        oss << i;
        std::string str = oss.str();
        const std::size_t length = str.size();

        std::vector<Vec3f> vertices;
        std::vector<Vec2f> UVs;

        glDisable(GL_LIGHTING);

        glPushMatrix();

        // Makes text always face the viewer by removing the scene rotation
        // get the current modelview matrix
        glGetFloatv(GL_MODELVIEW_MATRIX, modelviewM.ptr());
        modelviewM.transpose();

        type::Vec3f temp(positions[i][0], positions[i][1], positions[i][2]);
        temp = modelviewM.transform(temp);

        glLoadIdentity();
        //translate a little bit to center the text on the position (instead of starting from a top-left position)
        glTranslatef(temp[0] - (worldWidth*length*scale)*0.5f, temp[1] + worldHeight*scale*0.5f, temp[2]);
        glScalef(scale, scale, scale);
        glRotatef(180.0, 1, 0, 0);
        for (std::size_t j = 0; j < length; j++)
        {
            Vec3f vertex_up_left = Vec3f(j*worldWidth, worldHeight, 0.0f);
            Vec3f vertex_up_right = Vec3f(j*worldWidth + worldWidth, worldHeight, 0.0f);
            Vec3f vertex_down_right = Vec3f(j*worldWidth + worldWidth, 0.0f, 0.0f);
            Vec3f vertex_down_left = Vec3f(j*worldWidth, 0.0f, 0.0f);

            vertices.push_back(vertex_up_left);
            vertices.push_back(vertex_down_left);
            vertices.push_back(vertex_down_right);
            vertices.push_back(vertex_up_right);

            const char character = str[j] - 32;

            float uv_x = (character % nb_char_width) / (float)nb_char_width;
            float uv_y = 1.0f - ((character / nb_char_height) / (float)nb_char_height);

            Vec2f uv_up_left = Vec2f(uv_x, (uv_y - (1.0f / (float)nb_char_height)));
            Vec2f uv_up_right = Vec2f(uv_x + (1.0f / (float)nb_char_width), (uv_y - (1.0f / (float)nb_char_height)));
            Vec2f uv_down_right = Vec2f(uv_x + (1.0f / (float)nb_char_width), uv_y);
            Vec2f uv_down_left = Vec2f(uv_x, uv_y);

            UVs.push_back(uv_up_left);
            UVs.push_back(uv_down_left);
            UVs.push_back(uv_down_right);
            UVs.push_back(uv_up_right);
        }

        glBegin(GL_QUADS);
        for (std::size_t j = 0; j < vertices.size(); j++)
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



} // namespace sofa::gl
