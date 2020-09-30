/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <map>
#include <sofa/helper/gl/template.h>
#include <sofa/core/ObjectFactory.h>
#include <SofaOpenglVisual/PointSplatRenderer.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/RGBAColor.h>

namespace sofa::component::visualmodel
{

static int PointSplatRendererClass = core::RegisterObject("Render a set of points using point splatting technique.")
        .add< PointSplatRenderer >();


PointSplatRenderer::PointSplatRenderer() //const std::string &name, std::string filename, std::string loader, std::string textureName)
    : d_radius(initData(&d_radius, 1.0f, "radius", "Radius of the spheres.")),
      d_textureSize(initData(&d_textureSize, size_t{32}, "textureSize", "Size of the billboard texture.")),
      d_points(initData(&d_points, "points", "Positions of the splats")),
      d_normals(initData(&d_normals,"normals", "Normal for the splats")),
      d_colors(initData(&d_colors, "colors", "Colors for the splats")),
      texture_data(nullptr)
{
}

PointSplatRenderer::~PointSplatRenderer()
{
    if(texture_data != nullptr)
        delete [] texture_data;
}

void PointSplatRenderer::init()
{
    reinit();
}

void PointSplatRenderer::reinit()
{
    if(texture_data != nullptr)
        delete [] texture_data;

    size_t texture_size = d_textureSize.getValue();
    size_t half_texture_size = texture_size >> 1;
    texture_data = new unsigned char [texture_size * texture_size];

    for(unsigned int i=0; i<half_texture_size; ++i)
    {
        for(unsigned int j=0; j<half_texture_size; ++j)
        {
            const double x = i / double(half_texture_size);
            const double y = j / double(half_texture_size);
            const double dist = sqrt(x*x + y*y);
            const double value = cos(M_PI_2 * dist);
            const unsigned char texValue = (value < 0.0f) ? 0 : (unsigned char)(255.0f * 0.5 * value);

            texture_data[half_texture_size + i + (half_texture_size + j) * texture_size] = texValue;
            texture_data[half_texture_size + i + (half_texture_size - 1 - j) * texture_size] = texValue;
            texture_data[half_texture_size - 1 - i + (half_texture_size + j) * texture_size] = texValue;
            texture_data[half_texture_size - 1 - i + (half_texture_size - 1 - j) * texture_size] = texValue;
        }
    }
}

void PointSplatRenderer::drawTransparent(const core::visual::VisualParams* vparams)
{
    /// If we hide the visual models then we can just quit right now.
    if(!vparams->displayFlags().getShowVisualModels())
        return;

    glPushAttrib(GL_ENABLE_BIT);

    glDisable(GL_LIGHTING);
    glEnable (GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

    glEnable (GL_TEXTURE_2D);
    glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    GLuint	texture;
    size_t texture_size = d_textureSize.getValue();

    /// bind the texture.
    glGenTextures (1, &texture);
    glBindTexture (GL_TEXTURE_2D, texture);
    glTexImage2D (GL_TEXTURE_2D, 0, GL_ALPHA, texture_size, texture_size, 0,
            GL_ALPHA, GL_UNSIGNED_BYTE, texture_data);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    auto vertices = sofa::helper::getReadAccessor(d_points);
    auto colors = sofa::helper::getWriteAccessor(d_colors);

    if( colors.size() < vertices.size() )
        colors.resize(vertices.size());

    glPushMatrix ();

    double mat[16];
    glGetDoublev( GL_MODELVIEW_MATRIX, mat );

    Vec3 vRight( mat[0], mat[4], mat[8] );
    Vec3 vUp( mat[1], mat[5], mat[9] );
    Vec3 vView(vRight.cross(vUp));
    using Real = Vec3::value_type;

    std::multimap< Real, unsigned int, std::less<Real> >	mapPnt;
    for(unsigned int i=0; i<vertices.size(); ++i)
    {
        const Vec3&	vertex = vertices[i];
        Real dist(vertex * vView);

        mapPnt.insert(std::pair<Real, unsigned int>(dist, i));
    }

    const float radius = d_radius.getValue();

    for(std::multimap<Real, unsigned int>::const_iterator it = mapPnt.begin();
        it != mapPnt.end(); ++it)
    {
        size_t i = (*it).second;

        const Vec3& center = vertices[i];
        const RGBAColor& color = colors[i];

        /// Now, build a quad around the center point based on the vRight
        /// and vUp vectors. This will guarantee that the quad will be
        /// orthogonal to the view.
        Vec3 vPoint0(center + ((-vRight - vUp) * radius));
        Vec3 vPoint1(center + (( vRight - vUp) * radius));
        Vec3 vPoint2(center + (( vRight + vUp) * radius));
        Vec3 vPoint3(center + ((-vRight + vUp) * radius));

        glColor4f (color.r(), color.g(), color.b(), color.a());

        glBegin( GL_QUADS );
        glTexCoord2f(0.0f, 0.0f);
        glVertex3f((float) vPoint0[0], (float) vPoint0[1], (float) vPoint0[2]);
        glTexCoord2f(1.0f, 0.0f);
        glVertex3f((float) vPoint1[0], (float) vPoint1[1], (float) vPoint1[2]);
        glTexCoord2f(1.0f, 1.0f);
        glVertex3f((float) vPoint2[0], (float) vPoint2[1], (float) vPoint2[2]);
        glTexCoord2f(0.0f, 1.0f);
        glVertex3f((float) vPoint3[0], (float) vPoint3[1], (float) vPoint3[2]);
        glEnd();
    }

    /// restore the previously stored modelview matrix
    glPopMatrix ();
    glPopAttrib();
}

} /// namespace sofa::component::visualmodel

