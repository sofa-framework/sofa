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
#include <sofa/gl/NewAxis.h>

#include <sofa/gl/gl.h>

#include <cassert>
#include <algorithm>
#include <iostream>

constexpr int CYLINDER_SEGMENTS = 16;

namespace sofa::gl
{

std::map < type::Vec3f, CoordinateFrame > cacheCoordinateFrame;

// Function to calculate normal for a triangle
type::Vec3 calculate_normal(const type::Vec3& v0, const type::Vec3& v1, const type::Vec3& v2) {
    type::Vec3 edge1 = v1 - v0;
    type::Vec3 edge2 = v2 - v0;
    type::Vec3 normal = edge1.cross(edge2);
    return normal.normalized();
}

// Render the complete coordinate frame
void render_coordinate_frame(const CoordinateFrame& frame, const type::Vec3& center, const type::Quat<SReal>& orient, const type::Vec3& len, const type::RGBAColor& colorX, const type::RGBAColor& colorY, const type::RGBAColor& colorZ)
{
    glPushAttrib(GL_LIGHTING_BIT);
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);

    // Get mesh components
    const auto& mesh_components = frame.get_mesh_components();

    // Colors for each axis: Red, Red, Green, Green, Blue, Blue
    const sofa::type::RGBAColor colors[6] = {
        colorX, // X-cylinder (red)
        colorX, // X-arrowhead (bright red)
        colorY, // Y-cylinder (green)
        colorY, // Y-arrowhead (bright green)
        colorZ, // Z-cylinder (blue)
        colorZ  // Z-arrowhead (bright blue)
    };

    sofa::type::Vec3 rotAxis;
    double phi{};
    orient.quatToAxis(rotAxis, phi);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glTranslatef(center.x(), center.y(), center.z());
    glRotated(phi * 180.0 / M_PI, 
        rotAxis.x(),
        rotAxis.y(),
        rotAxis.z());
    

    glBegin(GL_TRIANGLES);
    // Render each component with its color
    for (int i = 0; i < 6; ++i) 
    {
        const auto& comp = mesh_components[i];

        glColor3f(colors[i][0], colors[i][1], colors[i][2]);

        for (int i = 0; i < comp.triangle_count; ++i) 
        {
            const Triangle& tri = comp.triangles[i];
            const type::Vec3& v0 = comp.vertices[tri.v0];
            const type::Vec3& v1 = comp.vertices[tri.v1];
            const type::Vec3& v2 = comp.vertices[tri.v2];

            // Calculate and set normal for lighting
            const type::Vec3 normal = calculate_normal(v0, v1, v2);
            glNormal3f(normal.x(), normal.y(), normal.z());

            // Emit vertices
            glVertex3f(v0.x(), v0.y(), v0.z());
            glVertex3f(v1.x(), v1.y(), v1.z());
            glVertex3f(v2.x(), v2.y(), v2.z());
        }

    }
    glEnd();

    glPopMatrix();
    glPopAttrib();
}

void Frame::draw(const type::Vec3& center, const Quaternion& orient, const type::Vec3& len, const type::RGBAColor& colorX, const type::RGBAColor& colorY, const type::RGBAColor& colorZ )
{
    if (cacheCoordinateFrame.find(len) == cacheCoordinateFrame.end())
    {
        type::Vec3 L = len;

        SReal Lmin = L[0];
        if (L[1] < Lmin) Lmin = L[1];
        if (L[2] < Lmin) Lmin = L[2];

        SReal Lmax = L[0];
        if (L[1] > Lmax) Lmax = L[1];
        if (L[2] > Lmax) Lmax = L[2];

        if (Lmax > Lmin * 2 && Lmin > 0.0)
            Lmax = Lmin * 2;

        if (Lmax > Lmin * 2)
            Lmin = Lmax / 1.414_sreal;

        type::Vec3 l(Lmin / 10_sreal, Lmin / 10_sreal, Lmin / 10_sreal);
        type::Vec3 lc(Lmax / 5_sreal, Lmax / 5_sreal, Lmax / 5_sreal); // = L / 5;
        type::Vec3 Lc = lc;

        CoordinateFrame frame({ L[0], Lc[0], l[0], lc[0] }, { L[1], Lc[1], l[1], lc[1] }, { L[2], Lc[2], l[2], lc[2] });

        cacheCoordinateFrame.emplace(len, frame);
    }

    const auto& frame = cacheCoordinateFrame.at(len);
    render_coordinate_frame(frame, center, orient, len, colorX, colorY, colorZ);
}

void Frame::draw(const type::Vec3& center, const double orient[4][4], const type::Vec3& len, const type::RGBAColor& colorX, const type::RGBAColor& colorY, const type::RGBAColor& colorZ)
{

}

void Frame::draw(const double *mat, const type::Vec3& len, const type::RGBAColor& colorX, const type::RGBAColor& colorY, const type::RGBAColor& colorZ)
{

}

void Frame::draw(const type::Vec3& center, const Quaternion& orient, SReal len, const type::RGBAColor& colorX, const type::RGBAColor& colorY, const type::RGBAColor& colorZ)
{

}

void Frame::draw(const type::Vec3& center, const double orient[4][4], SReal len, const type::RGBAColor& colorX, const type::RGBAColor& colorY, const type::RGBAColor& colorZ)
{

}

void Frame::draw(const double *mat, SReal len, const type::RGBAColor& colorX, const type::RGBAColor& colorY, const type::RGBAColor& colorZ)
{

}

} // namespace sofa::gl
