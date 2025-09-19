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
#include <sofa/gl/Frame.h>

#include <sofa/gl/gl.h>

#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/type/Quat.h>
#include <sofa/type/RGBAColor.h>

#include <unordered_map>
#include <memory>
#include <numbers>

template <>
struct std::hash<sofa::type::Vec3f>
{
    std::size_t operator()(const sofa::type::Vec3f& k) const
    {
        using std::size_t;
        using std::hash;
        using std::string;

        return ((hash<float>()(k[0])
            ^ (hash<float>()(k[1]) << 1)) >> 1)
            ^ (hash<float>()(k[2]) << 1);
    }
};

constexpr SReal PI = std::numbers::pi_v<SReal>;
constexpr SReal PI2 = 2. * std::numbers::pi_v<SReal>;
constexpr SReal PI_2 = std::numbers::pi_v<SReal> / 2.0;

constexpr SReal constexpr_abs(SReal x) 
{
    return x < 0 ? -x : x;
}


constexpr SReal sin_fast(SReal x) 
{
    while (x > PI) x -= PI2;
    while (x < -PI) x += PI2;

    // Quick approximation using polynomial
    constexpr float B = 4.0f / PI;
    constexpr float C = -4.0f / (PI * PI);
    constexpr float P = 0.225f;

    float y = B * x + C * x * constexpr_abs(x);
    return P * (y * constexpr_abs(y) - y) + y;
}

constexpr SReal cos_fast(SReal x) 
{
    return sin_fast(x + PI_2);
}

namespace sofa::gl
{

// Triangle structure
struct Triangle {
    int v0{}, v1{}, v2{};
};

// Cylinder mesh generator
template<int Segments>
struct CylinderMesh
{
    static constexpr int vertex_count = Segments * 2 + 2; // sides + 2 centers
    static constexpr int triangle_count = Segments * 4;   // sides + caps

    std::array<type::Vec3, vertex_count> vertices;
    std::array<Triangle, triangle_count> triangles;
    std::array<type::Vec3, triangle_count> normals;

    constexpr CylinderMesh(const type::Vec3& start, const type::Vec3& end, double radius)
    {
        const type::Vec3 direction = (end - start).normalized();

        // Create perpendicular vectors
        type::Vec3 up = type::Vec3(0, 1, 0);
        if (constexpr_abs(direction.y()) > 0.9)
        {
            up = type::Vec3(1, 0, 0);
        }

        const type::Vec3 right = up.cross(direction).normalized();
        up = direction.cross(right).normalized();

        // Generate vertices around the cylinder
        int vertex_idx = 0;

        // Side vertices (bottom and top circles)
        for (int i = 0; i < Segments; ++i)
        {
            const double angle = (static_cast<double>(i) / Segments) * 2.0 * std::numbers::pi_v<double>;
            const double cos_a = cos_fast(angle);
            const double sin_a = sin_fast(angle);

            const type::Vec3 offset = right * (cos_a * radius) + up * (sin_a * radius);

            // Bottom circle
            vertices[vertex_idx++] = start + offset;
            // Top circle  
            vertices[vertex_idx++] = end + offset;
        }

        //// Center points for caps
        vertices[vertex_idx++] = start; // Bottom center
        vertices[vertex_idx++] = end;   // Top center

        // Generate triangles
        int triangle_idx = 0;

        // Side triangles
        for (int i = 0; i < Segments; ++i)
        {
            const int bottom_curr = i * 2;
            const int top_curr = i * 2 + 1;
            const int bottom_next = ((i + 1) % Segments) * 2;
            const int top_next = ((i + 1) % Segments) * 2 + 1;

            // Two triangles per side segment
            triangles[triangle_idx++] = Triangle{bottom_curr, top_curr, bottom_next};
            triangles[triangle_idx++] = Triangle{top_curr, top_next, bottom_next};
        }

        // Cap triangles
        const int bottom_center = vertex_count - 2;
        const int top_center = vertex_count - 1;

        for (int i = 0; i < Segments; ++i)
        {
            const int bottom_curr = i * 2;
            const int top_curr = i * 2 + 1;
            const int bottom_next = ((i + 1) % Segments) * 2;
            const int top_next = ((i + 1) % Segments) * 2 + 1;

            // Bottom cap (clockwise from below)
            triangles[triangle_idx++] = Triangle{bottom_center, bottom_next, bottom_curr};
            // Top cap (counter-clockwise from above)
            triangles[triangle_idx++] = Triangle{top_center, top_curr, top_next};
        }

        // compute normals per vertex
        for (int i = 0; i < triangle_count; ++i)
        {
            const auto& tri = triangles[i];
            const type::Vec3 v0 = vertices[tri.v0];
            const type::Vec3 v1 = vertices[tri.v1];
            const type::Vec3 v2 = vertices[tri.v2];
            const type::Vec3 normal = (v1 - v0).cross(v2 - v0) * -1;
            normals[tri.v0] += normal;
            normals[tri.v1] += normal;
            normals[tri.v2] += normal;
        }

        //Not mandatory to do that if GL_NORMALIZE is set
        for (int i = 0; i < vertex_count; ++i)
        {
            normals[i].normalize();
        }
    }
};


// Cone mesh generator
template<int Segments>
struct ConeMesh
{
    static constexpr int vertex_count = Segments + 2; // base circle + tip + base center
    static constexpr int triangle_count = Segments * 2; // sides + base

    std::array<type::Vec3, vertex_count> vertices;
    std::array<Triangle, triangle_count> triangles;
    std::array<type::Vec3, vertex_count> normals{};

    constexpr ConeMesh(const type::Vec3& base, const type::Vec3& tip, double base_radius)
    {
        const type::Vec3 direction = (tip - base).normalized();

        // Create perpendicular vectors
        type::Vec3 up = type::Vec3(0, 1, 0);
        if (constexpr_abs(direction.y()) > 0.9) {
            up = type::Vec3(1, 0, 0);
        }

        type::Vec3 right = up.cross(direction).normalized();
        up = direction.cross(right).normalized();

        // Generate vertices
        int vertex_idx = 0;

        // Tip vertex
        vertices[vertex_idx++] = tip;

        // Base circle vertices
        for (int i = 0; i < Segments; ++i)
        {
            const double angle = (static_cast<double>(i) / Segments) * 2.0 * std::numbers::pi_v<double>;
            const double cos_a = cos_fast(angle);
            const double sin_a = sin_fast(angle);

            const type::Vec3 offset = right * (cos_a * base_radius) + up * (sin_a * base_radius);
            vertices[vertex_idx++] = base + offset;
        }

        // Base center
        vertices[vertex_idx++] = base;

        // Generate triangles
        int triangle_idx = 0;
        int tip_idx = 0;
        int base_center_idx = vertex_count - 1;

        // Side triangles
        for (int i = 0; i < Segments; ++i)
        {
            const int curr = i + 1;
            const int next = (i + 1) % Segments + 1;

            // Side triangle (tip to base edge)
            triangles[triangle_idx++] = Triangle{tip_idx, next, curr};

            // Base triangle (base center to edge)
            triangles[triangle_idx++] = Triangle{base_center_idx, curr, next};
        }

        // compute normals per vertex
        for (int i = 0; i < triangle_count; ++i)
        {
            const auto& tri = triangles[i];
            const type::Vec3 v0 = vertices[tri.v0];
            const type::Vec3 v1 = vertices[tri.v1];
            const type::Vec3 v2 = vertices[tri.v2];
            const type::Vec3 normal = (v1 - v0).cross(v2 - v0) * -1;
            normals[tri.v0] += normal;
            normals[tri.v1] += normal;
            normals[tri.v2] += normal;
        }

        //Not mandatory to do that if GL_NORMALIZE is set
        for (int i = 0; i < vertex_count; ++i)
        {
            normals[i].normalize();
        }
    }
};

// Complete coordinate frame structure
struct CoordinateFrame
{
    // Mesh types
    inline static const int CYLINDER_SEGMENTS = 16;
    using CylMesh = CylinderMesh<CYLINDER_SEGMENTS>;
    using ConMesh = ConeMesh<CYLINDER_SEGMENTS>;

    // Vertex and triangle counts
    static constexpr int cylinder_vertex_count = CylMesh::vertex_count;
    static constexpr int cylinder_triangle_count = CylMesh::triangle_count;
    static constexpr int cone_vertex_count = ConMesh::vertex_count;
    static constexpr int cone_triangle_count = ConMesh::triangle_count;

    static constexpr int total_vertex_count = (cylinder_vertex_count + cone_vertex_count) * 3;
    static constexpr int total_triangle_count = (cylinder_triangle_count + cone_triangle_count) * 3;

    // Meshes for each axis
    CylMesh x_cylinder;
    ConMesh x_arrowhead;
    CylMesh y_cylinder;
    ConMesh y_arrowhead;
    CylMesh z_cylinder;
    ConMesh z_arrowhead;

    struct FrameAxisParameters {
        double cylinderLength{};
        double arrowHeadLength{};
        double cylinderRadius{};
        double arrowHeadRadius{};
    };

    constexpr CoordinateFrame(const FrameAxisParameters& axisX, const FrameAxisParameters& axisY, const FrameAxisParameters& axisZ)
        : x_cylinder(type::Vec3(0, 0, 0), type::Vec3(axisX.cylinderLength, 0, 0), axisX.cylinderRadius)
        , x_arrowhead(type::Vec3(axisX.cylinderLength, 0, 0), type::Vec3(axisX.cylinderLength + axisX.arrowHeadLength, 0, 0), axisX.arrowHeadRadius)
        , y_cylinder(type::Vec3(0, 0, 0), type::Vec3(0, axisY.cylinderLength, 0), axisY.cylinderRadius)
        , y_arrowhead(type::Vec3(0, axisY.cylinderLength, 0), type::Vec3(0, axisY.cylinderLength + axisY.arrowHeadLength, 0), axisY.arrowHeadRadius)
        , z_cylinder(type::Vec3(0, 0, 0), type::Vec3(0, 0, axisZ.cylinderLength), axisZ.cylinderRadius)
        , z_arrowhead(type::Vec3(0, 0, axisZ.cylinderLength), type::Vec3(0, 0, axisZ.cylinderLength +  axisZ.arrowHeadLength), axisZ.arrowHeadRadius)
    {

    }

    // Get mesh data for specific axis components
    struct MeshData {
        const type::Vec3* vertices;
        const Triangle* triangles;
        const type::Vec3* normals;
        int vertex_count;
        int triangle_count;
        int vertex_offset;
    };

    constexpr std::array<MeshData, 6> get_mesh_components() const {
        return { {
            {x_cylinder.vertices.data(), x_cylinder.triangles.data(), x_cylinder.normals.data(), cylinder_vertex_count, cylinder_triangle_count, 0},
            {x_arrowhead.vertices.data(), x_arrowhead.triangles.data(), x_arrowhead.normals.data(), cone_vertex_count, cone_triangle_count, cylinder_vertex_count},
            {y_cylinder.vertices.data(), y_cylinder.triangles.data(), y_cylinder.normals.data(), cylinder_vertex_count, cylinder_triangle_count, cylinder_vertex_count + cone_vertex_count},
            {y_arrowhead.vertices.data(), y_arrowhead.triangles.data(), y_arrowhead.normals.data(), cone_vertex_count, cone_triangle_count, 2 * cylinder_vertex_count + cone_vertex_count},
            {z_cylinder.vertices.data(), z_cylinder.triangles.data(), z_cylinder.normals.data(), cylinder_vertex_count, cylinder_triangle_count, 2 * cylinder_vertex_count + 2 * cone_vertex_count},
            {z_arrowhead.vertices.data(), z_arrowhead.triangles.data(), z_arrowhead.normals.data(), cone_vertex_count, cone_triangle_count, 3 * cylinder_vertex_count + 2 * cone_vertex_count}
        } };
    }
};

// Render the complete coordinate frame
void render_coordinate_frame(const CoordinateFrame& frame, const type::Vec3& center, const type::Quat<SReal>& orient, const type::Vec3&, const type::RGBAColor& colorX, const type::RGBAColor& colorY, const type::RGBAColor& colorZ)
{
    glPushAttrib(GL_LIGHTING_BIT);
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);

    // Get mesh components
    const auto& mesh_components = frame.get_mesh_components();

    // Colors for each axis: Red, Red, Green, Green, Blue, Blue
    const sofa::type::RGBAColor colors[6] = {
        colorX, // X-cylinder (red)
        colorX, // X-arrowhead (red)
        colorY, // Y-cylinder (green)
        colorY, // Y-arrowhead (green)
        colorZ, // Z-cylinder (blue)
        colorZ  // Z-arrowhead (blue)
    };

    sofa::type::Vec3 rotAxis;
    double phi{};
    orient.quatToAxis(rotAxis, phi);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glTranslated(center.x(), center.y(), center.z());
    glRotated(phi * 180.0 / M_PI, 
        rotAxis.x(),
        rotAxis.y(),
        rotAxis.z());

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
        
    for (int i = 0; i < 6; ++i)
    {
        const auto& comp = mesh_components[i];
        glColor4d(colors[i][0], colors[i][1], colors[i][2], colors[i][3]);
        glVertexPointer(3, GL_DOUBLE, 0, comp.vertices);
        glNormalPointer(GL_DOUBLE, 0, comp.normals);
        glDrawElements(GL_TRIANGLES, comp.triangle_count * 3, GL_UNSIGNED_INT, comp.triangles);

        //glDrawArrays(GL_POINTS, 0, comp.vertex_count);
    }

    glPopMatrix();
    glPopAttrib();
}

std::unordered_map < type::Vec3f, CoordinateFrame > cacheCoordinateFrame;

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

        const type::Vec3 l(Lmin / 10_sreal, Lmin / 10_sreal, Lmin / 10_sreal);
        const type::Vec3 lc(Lmax / 5_sreal, Lmax / 5_sreal, Lmax / 5_sreal); // = L / 5;
        const type::Vec3 Lc = lc;

        cacheCoordinateFrame.emplace(len, CoordinateFrame ({ L[0], Lc[0], l[0], lc[0] }, { L[1], Lc[1], l[1], lc[1] }, { L[2], Lc[2], l[2], lc[2] }));
    }

    const auto& frame = cacheCoordinateFrame.at(len);
    render_coordinate_frame(frame, center, orient, len, colorX, colorY, colorZ);
}

void Frame::draw(const type::Vec3& center, const double orient[4][4], const type::Vec3& len, const type::RGBAColor& colorX, const type::RGBAColor& colorY, const type::RGBAColor& colorZ)
{
    const sofa::type::Matrix3 matOrient{ 
        {orient[0][0], orient[0][1] , orient[0][2] },
        {orient[1][0], orient[1][1] , orient[1][2] },
        {orient[2][0], orient[2][1] , orient[2][2] } 
    };
    Quaternion q(sofa::type::QNOINIT);
    q.fromMatrix(matOrient);
    draw(center, q, len, colorX, colorY, colorZ);
}

void Frame::draw(const type::Vec3& center, const Quaternion& orient, SReal len, const type::RGBAColor& colorX, const type::RGBAColor& colorY, const type::RGBAColor& colorZ)
{
    draw(center, orient, { len, len, len }, colorX, colorY, colorZ);
}

void Frame::draw(const type::Vec3& center, const double orient[4][4], SReal len, const type::RGBAColor& colorX, const type::RGBAColor& colorY, const type::RGBAColor& colorZ)
{
    const sofa::type::Matrix3 matOrient{
        {orient[0][0], orient[0][1] , orient[0][2] },
        {orient[1][0], orient[1][1] , orient[1][2] },
        {orient[2][0], orient[2][1] , orient[2][2] }
    };

    Quaternion q(sofa::type::QNOINIT);
    q.fromMatrix(matOrient);
    draw(center, q, {len, len, len}, colorX, colorY, colorZ);
}

} // namespace sofa::gl
