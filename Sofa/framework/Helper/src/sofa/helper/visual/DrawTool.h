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

#include <sofa/helper/config.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Quat.h>
#include <vector>

namespace sofa::helper::visual
{

/**
 *  \brief Utility class to perform debug drawing.
 *
 *  Class which contains a set of methods to perform minimal debug drawing regardless of the graphics API used.
 *  Components receive a pointer to the DrawTool through the VisualParams parameter of their draw method.
 *  Sofa provides a default concrete implementation of this class for the OpenGL API with the DrawToolGL class.
 *k
 */

class SOFA_HELPER_API DrawTool
{

public:
    typedef sofa::type::RGBAColor RGBAColor;
    typedef sofa::type::Vec3f Vec3f;
    typedef sofa::type::Vec3 Vec3;
    typedef sofa::type::Vec<3,int> Vec3i;
    typedef sofa::type::Vec<2,int> Vec2i;
    typedef sofa::type::Quat<SReal> Quaternion;

    DrawTool() { clear(); }
    virtual ~DrawTool() {}

    virtual void init() = 0;

    /// @name Primitive rendering methods
    /// @{
    virtual void drawPoints(const std::vector<Vec3> &points, float size,  const  RGBAColor& color) = 0 ;
    virtual void drawPoints(const std::vector<Vec3> &points, float size, const std::vector<RGBAColor>& color) = 0;

    virtual void drawLine(const Vec3 &p1, const Vec3 &p2, const RGBAColor& color) =  0;
    virtual void drawInfiniteLine(const Vec3 &point, const Vec3 &direction, const RGBAColor& color) = 0;
    virtual void drawLines(const std::vector<Vec3> &points, float size, const RGBAColor& color) = 0 ;
    virtual void drawLines(const std::vector<Vec3> &points, float size, const std::vector<RGBAColor>& colors) = 0 ;
    virtual void drawLines(const std::vector<Vec3> &points, const std::vector< Vec2i > &index , float size, const RGBAColor& color) = 0 ;

    virtual void drawLineStrip(const std::vector<Vec3> &points, float size, const RGBAColor& color) = 0 ;
    virtual void drawLineLoop(const std::vector<Vec3> &points, float size, const RGBAColor& color) = 0 ;

    virtual void drawDisk(float radius, double from, double to, int resolution, const RGBAColor& color) = 0;
    virtual void drawCircle(float radius, float lineThickness, int resolution, const RGBAColor& color) = 0;

    virtual void drawTriangles(const std::vector<Vec3> &points, const RGBAColor& color) = 0 ;
    virtual void drawTriangles(const std::vector<Vec3> &points, const Vec3& normal, const RGBAColor& color) = 0 ;
    virtual void drawTriangles(const std::vector<Vec3> &points,
            const std::vector< Vec3i > &index,
            const std::vector<Vec3>  &normal,
            const RGBAColor& color) = 0 ;
    virtual void drawTriangles(const std::vector<Vec3> &points,
        const std::vector< Vec3i > &index,
        const std::vector<Vec3>  &normal,
        const std::vector<RGBAColor>& color) = 0;
    virtual void drawTriangles(const std::vector<Vec3> &points,
            const std::vector< RGBAColor > &color) = 0 ;
    virtual void drawTriangles(const std::vector<Vec3> &points,
            const std::vector<Vec3>  &normal,
            const std::vector< RGBAColor > &color) = 0 ;
    virtual void drawTriangleStrip(const std::vector<Vec3> &points,
            const std::vector<Vec3>  &normal,
            const RGBAColor& color) = 0 ;
    virtual void drawTriangleFan(const std::vector<Vec3> &points,
            const std::vector<Vec3>  &normal,
            const RGBAColor& color) = 0 ;



    virtual void drawFrame   (const Vec3& position, const Quaternion &orientation, const Vec3f &size) = 0 ;
    virtual void drawFrame   (const Vec3& position, const Quaternion &orientation, const Vec3f &size, const RGBAColor &color) = 0 ;

    virtual void drawSpheres (const std::vector<Vec3> &points, const std::vector<float>& radius, const RGBAColor& color) = 0;
    virtual void drawSpheres (const std::vector<Vec3> &points, float radius, const RGBAColor& color) = 0 ;
    virtual void drawFakeSpheres(const std::vector<Vec3> &points, const std::vector<float>& radius, const RGBAColor& color) = 0;
    virtual void drawFakeSpheres(const std::vector<Vec3> &points, float radius, const RGBAColor& color) = 0;

    virtual void drawCone    (const Vec3& p1, const Vec3 &p2, float radius1, float radius2, const RGBAColor& color, int subd=16) = 0 ;

    /// Draw a cube of size one centered on the current point.
    virtual void drawCube    (const float& radius, const RGBAColor& color, const int& subd=16) = 0 ;

    virtual void drawCylinder(const Vec3& p1, const Vec3 &p2, float radius, const RGBAColor& color,  int subd=16) = 0 ;

    virtual void drawCapsule(const Vec3& p1, const Vec3 &p2, float radius, const RGBAColor& color,  int subd=16) = 0 ;

    virtual void drawArrow   (const Vec3& p1, const Vec3 &p2, float radius, const RGBAColor& color,  int subd=16) = 0 ;
    virtual void drawArrow   (const Vec3& p1, const Vec3 &p2, float radius, float coneLength, const RGBAColor& color,  int subd=16) = 0 ;
    virtual void drawArrow   (const Vec3& p1, const Vec3 &p2, float radius, float coneLength, float coneRadius, const RGBAColor& color,  int subd=16) = 0;

    /// Draw a cross (3 lines) centered on p
    virtual void drawCross(const Vec3&p, float length, const RGBAColor& color) = 0;

    /// Draw a plus sign of size one centered on the current point.
    virtual void drawPlus    (const float& radius, const RGBAColor& color, const int& subd=16) = 0 ;

    virtual void drawPoint(const Vec3 &p, const RGBAColor &c) = 0 ;
    virtual void drawPoint(const Vec3 &p, const Vec3 &n, const RGBAColor &c) = 0 ;

    virtual void drawTriangle(const Vec3 &p1,const Vec3 &p2,const Vec3 &p3,
            const Vec3 &normal) = 0 ;
    virtual void drawTriangle(const Vec3 &p1,const Vec3 &p2,const Vec3 &p3,
            const Vec3 &normal, const RGBAColor &c) = 0 ;
    virtual void drawTriangle(const Vec3 &p1,const Vec3 &p2,const Vec3 &p3,
            const Vec3 &normal,
            const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3) = 0 ;
    virtual void drawTriangle(const Vec3 &p1,const Vec3 &p2,const Vec3 &p3,
            const Vec3 &normal1, const Vec3 &normal2, const Vec3 &normal3,
            const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3) = 0 ;

    virtual void drawQuad(const Vec3 &p1,const Vec3 &p2,const Vec3 &p3,const Vec3 &p4,
            const Vec3 &normal) = 0 ;
    virtual void drawQuad(const Vec3 &p1,const Vec3 &p2,const Vec3 &p3,const Vec3 &p4,
            const Vec3 &normal, const RGBAColor &c) = 0 ;
    virtual void drawQuad(const Vec3 &p1,const Vec3 &p2,const Vec3 &p3,const Vec3 &p4,
            const Vec3 &normal,
            const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3, const RGBAColor &c4) = 0 ;
    virtual void drawQuad(const Vec3 &p1,const Vec3 &p2,const Vec3 &p3,const Vec3 &p4,
            const Vec3 &normal1, const Vec3 &normal2, const Vec3 &normal3, const Vec3 &normal4,
            const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3, const RGBAColor &c4) = 0 ;
    virtual void drawQuads(const std::vector<Vec3> &points, const RGBAColor& color) = 0 ;
    virtual void drawQuads(const std::vector<Vec3> &points, const std::vector<RGBAColor>& colors) = 0 ;

    virtual void drawTetrahedron(const Vec3 &p0, const Vec3 &p1, const Vec3 &p2, const Vec3 &p3, const RGBAColor &color) = 0 ;
    virtual void drawScaledTetrahedron(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, const RGBAColor& color, const float scale) = 0;
    virtual void drawTetrahedra(const std::vector<Vec3> &points, const RGBAColor& color) = 0;
    //Scale each tetrahedron
    virtual void drawScaledTetrahedra(const std::vector<Vec3> &points, const RGBAColor& color, const float scale) = 0;

    virtual void drawHexahedron(const Vec3 &p0, const Vec3 &p1, const Vec3 &p2, const Vec3 &p3,
        const Vec3 &p4, const Vec3 &p5, const Vec3 &p6, const Vec3 &p7, const RGBAColor &color) = 0;
    virtual void drawHexahedra(const std::vector<Vec3> &points, const RGBAColor& color) = 0;
    //Scale each hexahedron
    virtual void drawScaledHexahedra(const std::vector<Vec3> &points, const RGBAColor& color, const float scale) = 0;

    virtual void drawSphere( const Vec3 &p, float radius) = 0 ;
    virtual void drawSphere(const Vec3 &p, float radius, const RGBAColor& color) = 0;
    virtual void drawEllipsoid(const Vec3 &p, const Vec3 &radii) = 0;

    virtual void drawBoundingBox( const Vec3 &min, const Vec3 &max, float size = 1.0 ) = 0;

    virtual void draw3DText(const Vec3 &p, float scale, const RGBAColor &color, const char* text) = 0;
    virtual void draw3DText_Indices(const std::vector<Vec3> &positions, float scale, const RGBAColor &color) = 0;
    /// @}

    /// @name Transformation methods.
    /// @{
    virtual void pushMatrix() = 0;

    virtual void popMatrix() =  0;

    virtual void multMatrix(float*  ) = 0;

    virtual void scale(float ) = 0;
    virtual void translate(float x, float y, float z) = 0;
    /// @}

    /// @name Drawing style methods.
    virtual void setMaterial(const RGBAColor &color) = 0 ;

    virtual void resetMaterial(const RGBAColor &color) = 0 ;
    virtual void resetMaterial() = 0 ;

    virtual void setPolygonMode(int _mode, bool _wireframe) = 0 ;

    virtual void setLightingEnabled(bool _isAnabled) = 0 ;
    /// @}

    virtual void enableBlending() = 0;
    virtual void disableBlending() = 0;

    virtual void enableLighting() = 0;
    virtual void disableLighting() = 0;

    virtual void enableDepthTest() = 0;
    virtual void disableDepthTest() = 0;

    /// @name States (save/restore)
    virtual void saveLastState() = 0;
    virtual void restoreLastState() = 0;

    struct StateLifeCycle;

    /// Use RAII to bind state save and restore to the returned object: saveLastState is called
    /// at object creation, and restoreLastState is called when the object is destroyed.
    [[nodiscard]]
    StateLifeCycle makeStateLifeCycle();

    /// @name Overlay methods

    /// draw 2D text at position (x,y) from top-left corner
    virtual void writeOverlayText( int x, int y, unsigned fontSize, const RGBAColor &color, const char* text ) = 0;

    /// Allow a variable depth offset for polygon drawing
    virtual void enablePolygonOffset(float factor, float units) = 0;
    /// Remove variable depth offset for polygon drawing
    virtual void disablePolygonOffset() = 0;

    // @name Color Buffer method
    virtual void readPixels(int x, int y, int w, int h, float* rgb, float* z = nullptr) = 0;
    /// @}

    virtual void clear() {}

    /// Compatibility wrapper functions
    using Vec4f = sofa::type::Vec4f;
};

struct SOFA_HELPER_API DrawTool::StateLifeCycle
{
    StateLifeCycle() = delete;
    StateLifeCycle(const StateLifeCycle&) = delete;
    StateLifeCycle(DrawTool* drawTool);
    ~StateLifeCycle();
private:
    DrawTool* m_drawTool { nullptr };
};

} // namespace sofa::helper::visual
