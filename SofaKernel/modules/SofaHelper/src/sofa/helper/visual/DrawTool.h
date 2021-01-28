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

#include <sofa/core/config.h>
#include <sofa/helper/types/RGBAColor.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/fwd.h>

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

class DrawTool
{

public:
    typedef sofa::helper::types::RGBAColor RGBAColor;
    typedef sofa::defaulttype::Vec3f   Vec3f;
    typedef sofa::defaulttype::Vector3 Vector3;
    typedef sofa::defaulttype::Vec<3,int> Vec3i;
    typedef sofa::defaulttype::Vec<2,int> Vec2i;
    typedef sofa::defaulttype::Quaternion Quaternion;

    DrawTool() { clear(); }
    virtual ~DrawTool() {}

    virtual void init() = 0;

    /// @name Primitive rendering methods
    /// @{
    virtual void drawPoints(const std::vector<Vector3> &points, float size,  const  RGBAColor& colour) = 0 ;
    virtual void drawPoints(const std::vector<Vector3> &points, float size, const std::vector<RGBAColor>& colour) = 0;

    virtual void drawLine(const Vector3 &p1, const Vector3 &p2, const RGBAColor& colour) =  0;
    virtual void drawInfiniteLine(const Vector3 &point, const Vector3 &direction, const RGBAColor& color) = 0;
    virtual void drawLines(const std::vector<Vector3> &points, float size, const RGBAColor& colour) = 0 ;
    virtual void drawLines(const std::vector<Vector3> &points, float size, const std::vector<RGBAColor>& colours) = 0 ;
    virtual void drawLines(const std::vector<Vector3> &points, const std::vector< Vec2i > &index , float size, const RGBAColor& colour) = 0 ;

    virtual void drawLineStrip(const std::vector<Vector3> &points, float size, const RGBAColor& colour) = 0 ;
    virtual void drawLineLoop(const std::vector<Vector3> &points, float size, const RGBAColor& colour) = 0 ;

    virtual void drawDisk(float radius, double from, double to, int resolution, const RGBAColor& color) = 0;
    virtual void drawCircle(float radius, float lineThickness, int resolution, const RGBAColor& color) = 0;

    virtual void drawTriangles(const std::vector<Vector3> &points, const RGBAColor& colour) = 0 ;
    virtual void drawTriangles(const std::vector<Vector3> &points, const Vector3& normal, const RGBAColor& colour) = 0 ;
    virtual void drawTriangles(const std::vector<Vector3> &points,
            const std::vector< Vec3i > &index,
            const std::vector<Vector3>  &normal,
            const RGBAColor& colour) = 0 ;
    virtual void drawTriangles(const std::vector<Vector3> &points,
        const std::vector< Vec3i > &index,
        const std::vector<Vector3>  &normal,
        const std::vector<RGBAColor>& colour) = 0;
    virtual void drawTriangles(const std::vector<Vector3> &points,
            const std::vector< RGBAColor > &colour) = 0 ;
    virtual void drawTriangles(const std::vector<Vector3> &points,
            const std::vector<Vector3>  &normal,
            const std::vector< RGBAColor > &colour) = 0 ;
    virtual void drawTriangleStrip(const std::vector<Vector3> &points,
            const std::vector<Vector3>  &normal,
            const RGBAColor& colour) = 0 ;
    virtual void drawTriangleFan(const std::vector<Vector3> &points,
            const std::vector<Vector3>  &normal,
            const RGBAColor& colour) = 0 ;



    virtual void drawFrame   (const Vector3& position, const Quaternion &orientation, const Vec3f &size) = 0 ;
    virtual void drawFrame   (const Vector3& position, const Quaternion &orientation, const Vec3f &size, const RGBAColor &colour) = 0 ;

    virtual void drawSpheres (const std::vector<Vector3> &points, const std::vector<float>& radius, const RGBAColor& colour) = 0;
    virtual void drawSpheres (const std::vector<Vector3> &points, float radius, const RGBAColor& colour) = 0 ;
    virtual void drawFakeSpheres(const std::vector<Vector3> &points, const std::vector<float>& radius, const RGBAColor& colour) = 0;
    virtual void drawFakeSpheres(const std::vector<Vector3> &points, float radius, const RGBAColor& colour) = 0;

    virtual void drawCone    (const Vector3& p1, const Vector3 &p2, float radius1, float radius2, const RGBAColor& colour, int subd=16) = 0 ;

    /// Draw a cube of size one centered on the current point.
    virtual void drawCube    (const float& radius, const RGBAColor& colour, const int& subd=16) = 0 ;

    virtual void drawCylinder(const Vector3& p1, const Vector3 &p2, float radius, const RGBAColor& colour,  int subd=16) = 0 ;

    virtual void drawCapsule(const Vector3& p1, const Vector3 &p2, float radius, const RGBAColor& colour,  int subd=16) = 0 ;

    virtual void drawArrow   (const Vector3& p1, const Vector3 &p2, float radius, const RGBAColor& colour,  int subd=16) = 0 ;
    virtual void drawArrow   (const Vector3& p1, const Vector3 &p2, float radius, float coneLength, const RGBAColor& colour,  int subd=16) = 0 ;
    virtual void drawArrow   (const Vector3& p1, const Vector3 &p2, float radius, float coneLength, float coneRadius, const RGBAColor& color,  int subd=16) = 0;

    /// Draw a cross (3 lines) centered on p
    virtual void drawCross(const Vector3&p, float length, const RGBAColor& colour) = 0;

    /// Draw a plus sign of size one centered on the current point.
    virtual void drawPlus    (const float& radius, const RGBAColor& colour, const int& subd=16) = 0 ;

    virtual void drawPoint(const Vector3 &p, const RGBAColor &c) = 0 ;
    virtual void drawPoint(const Vector3 &p, const Vector3 &n, const RGBAColor &c) = 0 ;

    virtual void drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
            const Vector3 &normal) = 0 ;
    virtual void drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
            const Vector3 &normal, const RGBAColor &c) = 0 ;
    virtual void drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
            const Vector3 &normal,
            const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3) = 0 ;
    virtual void drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
            const Vector3 &normal1, const Vector3 &normal2, const Vector3 &normal3,
            const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3) = 0 ;

    virtual void drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
            const Vector3 &normal) = 0 ;
    virtual void drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
            const Vector3 &normal, const RGBAColor &c) = 0 ;
    virtual void drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
            const Vector3 &normal,
            const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3, const RGBAColor &c4) = 0 ;
    virtual void drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
            const Vector3 &normal1, const Vector3 &normal2, const Vector3 &normal3, const Vector3 &normal4,
            const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3, const RGBAColor &c4) = 0 ;
    virtual void drawQuads(const std::vector<Vector3> &points, const RGBAColor& colour) = 0 ;
    virtual void drawQuads(const std::vector<Vector3> &points, const std::vector<RGBAColor>& colours) = 0 ;

    virtual void drawTetrahedron(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2, const Vector3 &p3, const RGBAColor &colour) = 0 ;
    virtual void drawTetrahedra(const std::vector<Vector3> &points, const RGBAColor& colour) = 0;
    //Scale each tetrahedron
    virtual void drawScaledTetrahedra(const std::vector<Vector3> &points, const RGBAColor& colour, const float scale) = 0;

    virtual void drawHexahedron(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2, const Vector3 &p3,
        const Vector3 &p4, const Vector3 &p5, const Vector3 &p6, const Vector3 &p7, const RGBAColor &colour) = 0;
    virtual void drawHexahedra(const std::vector<Vector3> &points, const RGBAColor& colour) = 0;
    //Scale each hexahedron
    virtual void drawScaledHexahedra(const std::vector<Vector3> &points, const RGBAColor& colour, const float scale) = 0;

    virtual void drawSphere( const Vector3 &p, float radius) = 0 ;
    virtual void drawSphere(const Vector3 &p, float radius, const RGBAColor& colour) = 0;
    virtual void drawEllipsoid(const Vector3 &p, const Vector3 &radii) = 0;

    virtual void drawBoundingBox( const Vector3 &min, const Vector3 &max, float size = 1.0 ) = 0;

    virtual void draw3DText(const Vector3 &p, float scale, const RGBAColor &color, const char* text) = 0;
    virtual void draw3DText_Indices(const std::vector<Vector3> &positions, float scale, const RGBAColor &color) = 0;
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
    virtual void setMaterial(const RGBAColor &colour) = 0 ;

    virtual void resetMaterial(const RGBAColor &colour) = 0 ;
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
    using Vec4f = sofa::defaulttype::Vec4f;
#define DEPRECATE_VEC4F \
    [[deprecated("This function has been deprecated in #PR 1626. The function will be removed " \
    "in the v21.06 release. Vec4f defining a color is deprecated, use RGBAColor instead.")]]


    // Necessary to not break existing code
    // as std::vector<RGBAColor> is not a std::vector<Vec4f>
    DEPRECATE_VEC4F
    void drawPoints(const std::vector<Vector3>& points, float size, const std::vector<Vec4f>& colour)
    {
        std::vector<RGBAColor> rgbaColours;
        std::copy(colour.begin(), colour.end(), rgbaColours.begin());
        drawPoints(points, size, rgbaColours);
    }
    
    DEPRECATE_VEC4F
    void drawLines(const std::vector<Vector3>& points, float size, const std::vector<Vec4f>& colours)
    {
        std::vector<RGBAColor> rgbaColours;
        std::copy(colours.begin(), colours.end(), rgbaColours.begin());
        drawLines(points, size, rgbaColours);
    }

    DEPRECATE_VEC4F
    void drawTriangles(const std::vector<Vector3>& points, const std::vector< Vec3i >& index, const std::vector<Vector3>& normal, const std::vector<Vec4f>& colour)
    {
        std::vector<RGBAColor> rgbaColours;
        std::copy(colour.begin(), colour.end(), rgbaColours.begin());
        drawTriangles(points, index, normal, rgbaColours);
    }


    DEPRECATE_VEC4F
    void drawTriangles(const std::vector<Vector3>& points, const std::vector< Vec4f >& colour)
    {
        std::vector<RGBAColor> rgbaColours;
        std::copy(colour.begin(), colour.end(), rgbaColours.begin());
        drawTriangles(points, rgbaColours);
    }

    DEPRECATE_VEC4F
    void drawTriangles(const std::vector<Vector3>& points,
            const std::vector<Vector3>& normal,
            const std::vector< Vec4f >& colour)
    {
        std::vector<RGBAColor> rgbaColours;
        std::copy(colour.begin(), colour.end(), rgbaColours.begin());
        drawTriangles(points, normal, rgbaColours);
    }

    DEPRECATE_VEC4F
    void drawQuads(const std::vector<Vector3>& points, const std::vector<Vec4f>& colours)
    {        
        std::vector<RGBAColor> rgbaColours;
        std::copy(colours.begin(), colours.end(), rgbaColours.begin());
        drawQuads(points,rgbaColours);
    }

    ///////
    // Just for the deprecation
    DEPRECATE_VEC4F
    void drawPoints(const std::vector<Vector3>& points, float size, const Vec4f& colour)
    {
        drawPoints(points, size, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawLine(const Vector3& p1, const Vector3& p2, const Vec4f& colour)
    {
        drawLine(p1, p2, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawInfiniteLine(const Vector3& point, const Vector3& direction, const Vec4f& color)
    {
        drawInfiniteLine(point, direction, RGBAColor(color));
    }

    DEPRECATE_VEC4F
    void drawLines(const std::vector<Vector3>& points, float size, const Vec4f& colour)
    {
        drawLines(points, size, RGBAColor(colour));
    }


    void drawLines(const std::vector<Vector3>& points, const std::vector< Vec2i >& index, float size, const Vec4f& colour)
    {
        drawLines(points, index, size, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawLineStrip(const std::vector<Vector3>& points, float size, const Vec4f& colour)
    {
        drawLineStrip(points, size, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawLineLoop(const std::vector<Vector3>& points, float size, const Vec4f& colour)
    {
        drawLineLoop(points, size, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawDisk(float radius, double from, double to, int resolution, const Vec4f& color) 
    {
        drawDisk(radius, from, to, resolution, RGBAColor(color));
    }

    DEPRECATE_VEC4F
    void drawCircle(float radius, float lineThickness, int resolution, const Vec4f& color)
    {
        drawCircle(radius, lineThickness, resolution, RGBAColor(color));
    }

    DEPRECATE_VEC4F
    void drawTriangles(const std::vector<Vector3>& points, const Vec4f& colour)
    {
        drawTriangles(points, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawTriangles(const std::vector<Vector3>& points, const Vector3& normal, const Vec4f& colour)
    {
        drawTriangles(points, normal, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawTriangles(const std::vector<Vector3>& points,
        const std::vector< Vec3i >& index,
        const std::vector<Vector3>& normal,
        const Vec4f& colour)
    {
        drawTriangles(points, index, normal, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawTriangleStrip(const std::vector<Vector3>& points,
        const std::vector<Vector3>& normal,
        const Vec4f& colour)
    {
        drawTriangleStrip(points, normal, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawTriangleFan(const std::vector<Vector3>& points,
        const std::vector<Vector3>& normal,
        const Vec4f& colour)
    {
        drawTriangleFan(points, normal, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawFrame(const Vector3& position, const Quaternion& orientation, const Vec3f& size, const Vec4f& colour)
    {
        drawFrame(position, orientation, size, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawSpheres(const std::vector<Vector3>& points, const std::vector<float>& radius, const Vec4f& colour)
    {
        drawSpheres(points, radius, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawSpheres(const std::vector<Vector3>& points, float radius, const Vec4f& colour)
    {
        drawSpheres(points, radius, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawFakeSpheres(const std::vector<Vector3>& points, const std::vector<float>& radius, const Vec4f& colour)
    {
        drawFakeSpheres(points, radius, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawFakeSpheres(const std::vector<Vector3>& points, float radius, const Vec4f& colour)
    {
        drawFakeSpheres(points, radius, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawCone(const Vector3& p1, const Vector3& p2, float radius1, float radius2, const Vec4f& colour, int subd = 16)
    {
        drawCone(p1, p2, radius1, radius2, RGBAColor(colour), subd);
    }

    DEPRECATE_VEC4F
    void drawCube(const float& radius, const Vec4f& colour, const int& subd = 16)
    {
        drawCube(radius, RGBAColor(colour), subd);
    }

    DEPRECATE_VEC4F
    void drawCylinder(const Vector3& p1, const Vector3& p2, float radius, const Vec4f& colour, int subd = 16)
    {
        drawCylinder(p1, p2, radius, RGBAColor(colour), subd);
    }

    DEPRECATE_VEC4F
    void drawCapsule(const Vector3& p1, const Vector3& p2, float radius, const Vec4f& colour, int subd = 16)
    {
        drawCapsule(p1, p2, radius, RGBAColor(colour), subd);
    }

    DEPRECATE_VEC4F
    void drawArrow(const Vector3& p1, const Vector3& p2, float radius, const Vec4f& colour, int subd = 16)
    {
        drawArrow(p1, p2, radius, RGBAColor(colour), subd);
    }

    DEPRECATE_VEC4F
    void drawArrow(const Vector3& p1, const Vector3& p2, float radius, float coneLength, const Vec4f& colour, int subd = 16)
    {
        drawArrow(p1, p2, radius, coneLength, RGBAColor(colour), subd);
    }

    DEPRECATE_VEC4F
    void drawArrow(const Vector3& p1, const Vector3& p2, float radius, float coneLength, float coneRadius, const Vec4f& color, int subd = 16)
    {
        drawArrow(p1, p2, radius, coneLength, coneRadius, RGBAColor(color), subd);
    }

    DEPRECATE_VEC4F
    void drawCross(const Vector3& p, float length, const Vec4f& colour)
    {
        drawCross(p, length, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawPlus(const float& radius, const Vec4f& colour, const int& subd = 16)
    {
        drawPlus(radius, RGBAColor(colour), subd);
    }

    DEPRECATE_VEC4F
    void drawPoint(const Vector3& p, const Vec4f& c)
    {
        drawPoint(p, RGBAColor(c));
    }

    DEPRECATE_VEC4F
    void drawPoint(const Vector3& p, const Vector3& n, const Vec4f& c)
    {
        drawPoint(p, n, RGBAColor(c));
    }

    DEPRECATE_VEC4F
    void drawTriangle(const Vector3& p1, const Vector3& p2, const Vector3& p3,
        const Vector3& normal,
        const Vec4f& c1, const Vec4f& c2, const Vec4f& c3)
    {
        drawTriangle(p1, p2, p3, normal, RGBAColor(c1), RGBAColor(c2), RGBAColor(c3));
    }

    DEPRECATE_VEC4F
    void drawTriangle(const Vector3& p1, const Vector3& p2, const Vector3& p3,
        const Vector3& normal1, const Vector3& normal2, const Vector3& normal3,
        const Vec4f& c1, const Vec4f& c2, const Vec4f& c3)
    {
        drawTriangle(p1, p2, p3, normal1, normal2, normal3, RGBAColor(c1), RGBAColor(c2), RGBAColor(c3));
    }

    DEPRECATE_VEC4F
    void drawQuad(const Vector3& p1, const Vector3& p2, const Vector3& p3, const Vector3& p4,
        const Vector3& normal, const Vec4f& c)
    {
        drawQuad(p1, p2, p3, p4, normal, RGBAColor(c));
    }

    DEPRECATE_VEC4F
    void drawQuad(const Vector3& p1, const Vector3& p2, const Vector3& p3, const Vector3& p4,
        const Vector3& normal1, const Vector3& normal2, const Vector3& normal3, const Vector3& normal4,
        const Vec4f& c1, const Vec4f& c2, const Vec4f& c3, const Vec4f& c4)
    {
        drawQuad(p1, p2, p3, p4, normal1, normal2, normal3, normal4, RGBAColor(c1), RGBAColor(c2), RGBAColor(c3), RGBAColor(c4));
    }

    DEPRECATE_VEC4F
    void drawQuads(const std::vector<Vector3>& points, const Vec4f& colour)
    {
        drawQuads(points, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawTetrahedron(const Vector3& p0, const Vector3& p1, const Vector3& p2, const Vector3& p3, const Vec4f& colour)
    {
        drawTetrahedron(p0, p1, p2, p3, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawTetrahedra(const std::vector<Vector3>& points, const Vec4f& colour)
    {
        drawTetrahedra(points, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawScaledTetrahedra(const std::vector<Vector3>& points, const Vec4f& colour, const float scale)
    {
        drawScaledTetrahedra(points, RGBAColor(colour), scale);
    }

    DEPRECATE_VEC4F
    void drawHexahedron(const Vector3& p0, const Vector3& p1, const Vector3& p2, const Vector3& p3,
        const Vector3& p4, const Vector3& p5, const Vector3& p6, const Vector3& p7, const Vec4f& colour)
    {
        drawHexahedron(p0, p1, p2, p3, p4, p5, p6, p7, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawHexahedra(const std::vector<Vector3>& points, const Vec4f& colour)
    {
        drawHexahedra(points, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void drawScaledHexahedra(const std::vector<Vector3>& points, const Vec4f& colour, const float scale)
    {
        drawScaledHexahedra(points, RGBAColor(colour), scale);
    }

    DEPRECATE_VEC4F
    void drawSphere(const Vector3& p, float radius, const Vec4f& colour)
    {
        drawSphere(p, radius, RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void draw3DText(const Vector3& p, float scale, const Vec4f& color, const char* text)
    {
        draw3DText(p, scale, RGBAColor(color), text);
    }

    DEPRECATE_VEC4F
    void draw3DText_Indices(const std::vector<Vector3>& positions, float scale, const Vec4f& color)
    {
        draw3DText_Indices(positions, scale, RGBAColor(color));
    }

    DEPRECATE_VEC4F
    void setMaterial(const Vec4f& colour)
    {
        setMaterial(RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void resetMaterial(const Vec4f& colour)
    {
        resetMaterial(RGBAColor(colour));
    }

    DEPRECATE_VEC4F
    void writeOverlayText(int x, int y, unsigned fontSize, const Vec4f& color, const char* text)
    {
        writeOverlayText(x, y, fontSize, RGBAColor(color), text);
    }

#undef DEPRECATE_VEC4F

};

} // namespace sofa::helper::visual
