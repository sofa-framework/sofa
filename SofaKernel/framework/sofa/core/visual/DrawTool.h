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
#ifndef SOFA_HELPER_GL_DRAWTOOL_H
#define SOFA_HELPER_GL_DRAWTOOL_H

#include <sofa/core/core.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/types/RGBAColor.h>

#include <vector>

namespace sofa
{
namespace core
{
namespace visual
{

/**
 *  \brief Utility class to perform debug drawing.
 *
 *  Class which contains a set of methods to perform minimal debug drawing regardless of the graphics API used.
 *  Components receive a pointer to the DrawTool through the VisualParams parameter of their draw method.
 *  Sofa provides a default concrete implementation of this class for the OpenGL API with the DrawToolGL class.
 *
 */

class SOFA_CORE_API DrawTool
{

public:
    typedef sofa::helper::types::RGBAColor RGBAColor;
    typedef sofa::defaulttype::Vec4f   Vec4f;
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
    virtual void drawPoints(const std::vector<Vector3> &points, float size,  const  Vec4f& colour) = 0 ;
    virtual void drawPoints(const std::vector<Vector3> &points, float size, const std::vector<Vec4f>& colour) = 0;

    virtual void drawLine(const Vector3 &p1, const Vector3 &p2, const Vec4f& colour) =  0;
    virtual void drawLines(const std::vector<Vector3> &points, float size, const Vec4f& colour) = 0 ;
    virtual void drawLines(const std::vector<Vector3> &points, float size, const std::vector<Vec4f>& colours) = 0 ;
    virtual void drawLines(const std::vector<Vector3> &points, const std::vector< Vec2i > &index , float size, const Vec4f& colour) = 0 ;

    virtual void drawLineStrip(const std::vector<Vector3> &points, float size, const Vec4f& colour) = 0 ;
    virtual void drawLineLoop(const std::vector<Vector3> &points, float size, const Vec4f& colour) = 0 ;

    virtual void drawTriangles(const std::vector<Vector3> &points, const Vec4f& colour) = 0 ;
    virtual void drawTriangles(const std::vector<Vector3> &points, const Vector3& normal, const Vec4f& colour) = 0 ;
    virtual void drawTriangles(const std::vector<Vector3> &points,
            const std::vector< Vec3i > &index,
            const std::vector<Vector3>  &normal,
            const Vec4f& colour) = 0 ;
    virtual void drawTriangles(const std::vector<Vector3> &points,
            const std::vector<Vector3>  &normal,
            const std::vector< Vec4f > &colour) = 0 ;

    virtual void drawTriangleStrip(const std::vector<Vector3> &points,
            const std::vector<Vector3>  &normal,
            const Vec4f& colour) = 0 ;

    virtual void drawTriangleFan(const std::vector<Vector3> &points,
            const std::vector<Vector3>  &normal,
            const Vec4f& colour) = 0 ;



    virtual void drawFrame   (const Vector3& position, const Quaternion &orientation, const Vec3f &size) = 0 ;
    virtual void drawFrame   (const Vector3& position, const Quaternion &orientation, const Vec3f &size, const Vec4f &colour) = 0 ;

    virtual void drawSpheres (const std::vector<Vector3> &points, const std::vector<float>& radius, const Vec4f& colour) = 0;
    virtual void drawSpheres (const std::vector<Vector3> &points, float radius, const Vec4f& colour) = 0 ;
    virtual void drawFakeSpheres(const std::vector<Vector3> &points, const std::vector<float>& radius, const Vec4f& colour) = 0;
    virtual void drawFakeSpheres(const std::vector<Vector3> &points, float radius, const Vec4f& colour) = 0;

    virtual void drawCone    (const Vector3& p1, const Vector3 &p2, float radius1, float radius2, const Vec4f& colour, int subd=16) = 0 ;

    /// Draw a cube of size one centered on the current point.
    virtual void drawCube    (const float& radius, const Vec4f& colour, const int& subd=16) = 0 ;

    virtual void drawCylinder(const Vector3& p1, const Vector3 &p2, float radius, const Vec4f& colour,  int subd=16) = 0 ;

    virtual void drawCapsule(const Vector3& p1, const Vector3 &p2, float radius, const Vec4f& colour,  int subd=16) = 0 ;

    virtual void drawArrow   (const Vector3& p1, const Vector3 &p2, float radius, const Vec4f& colour,  int subd=16) = 0 ;
    virtual void drawArrow   (const Vector3& p1, const Vector3 &p2, float radius, float coneLength, const Vec4f& colour,  int subd=16) = 0 ;

    /// Draw a cross (3 lines) centered on p
    virtual void drawCross(const Vector3&p, float length, const Vec4f& colour) = 0;

    /// Draw a plus sign of size one centered on the current point.
    virtual void drawPlus    (const float& radius, const Vec4f& colour, const int& subd=16) = 0 ;

    virtual void drawPoint(const Vector3 &p, const Vec4f &c) = 0 ;
    virtual void drawPoint(const Vector3 &p, const Vector3 &n, const Vec4f &c) = 0 ;

    virtual void drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
            const Vector3 &normal) = 0 ;
    virtual void drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
            const Vector3 &normal, const Vec4f &c) = 0 ;
    virtual void drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
            const Vector3 &normal,
            const Vec4f &c1, const Vec4f &c2, const Vec4f &c3) = 0 ;
    virtual void drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
            const Vector3 &normal1, const Vector3 &normal2, const Vector3 &normal3,
            const Vec4f &c1, const Vec4f &c2, const Vec4f &c3) = 0 ;

    virtual void drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
            const Vector3 &normal) = 0 ;
    virtual void drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
            const Vector3 &normal, const Vec4f &c) = 0 ;
    virtual void drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
            const Vector3 &normal,
            const Vec4f &c1, const Vec4f &c2, const Vec4f &c3, const Vec4f &c4) = 0 ;
    virtual void drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
            const Vector3 &normal1, const Vector3 &normal2, const Vector3 &normal3, const Vector3 &normal4,
            const Vec4f &c1, const Vec4f &c2, const Vec4f &c3, const Vec4f &c4) = 0 ;
    virtual void drawQuads(const std::vector<Vector3> &points, const Vec4f& colour) = 0 ;

    virtual void drawTetrahedron(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2, const Vector3 &p3, const Vec4f &colour) = 0 ;
    virtual void drawTetrahedra(const std::vector<Vector3> &points, const Vec4f& colour) = 0;
    //Scale each tetrahedron
    virtual void drawScaledTetrahedra(const std::vector<Vector3> &points, const Vec4f& colour, const float scale) = 0;

    virtual void drawHexahedron(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2, const Vector3 &p3,
        const Vector3 &p4, const Vector3 &p5, const Vector3 &p6, const Vector3 &p7, const Vec4f &colour) = 0;
    virtual void drawHexahedra(const std::vector<Vector3> &points, const Vec4f& colour) = 0;
    //Scale each hexahedron
    virtual void drawScaledHexahedra(const std::vector<Vector3> &points, const Vec4f& colour, const float scale) = 0;

    virtual void drawSphere( const Vector3 &p, float radius) = 0 ;
    virtual void drawEllipsoid(const Vector3 &p, const Vector3 &radii) = 0;

    virtual void drawBoundingBox( const Vector3 &min, const Vector3 &max ) = 0;

    virtual void draw3DText(const Vector3 &p, float scale, const Vec4f &color, const char* text) = 0;
    virtual void draw3DText_Indices(const helper::vector<Vector3> &positions, float scale, const Vec4f &color) = 0;
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
    virtual void setMaterial(const Vec4f &colour) = 0 ;

    virtual void resetMaterial(const Vec4f &colour) = 0 ;
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
    virtual void writeOverlayText( int x, int y, unsigned fontSize, const Vec4f &color, const char* text ) = 0;

    // @name Color Buffer method
    virtual void readPixels(int x, int y, int w, int h, float* rgb, float* z = NULL) = 0;
    /// @}

    virtual void clear() {}


};

} // namespace visual

} // namespace core

} // namespace sofa

#endif //SOFA_CORE_VISUAL_DRAWTOOL_H
