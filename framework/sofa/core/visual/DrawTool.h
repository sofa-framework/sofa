/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_HELPER_GL_DRAWTOOL_H
#define SOFA_HELPER_GL_DRAWTOOL_H

#include <sofa/SofaFramework.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Quat.h>

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
    typedef sofa::defaulttype::Vec4f   Vec4f;
    typedef sofa::defaulttype::Vec3f   Vec3f;
    typedef sofa::defaulttype::Vector3 Vector3;
    typedef sofa::defaulttype::Vec<3,int> Vec3i;
    typedef sofa::defaulttype::Vec<2,int> Vec2i;
    typedef sofa::defaulttype::Quaternion Quaternion;

    DrawTool() { clear(); }
    virtual ~DrawTool() {}

    /// @name Primitive rendering methods
    /// @{
    virtual void drawPoints(const std::vector<Vector3> &points, float size,  const Vec4f colour) = 0 ;
    virtual void drawPoints(const std::vector<Vector3> &points, float size, const std::vector<Vec4f> colour) = 0;
    virtual void drawLines(const std::vector<Vector3> &points, float size, const Vec4f colour) = 0 ;
    virtual void drawLines(const std::vector<Vector3> &points, const std::vector< Vec2i > &index , float size, const Vec4f colour) = 0 ;

    virtual void drawTriangles(const std::vector<Vector3> &points, const Vec4f colour) = 0 ;
    virtual void drawTriangles(const std::vector<Vector3> &points, const Vector3 normal, const Vec4f colour) = 0 ;
    virtual void drawTriangles(const std::vector<Vector3> &points,
            const std::vector< Vec3i > &index,
            const std::vector<Vector3>  &normal,
            const Vec4f colour) = 0 ;
    virtual void drawTriangles(const std::vector<Vector3> &points,
            const std::vector<Vector3>  &normal,
            const std::vector< Vec4f > &colour) = 0 ;

    virtual void drawTriangleStrip(const std::vector<Vector3> &points,
            const std::vector<Vector3>  &normal,
            const Vec4f colour) = 0 ;

    virtual void drawTriangleFan(const std::vector<Vector3> &points,
            const std::vector<Vector3>  &normal,
            const Vec4f colour) = 0 ;

    virtual void drawFrame   (const Vector3& position, const Quaternion &orientation, const Vec3f &size) = 0 ;
    virtual void drawFrame   (const Vector3& position, const Quaternion &orientation, const Vec3f &size, const Vec4f &colour) = 0 ;

    virtual void drawSpheres (const std::vector<Vector3> &points, const std::vector<float>& radius, const Vec4f colour) = 0;
    virtual void drawSpheres (const std::vector<Vector3> &points, float radius, const Vec4f colour) = 0 ;

    virtual void drawCone    (const Vector3& p1, const Vector3 &p2, float radius1, float radius2, const Vec4f colour, int subd=16) = 0 ;

    /// Draw a cube of size one centered on the current point.
    virtual void drawCube    (const float& radius, const Vec4f& colour, const int& subd=16) = 0 ;

    virtual void drawCylinder(const Vector3& p1, const Vector3 &p2, float radius, const Vec4f colour,  int subd=16) = 0 ;

    virtual void drawCapsule(const Vector3& p1, const Vector3 &p2, float radius, const Vec4f colour,  int subd=16) = 0 ;

    virtual void drawArrow   (const Vector3& p1, const Vector3 &p2, float radius, const Vec4f colour,  int subd=16) = 0 ;

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

    virtual void drawSphere( const Vector3 &p, float radius) = 0 ;

    virtual void drawBoundingBox( const Vector3 &min, const Vector3 &max ) = 0;
    /// @}

    /// @name Transformation methods.
    /// @{
    virtual void pushMatrix() = 0;

    virtual void popMatrix() =  0;

    virtual void multMatrix(float*  ) = 0;

    virtual void scale(float ) = 0;
    /// @}

    /// @name Drawing style methods.
    virtual void setMaterial(const Vec4f &colour, std::string name=std::string()) = 0 ;

    virtual void resetMaterial(const Vec4f &colour, std::string name=std::string()) = 0 ;

    virtual void setPolygonMode(int _mode, bool _wireframe) = 0 ;

    virtual void setLightingEnabled(bool _isAnabled) = 0 ;
    /// @}


    /// @name Overlay methods

    /// draw 2D text at position (x,y) from top-left corner
    virtual void writeOverlayText( int x, int y, unsigned fontSize, const Vec4f &color, const char* text ) = 0;

    /// @}

    virtual void clear() {};


};

} // namespace visual

} // namespace core

} // namespace sofa

#endif //SOFA_CORE_VISUAL_DRAWTOOL_H
