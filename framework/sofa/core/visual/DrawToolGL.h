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


#ifndef SOFA_CORE_VISUAL_DRAWTOOLGL_H
#define SOFA_CORE_VISUAL_DRAWTOOLGL_H

#include <sofa/core/visual/DrawTool.h>
#include <sofa/defaulttype/Vec.h>


namespace sofa
{

namespace core
{

namespace visual
{

class SOFA_CORE_API DrawToolGL : public DrawTool
{

public:
    typedef sofa::defaulttype::Vec4f   Vec4f;
    typedef sofa::defaulttype::Vec3f   Vec3f;
    typedef sofa::defaulttype::Vector3 Vector3;
    typedef sofa::defaulttype::Vec<3,int> Vec3i;
    typedef sofa::defaulttype::Vec<2,int> Vec2i;
    typedef sofa::defaulttype::Quaternion Quaternion;

    DrawToolGL();
    ~DrawToolGL();

    virtual void drawPoints(const std::vector<Vector3> &points, float size,  const Vec4f colour);
    virtual void drawPoints(const std::vector<Vector3> &points, float size, const std::vector<Vec4f> colour);

    virtual void drawLines(const std::vector<Vector3> &points, float size, const Vec4f colour);
    virtual void drawLines(const std::vector<Vector3> &points, const std::vector< Vec2i > &index, float size, const Vec4f colour);

    virtual void drawTriangles(const std::vector<Vector3> &points, const Vec4f colour);
    virtual void drawTriangles(const std::vector<Vector3> &points, const Vector3 normal, const Vec4f colour);
    virtual void drawTriangles(const std::vector<Vector3> &points,
            const std::vector< Vec3i > &index,
            const std::vector<Vector3>  &normal,
            const Vec4f colour);
    virtual void drawTriangles(const std::vector<Vector3> &points,
            const std::vector<Vector3>  &normal,
            const std::vector< Vec4f > &colour);

    virtual void drawTriangleStrip(const std::vector<Vector3> &points,
            const std::vector<Vector3>  &normal,
            const Vec4f colour);

    virtual void drawTriangleFan(const std::vector<Vector3> &points,
            const std::vector<Vector3>  &normal,
            const Vec4f colour);

    virtual void drawFrame(const Vector3& position, const Quaternion &orientation, const Vec3f &size);
    virtual void drawFrame(const Vector3& position, const Quaternion &orientation, const Vec3f &size, const Vec4f &colour);

    virtual void drawSpheres (const std::vector<Vector3> &points, const std::vector<float>& radius, const Vec4f colour);
    virtual void drawSpheres (const std::vector<Vector3> &points, float radius, const Vec4f colour);

    virtual void drawCone    (const Vector3& p1, const Vector3 &p2, float radius1, float radius2, const Vec4f colour, int subd=16);

    virtual void drawCube    (const float& radius, const Vec4f& colour, const int& subd=16);

    virtual void drawCylinder(const Vector3& p1, const Vector3 &p2, float radius, const Vec4f colour,  int subd=16);

    virtual void drawCapsule(const Vector3& p1, const Vector3 &p2, float radius, const Vec4f colour,  int subd=16);

    virtual void drawArrow   (const Vector3& p1, const Vector3 &p2, float radius, const Vec4f colour,  int subd=16);

    virtual void drawPlus    (const float& radius, const Vec4f& colour, const int& subd=16);

    virtual void drawPoint(const Vector3 &p, const Vec4f &c);
    virtual void drawPoint(const Vector3 &p, const Vector3 &n, const Vec4f &c);

    virtual void drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
            const Vector3 &normal);
    virtual void drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
            const Vector3 &normal, const Vec4f &c);
    virtual void drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
            const Vector3 &normal,
            const Vec4f &c1, const Vec4f &c2, const Vec4f &c3);
    virtual void drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
            const Vector3 &normal1, const Vector3 &normal2, const Vector3 &normal3,
            const Vec4f &c1, const Vec4f &c2, const Vec4f &c3);

    virtual void drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
            const Vector3 &normal);
    virtual void drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
            const Vector3 &normal, const Vec4f &c);
    virtual void drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
            const Vector3 &normal,
            const Vec4f &c1, const Vec4f &c2, const Vec4f &c3, const Vec4f &c4);
    virtual void drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
            const Vector3 &normal1, const Vector3 &normal2, const Vector3 &normal3, const Vector3 &normal4,
            const Vec4f &c1, const Vec4f &c2, const Vec4f &c3, const Vec4f &c4);

    virtual void drawSphere( const Vector3 &p, float radius);

    virtual void drawBoundingBox( const Vector3 &min, const Vector3 &max );

    virtual void clear();

    virtual void setMaterial(const Vec4f &colour, std::string name=std::string());

    virtual void resetMaterial(const Vec4f &colour, std::string name=std::string());

    virtual void pushMatrix();
    virtual void popMatrix();
    virtual void multMatrix(float* glTransform );
    virtual void scale( float s );

    virtual void writeOverlayText( int x, int y, unsigned fontSize, const Vec4f &color, const char* text );


protected:

    bool mLightEnabled;
    int  mPolygonMode;      //0: no cull, 1 front (CULL_CLOCKWISE), 2 back (CULL_ANTICLOCKWISE)
    bool mWireFrameEnabled;

public:
    // getter & setter
    virtual void setLightingEnabled(bool _isAnabled);

    bool getLightEnabled() {return mLightEnabled;}

    virtual void setPolygonMode(int _mode, bool _wireframe);

    int getPolygonMode() {return mPolygonMode;}
    bool getWireFrameEnabled() {return mWireFrameEnabled;}

};

}//namespace visual

}//namespace core

}//namespace sofa

#endif // SOFA_CORE_VISUAL_DRAWTOOLGL_H
