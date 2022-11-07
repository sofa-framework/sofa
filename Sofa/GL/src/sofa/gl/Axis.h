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
#include <sofa/type/Vec.h>
#include <sofa/type/Quat.h>

#include <sofa/gl/gl.h>
#include <sofa/gl/glu.h>

#include <map>

#include <sofa/gl/config.h>

namespace sofa::gl
{

class SOFA_GL_API Axis
{
public:
    typedef sofa::type::Vec3 Vector3;
    typedef sofa::type::Vec4f   Vec4f;
    typedef sofa::type::Vec3d   Vec3d;
    typedef sofa::type::Quat<SReal> Quaternion;
    Axis(SReal len=1.0_sreal);
    Axis(const Vector3& len);
    Axis(const Vector3& center, const Quaternion &orient, const Vector3& length);
    Axis(const Vector3& center, const double orient[4][4], const Vector3& length);
    Axis(const double *mat, const Vector3& length);
    Axis(const Vector3& center, const Quaternion &orient, SReal length=1.0_sreal);
    Axis(const Vector3& center, const double orient[4][4], SReal length=1.0_sreal);
    Axis(const double *mat, SReal length=1.0_sreal);

    ~Axis();

    void update(const Vector3& center, const Quaternion& orient = Quaternion());
    void update(const Vector3& center, const double orient[4][4]);
    void update(const double *mat);

    void draw( const Vec4f& colorX=Vec4f(1,0,0,1), const Vec4f& colorY=Vec4f(0,1,0,1), const Vec4f& colorZ=Vec4f(0,0,1,1) );

    static void draw(const Vector3& center, const Quaternion& orient, const Vector3& length, const Vec4f& colorX=Vec4f(1,0,0,1), const Vec4f& colorY=Vec4f(0,1,0,1), const Vec4f& colorZ=Vec4f(0,0,1,1) );
    static void draw(const Vector3& center, const double orient[4][4], const Vector3& length, const Vec4f& colorX=Vec4f(1,0,0,1), const Vec4f& colorY=Vec4f(0,1,0,1), const Vec4f& colorZ=Vec4f(0,0,1,1) );
    static void draw(const double *mat, const Vector3& length, const Vec4f& colorX=Vec4f(1,0,0,1), const Vec4f& colorY=Vec4f(0,1,0,1), const Vec4f& colorZ=Vec4f(0,0,1,1) );
    static void draw(const Vector3& center, const Quaternion& orient, SReal length=1.0_sreal, const Vec4f& colorX=Vec4f(1,0,0,1), const Vec4f& colorY=Vec4f(0,1,0,1), const Vec4f& colorZ=Vec4f(0,0,1,1) );
    static void draw(const Vector3& center, const double orient[4][4], SReal length=1.0_sreal, const Vec4f& colorX=Vec4f(1,0,0,1), const Vec4f& colorY=Vec4f(0,1,0,1), const Vec4f& colorZ=Vec4f(0,0,1,1) );
    static void draw(const double *mat, SReal length=1.0_sreal, const Vec4f& colorX=Vec4f(1,0,0,1), const Vec4f& colorY=Vec4f(0,1,0,1), const Vec4f& colorZ=Vec4f(0,0,1,1) );

    //Draw a nice vector (cylinder + cone) given 2 points and a radius (used to draw the cylinder)
    static void draw(const Vector3& center, const Vector3& ext, const double& radius );
    //Draw a cylinder given two points and the radius of the extremities (to have a cone, simply set one radius to zero)
    static void draw(const Vector3& center, const Vector3& ext, const double& r1, const double& r2 );
private:

    Vector3 length;
    double matTransOpenGL[16];

    GLUquadricObj *quadratic;
    GLuint displayLists;

    void initDraw();

    static std::map < std::pair<std::pair<float,float>,float>, Axis* > axisMap;
    static Axis* get(const Vector3& len);
public:
    static void clear() { axisMap.clear(); } // need to be called when display list has been created in another opengl context

private:
    static const int quadricDiscretisation;
};

} // namespace sofa::gl
