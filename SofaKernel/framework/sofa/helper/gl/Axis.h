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
#ifndef SOFA_HELPER_GL_AXIS_H
#define SOFA_HELPER_GL_AXIS_H

#ifndef SOFA_NO_OPENGL

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Quat.h>

#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glu.h>

#include <map>

#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

namespace gl
{

class SOFA_HELPER_API Axis
{
public:
    typedef sofa::defaulttype::Vector3 Vector3;
    typedef sofa::defaulttype::Vec4f   Vec4f;
    typedef sofa::defaulttype::Vec3d   Vec3d;
    typedef sofa::defaulttype::Quaternion Quaternion;
    Axis(SReal len=(SReal)1);
    Axis(const Vector3& len);
    Axis(const Vector3& center, const Quaternion &orient, const Vector3& length);
    Axis(const Vector3& center, const double orient[4][4], const Vector3& length);
    Axis(const double *mat, const Vector3& length);
    Axis(const Vector3& center, const Quaternion &orient, SReal length=(SReal)1);
    Axis(const Vector3& center, const double orient[4][4], SReal length=(SReal)1);
    Axis(const double *mat, SReal length=(SReal)1.0);

    ~Axis();

    void update(const Vector3& center, const Quaternion& orient = Quaternion());
    void update(const Vector3& center, const double orient[4][4]);
    void update(const double *mat);

    void draw( const Vec4f& colorX=Vec4f(1,0,0,1), const Vec4f& colorY=Vec4f(0,1,0,1), const Vec4f& colorZ=Vec4f(0,0,1,1) );

    static void draw(const Vector3& center, const Quaternion& orient, const Vector3& length, const Vec4f& colorX=Vec4f(1,0,0,1), const Vec4f& colorY=Vec4f(0,1,0,1), const Vec4f& colorZ=Vec4f(0,0,1,1) );
    static void draw(const Vector3& center, const double orient[4][4], const Vector3& length, const Vec4f& colorX=Vec4f(1,0,0,1), const Vec4f& colorY=Vec4f(0,1,0,1), const Vec4f& colorZ=Vec4f(0,0,1,1) );
    static void draw(const double *mat, const Vector3& length, const Vec4f& colorX=Vec4f(1,0,0,1), const Vec4f& colorY=Vec4f(0,1,0,1), const Vec4f& colorZ=Vec4f(0,0,1,1) );
    static void draw(const Vector3& center, const Quaternion& orient, SReal length=(SReal)1, const Vec4f& colorX=Vec4f(1,0,0,1), const Vec4f& colorY=Vec4f(0,1,0,1), const Vec4f& colorZ=Vec4f(0,0,1,1) );
    static void draw(const Vector3& center, const double orient[4][4], SReal length=(SReal)1, const Vec4f& colorX=Vec4f(1,0,0,1), const Vec4f& colorY=Vec4f(0,1,0,1), const Vec4f& colorZ=Vec4f(0,0,1,1) );
    static void draw(const double *mat, SReal length=(SReal)1.0, const Vec4f& colorX=Vec4f(1,0,0,1), const Vec4f& colorY=Vec4f(0,1,0,1), const Vec4f& colorZ=Vec4f(0,0,1,1) );

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

};

} // namespace gl

} // namespace helper

} // namespace sofa

#endif /* SOFA_NO_OPENGL */

#endif
