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
#ifndef SOFA_HELPER_GL_AXIS_H
#define SOFA_HELPER_GL_AXIS_H

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

using namespace sofa::defaulttype;

class SOFA_HELPER_API Axis
{
public:

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

    void draw();

    static void draw(const Vector3& center, const Quaternion& orient, const Vector3& length);
    static void draw(const Vector3& center, const double orient[4][4], const Vector3& length);
    static void draw(const double *mat, const Vector3& length);
    static void draw(const Vector3& center, const Quaternion& orient, SReal length=(SReal)1);
    static void draw(const Vector3& center, const double orient[4][4], SReal length=(SReal)1);
    static void draw(const double *mat, SReal length=(SReal)1.0);

    //Draw a nice vector (cylinder + cone) given 2 points and a radius (used to draw the cylinder)
    static void draw(const Vector3& center, const Vector3& ext, const double& radius);
    //Draw a cylinder given two points and the radius of the extremities (to have a cone, simply set one radius to zero)
    static void draw(const Vector3& center, const Vector3& ext, const double& r1, const double& r2);
private:

    Vector3 length;
    double matTransOpenGL[16];

    GLUquadricObj *quadratic;
    GLuint displayList;

    void initDraw();

    static std::map < std::pair<std::pair<float,float>,float>, Axis* > axisMap;
    static Axis* get(const Vector3& len);

};

} // namespace gl

} // namespace helper

} // namespace sofa

#endif
