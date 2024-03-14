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
#include <sofa/type/RGBAColor.h>

#include <sofa/gl/gl.h>
#include <sofa/gl/glu.h>

#include <map>
#include <memory>

#include <sofa/gl/config.h>

namespace sofa::gl
{

class SOFA_GL_API Axis
{
public:
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vector3, sofa::type::Vec3);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vec4f, sofa::type::Vec4f);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vec3d, sofa::type::Vec3d);

    typedef sofa::type::Quat<SReal> Quaternion;
    Axis(SReal len=1.0_sreal);
    Axis(const type::Vec3& len);
    Axis(const type::Vec3& center, const Quaternion &orient, const type::Vec3& length);
    Axis(const type::Vec3& center, const double orient[4][4], const type::Vec3& length);
    Axis(const double *mat, const type::Vec3& length);
    Axis(const type::Vec3& center, const Quaternion &orient, SReal length=1.0_sreal);
    Axis(const type::Vec3& center, const double orient[4][4], SReal length=1.0_sreal);
    Axis(const double *mat, SReal length=1.0_sreal);

    ~Axis();

    void update(const type::Vec3& center, const Quaternion& orient = Quaternion());
    void update(const type::Vec3& center, const double orient[4][4]);
    void update(const double *mat);

    void draw(const type::RGBAColor& colorX = type::RGBAColor::red(), const type::RGBAColor& colorY = type::RGBAColor::green(), const type::RGBAColor& colorZ = type::RGBAColor::red());
    static void draw(const type::Vec3& center, const Quaternion& orient, const type::Vec3& length, const type::RGBAColor& colorX = type::RGBAColor::red(), const type::RGBAColor& colorY = type::RGBAColor::green(), const type::RGBAColor& colorZ = type::RGBAColor::red());
    static void draw(const type::Vec3& center, const double orient[4][4], const type::Vec3& length, const type::RGBAColor& colorX = type::RGBAColor::red(), const type::RGBAColor& colorY = type::RGBAColor::green(), const type::RGBAColor& colorZ = type::RGBAColor::red());
    static void draw(const double* mat, const type::Vec3& length, const type::RGBAColor& colorX = type::RGBAColor::red(), const type::RGBAColor& colorY = type::RGBAColor::green(), const type::RGBAColor& colorZ = type::RGBAColor::red());
    static void draw(const type::Vec3& center, const Quaternion& orient, SReal length = 1.0_sreal, const type::RGBAColor& colorX = type::RGBAColor::red(), const type::RGBAColor& colorY = type::RGBAColor::green(), const type::RGBAColor& colorZ = type::RGBAColor::red());
    static void draw(const type::Vec3& center, const double orient[4][4], SReal length = 1.0_sreal, const type::RGBAColor& colorX = type::RGBAColor::red(), const type::RGBAColor& colorY = type::RGBAColor::green(), const type::RGBAColor& colorZ = type::RGBAColor::red());
    static void draw(const double* mat, SReal length = 1.0_sreal, const type::RGBAColor& colorX = type::RGBAColor::red(), const type::RGBAColor& colorY = type::RGBAColor::green(), const type::RGBAColor& colorZ = type::RGBAColor::red());

    SOFA_ATTRIBUTE_DEPRECATED__RGBACOLOR_AS_FIXEDARRAY()
    void draw(const type::Vec4f& colorX = type::Vec4f(1, 0, 0, 1), const type::Vec4f& colorY = type::Vec4f(0, 1, 0, 1), const type::Vec4f& colorZ = type::Vec4f(0, 0, 1, 1));
    SOFA_ATTRIBUTE_DEPRECATED__RGBACOLOR_AS_FIXEDARRAY()
    static void draw(const type::Vec3& center, const Quaternion& orient, const type::Vec3& length, const type::Vec4f& colorX = type::Vec4f(1, 0, 0, 1), const type::Vec4f& colorY = type::Vec4f(0, 1, 0, 1), const type::Vec4f& colorZ = type::Vec4f(0, 0, 1, 1));
    SOFA_ATTRIBUTE_DEPRECATED__RGBACOLOR_AS_FIXEDARRAY()
    static void draw(const type::Vec3& center, const double orient[4][4], const type::Vec3& length, const type::Vec4f& colorX = type::Vec4f(1, 0, 0, 1), const type::Vec4f& colorY = type::Vec4f(0, 1, 0, 1), const type::Vec4f& colorZ = type::Vec4f(0, 0, 1, 1));
    SOFA_ATTRIBUTE_DEPRECATED__RGBACOLOR_AS_FIXEDARRAY()
    void draw(const double* mat, const type::Vec3& length, const type::Vec4f& colorX = type::Vec4f(1, 0, 0, 1), const type::Vec4f& colorY = type::Vec4f(0, 1, 0, 1), const type::Vec4f& colorZ = type::Vec4f(0, 0, 1, 1));
    SOFA_ATTRIBUTE_DEPRECATED__RGBACOLOR_AS_FIXEDARRAY()
    static void draw(const type::Vec3& center, const Quaternion& orient, SReal length = 1.0_sreal, const type::Vec4f& colorX = type::Vec4f(1, 0, 0, 1), const type::Vec4f& colorY = type::Vec4f(0, 1, 0, 1), const type::Vec4f& colorZ = type::Vec4f(0, 0, 1, 1));
    SOFA_ATTRIBUTE_DEPRECATED__RGBACOLOR_AS_FIXEDARRAY()
    static void draw(const type::Vec3& center, const double orient[4][4], SReal length = 1.0_sreal, const type::Vec4f& colorX = type::Vec4f(1, 0, 0, 1), const type::Vec4f& colorY = type::Vec4f(0, 1, 0, 1), const type::Vec4f& colorZ = type::Vec4f(0, 0, 1, 1));
    SOFA_ATTRIBUTE_DEPRECATED__RGBACOLOR_AS_FIXEDARRAY()
    static void draw(const double* mat, SReal length = 1.0_sreal, const type::Vec4f& colorX = type::Vec4f(1, 0, 0, 1), const type::Vec4f& colorY = type::Vec4f(0, 1, 0, 1), const type::Vec4f& colorZ = type::Vec4f(0, 0, 1, 1));


    //Draw a nice vector (cylinder + cone) given 2 points and a radius (used to draw the cylinder)
    static void draw(const type::Vec3& center, const type::Vec3& ext, const double& radius );
    //Draw a cylinder given two points and the radius of the extremities (to have a cone, simply set one radius to zero)
    static void draw(const type::Vec3& center, const type::Vec3& ext, const double& r1, const double& r2 );
private:

    type::Vec3 length;
    double matTransOpenGL[16];

    GLUquadricObj *quadratic;
    GLuint displayLists;

    void initDraw();

    using AxisSPtr = std::shared_ptr<Axis>;

    static std::map < type::Vec3f, AxisSPtr > axisMap;
    static AxisSPtr get(const type::Vec3& len);
public:
    static void clear() { axisMap.clear(); } // need to be called when display list has been created in another opengl context

private:
    static const int quadricDiscretisation;
};

} // namespace sofa::gl
