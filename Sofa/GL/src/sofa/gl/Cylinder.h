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

#include <sofa/gl/config.h>

#include <map>

namespace sofa::gl
{

namespace
{
    using sofa::type::Vec3;
}

class SOFA_GL_API Cylinder
{
public:
    typedef sofa::type::Quat<SReal> Quaternion;

    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vector3, sofa::type::Vec3);

    Cylinder(SReal len=1.0_sreal);
    Cylinder(const Vec3& len);
    Cylinder(const Vec3& center, const Quaternion &orient, const Vec3& length);
    Cylinder(const Vec3& center, const double orient[4][4], const Vec3& length);
    Cylinder(const double *mat, const Vec3& length);
    Cylinder(const Vec3& center, const Quaternion &orient, SReal length=1.0_sreal);
    Cylinder(const Vec3& center, const double orient[4][4], SReal length=1.0_sreal);
    Cylinder(const double *mat, SReal length=1.0_sreal);

    ~Cylinder();

    void update(const Vec3& center, const Quaternion& orient = Quaternion());
    void update(const Vec3& center, const double orient[4][4]);
    void update(const double *mat);

    void draw();

    static void draw(const Vec3& center, const Quaternion& orient, const Vec3& length);
    static void draw(const Vec3& center, const double orient[4][4], const Vec3& length);
    static void draw(const double *mat, const Vec3& length);
    static void draw(const Vec3& center, const Quaternion& orient, SReal length=1.0_sreal);
    static void draw(const Vec3& center, const double orient[4][4], SReal length=1.0_sreal);
    static void draw(const double *mat, SReal length=1.0_sreal);

private:

    Vec3 length;
    double matTransOpenGL[16];

    GLUquadricObj *quadratic;
    GLuint displayList;

    void initDraw();

    static std::map < std::pair<std::pair<float,float>,float>, Cylinder* > CylinderMap;
    static Cylinder* get(const Vec3& len);
public:
    static void clear() { CylinderMap.clear(); } // need to be called when display list has been created in another opengl context

private:
    static const int quadricDiscretisation;
};

} // namespace sofa::gl
