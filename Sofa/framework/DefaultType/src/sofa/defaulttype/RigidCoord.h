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
#include <sofa/defaulttype/config.h>

#include <sofa/defaulttype/RigidDeriv.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/type/Quat.h>
#include <sofa/type/vector.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/random.h>
#include <cmath>

#if !defined(NDEBUG)
#include <sofa/helper/logging/Messaging.h>
#endif

namespace sofa::defaulttype
{

template<typename real>
class RigidCoord<3,real>
{
public:
    typedef real value_type;
    typedef sofa::Size Size;
    typedef real Real;
    typedef type::Vec<3,Real> Pos;
    typedef type::Quat<Real> Rot;
    typedef type::Vec<3,Real> Vec3;
    typedef type::Quat<Real> Quat;
    typedef RigidDeriv<3,Real> Deriv;
    typedef type::Mat<4,4,Real> HomogeneousMat;
    typedef type::Vec<4,Real> HomogeneousVec;

protected:
    Vec3 center;
    Quat orientation;
public:
    constexpr RigidCoord(const Vec3& posCenter, const Quat& orient)
        : center(posCenter), orientation(orient)
    {
    }

    constexpr RigidCoord()
    {
        clear();
    }

    template<typename real2>
    constexpr RigidCoord(const RigidCoord<3, real2>& c)
        : center(c.getCenter()), orientation(c.getOrientation())
    {
    }


    constexpr void clear()
    {
        center.clear();
        orientation.clear();
    }

    /**
     * @brief Random rigid transform composed of 3 random translations and 3 random Euler angles
     * @param a Range of each random value: (-a,+a)
     * @return random rigid transform
     */
    static RigidCoord rand(SReal a)
    {
        RigidCoord t;
        t.center = Pos(SReal(helper::drand(a)), SReal(helper::drand(a)), SReal(helper::drand(a)));
        t.orientation = Quat::fromEuler(SReal(helper::drand(a)), SReal(helper::drand(a)), SReal(helper::drand(a)));
        return t;
    }

    template<typename real2>
    constexpr void operator=(const RigidCoord<3, real2>& c)
    {
        center = c.getCenter();
        orientation = c.getOrientation();
    }

    constexpr void operator =(const Vec3& p)
    {
        center = p;
    }

    void operator +=(const Deriv& dg) {
        // R3 x SO(3) exponential integration
        center += dg.getVCenter();

        const Vec3 omega = dg.getVOrientation() / 2;
        const real theta = omega.norm();

        constexpr const real epsilon = std::numeric_limits<real>::epsilon();

        if (theta < epsilon) {
            // fallback to gnomonic projection
            Quat exp(omega[0], omega[1], omega[2], 1);
            exp.normalize();
            orientation = exp * orientation;
        }
        else {
            // expontential
            const real sinc = std::sin(theta) / theta;
            const Quat exp(sinc * omega[0],
                sinc * omega[1],
                sinc * omega[2],
                std::cos(theta));
            orientation = exp * orientation;
        }

    }

    constexpr RigidCoord<3, real> operator+(const Deriv& dg) const {
        RigidCoord c = *this;
        c += dg;
        return c;
    }


    constexpr RigidCoord<3, real> operator-(const RigidCoord<3, real>& a) const
    {
        return RigidCoord<3, real>(this->center - a.getCenter(), a.orientation.inverse() * this->orientation);
    }

    constexpr RigidCoord<3, real> operator+(const RigidCoord<3, real>& a) const
    {
        return RigidCoord<3, real>(this->center + a.getCenter(), a.orientation * this->orientation);
    }

    constexpr RigidCoord<3, real> operator-() const
    {
        return RigidCoord<3, real>(-this->center, this->orientation.inverse());
    }

    constexpr void operator +=(const RigidCoord<3, real>& a)
    {
        center += a.getCenter();
        orientation *= a.getOrientation();
    }

    template<typename real2>
    constexpr void operator*=(real2 a)
    {
        center *= a;
    }

    template<typename real2>
    constexpr void operator/=(real2 a)
    {
        center /= a;
    }

    template<typename real2>
    constexpr RigidCoord<3, real> operator*(real2 a) const
    {
        RigidCoord r = *this;
        r *= a;
        return r;
    }

    template<typename real2>
    constexpr RigidCoord<3, real> operator/(real2 a) const
    {
        RigidCoord r = *this;
        r /= a;
        return r;
    }

    /// dot product, mostly used to compute residuals as sqrt(x*x)
    constexpr Real operator*(const RigidCoord<3, real>& a) const
    {
        return center[0] * a.center[0] + center[1] * a.center[1] + center[2] * a.center[2]
            + orientation[0] * a.orientation[0] + orientation[1] * a.orientation[1]
            + orientation[2] * a.orientation[2] + orientation[3] * a.orientation[3];
    }

    /** Squared norm. For the rotation we use the xyz components of the quaternion.
    Note that this is not equivalent to the angle, so a 2d rotation and the equivalent 3d rotation have different norms.
      */
    constexpr real norm2() const
    {
        return center * center
            + orientation[0] * orientation[0]
            + orientation[1] * orientation[1]
            + orientation[2] * orientation[2]; // xyzw quaternion has null x,y,z if rotation is null
    }

    /// Euclidean norm
    real norm() const
    {
        return helper::rsqrt(norm2());
    }


    constexpr Vec3& getCenter() { return center; }
    constexpr Quat& getOrientation() { return orientation; }
    constexpr const Vec3& getCenter() const { return center; }
    constexpr const Quat& getOrientation() const { return orientation; }

    static constexpr RigidCoord<3, real> identity()
    {
        RigidCoord c;
        return c;
    }

    constexpr Vec3 rotate(const Vec3& v) const
    {
        return orientation.rotate(v);
    }

    constexpr RigidCoord<3, real> rotate(const RigidCoord<3, real>& c) const
    {
        RigidCoord r;
        r.center = center + orientation.rotate(c.center);
        r.orientation = orientation * c.getOrientation();
        return r;
    }

    constexpr Vec3 inverseRotate(const Vec3& v) const
    {
        return orientation.inverseRotate(v);
    }

    constexpr Vec3 translate(const Vec3& v) const
    {
        return v + center;
    }

    constexpr RigidCoord<3, real> translate(const RigidCoord<3, real>& c) const
    {
        RigidCoord r;
        r.center = c.center + center;
        r.orientation = c.orientation();
        return r;
    }

    /// Apply a transformation with respect to itself
    constexpr void multRight(const RigidCoord<3, real>& c)
    {
        center += orientation.rotate(c.getCenter());
        orientation = orientation * c.getOrientation();
    }

    /// compute the product with another frame on the right
    constexpr Vec3 mult(const Vec3& v) const
    {
        return orientation.rotate(v) + center;
    }

    /// compute the product with another frame on the right
    constexpr RigidCoord<3, real> mult(const RigidCoord<3, real>& c) const
    {
        RigidCoord r;
        r.center = center + orientation.rotate(c.center);
        r.orientation = orientation * c.getOrientation();
        return r;
    }

    /// Set from the given matrix
    template<class Mat>
    void fromMatrix(const Mat& m)
    {
        center[0] = m[0][3];
        center[1] = m[1][3];
        center[2] = m[2][3];
        sofa::type::Mat<3, 3, Real> rot; rot = m;
        orientation.fromMatrix(rot);
    }

    /// Write to the given 3x3 matrix
    void toMatrix( sofa::type::Mat<3,3,real>& m) const
    {
        m.identity();
        orientation.toMatrix(m);
    }

    /// Write to the given 4x4 matrix
    void toMatrix( sofa::type::Mat<4,4,real>& m) const
    {
        toHomogeneousMatrix(m);
    }

    constexpr void toHomogeneousMatrix(HomogeneousMat& m) const
    {
        m.identity();
        orientation.toHomogeneousMatrix(m);
        m[0][3] = center[0];
        m[1][3] = center[1];
        m[2][3] = center[2];
    }

    /// create a homogeneous vector from a 3d vector
    template <typename V>
    static constexpr HomogeneousVec toHomogeneous(V v, real r = 1.) {
        return HomogeneousVec(v[0], v[1], v[2], r);
    }
    /// create a 3d vector from a homogeneous vector
    template <typename V>
    static constexpr Vec3 fromHomogeneous(V v) {
        return Vec3(v[0], v[1], v[2]);
    }


    template<class Mat>
    void writeRotationMatrix(Mat& m) const
    {
        orientation.toMatrix(m);
    }

    /// Write the OpenGL transformation matrix
    constexpr void writeOpenGlMatrix(float m[16]) const
    {
        orientation.writeOpenGlMatrix(m);
        m[12] = (float)center[0];
        m[13] = (float)center[1];
        m[14] = (float)center[2];
    }

    /// Apply the transform to a point, i.e. project a point from the child frame to the parent frame (translation and rotation)
    constexpr Vec3 projectPoint(const Vec3& v) const
    {
        return orientation.rotate(v) + center;
    }

    /// Apply the transform to a vector, i.e. project a vector from the child frame to the parent frame (rotation only, no translation added)
    constexpr Vec3 projectVector(const Vec3& v) const
    {
        return orientation.rotate(v);
    }

    /// Apply the inverse transform to a point, i.e. project a point from the parent frame to the child frame (translation and rotation)
    constexpr Vec3 unprojectPoint(const Vec3& v) const
    {
        return orientation.inverseRotate(v - center);
    }

    ///  Apply the inverse transform to a vector, i.e. project a vector from the parent frame to the child frame (rotation only, no translation)
    constexpr Vec3 unprojectVector(const Vec3& v) const
    {
        return orientation.inverseRotate(v);
    }

    /// obsolete. Use projectPoint.
    constexpr Vec3 pointToParent(const Vec3& v) const { return projectPoint(v); }
    /// obsolete. Use unprojectPoint.
    constexpr Vec3 pointToChild(const Vec3& v) const { return unprojectPoint(v); }


    /// write to an output stream
    inline friend std::ostream& operator << (std::ostream& out, const RigidCoord<3, real>& v)
    {
        out << v.center << " " << v.orientation;
        return out;
    }
    /// read from an input stream
    inline friend std::istream& operator >> (std::istream& in, RigidCoord<3, real>& v)
    {
        in >> v.center >> v.orientation;
#if !defined(NDEBUG)
        if (!v.orientation.isNormalized())
        {
            std::stringstream text;
            text << "Rigid Object with invalid quaternion (non-unitary norm)! Normalising quaternion value... " << msgendl;
            text << "Previous value was: " << v.orientation << msgendl;
            v.orientation.normalize();
            text << "New value is: " << v.orientation;
            msg_warning("Rigid") << text.str();
}
#endif // NDEBUG
        return in;
}
    static constexpr Size max_size()
    {
        return 3;
    }

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    static constexpr sofa::Size total_size = 7;
    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
    static constexpr sofa::Size spatial_dimensions = 3;

    constexpr real* ptr() { return center.ptr(); }
    constexpr const real* ptr() const { return center.ptr(); }

    static constexpr Size size() { return 7; }

    /// Access to i-th element.
    constexpr real& operator[](Size i)
    {
        if (i < 3)
            return this->center(i);
        else
            return this->orientation[i - 3];
    }

    /// Const access to i-th element.
    constexpr const real& operator[](Size i) const
    {
        if (i < 3)
            return this->center(i);
        else
            return this->orientation[i - 3];
    }

    /// @name Tests operators
    /// @{

    constexpr bool operator==(const RigidCoord<3, real>& b) const
    {
        return center == b.center && orientation == b.orientation;
    }

    constexpr bool operator!=(const RigidCoord<3, real>& b) const
    {
        return center != b.center || orientation != b.orientation;
    }

    /// @}

};

template<typename real>
class RigidCoord<2,real>
{
public:
    typedef real value_type;
    typedef sofa::Size Size;
    typedef real Real;
    typedef type::Vec<2,Real> Pos;
    typedef Real Rot;
    typedef type::Vec<2,Real> Vec2;
    typedef type::Mat<3,3,Real> HomogeneousMat;
    typedef type::Vec<3,Real> HomogeneousVec;
private:
    Vec2 center;
    Real orientation;
public:
    constexpr RigidCoord(const Vec2& posCenter, const Real& orient)
        : center(posCenter), orientation(orient)
    {}

    constexpr RigidCoord()
    {
        clear();
    }

    constexpr void clear()
    {
        center.clear();
        orientation = 0;
    }

    /**
     * @brief Random rigid transform composed of 2 random translations and a random angle
     * @param a Range of each random value: (-a,+a)
     * @return random rigid transform
     */
    static RigidCoord rand(SReal a)
    {
        RigidCoord t;
        t.center = Pos(SReal(helper::drand(a)), SReal(helper::drand(a)));
        t.orientation = SReal(helper::drand(a));
        return t;
    }

    constexpr void operator +=(const RigidDeriv<2, real>& a)
    {
        center += getVCenter(a);
        orientation += getVOrientation(a);
    }

    constexpr RigidCoord<2, real> operator + (const RigidDeriv<2, real>& a) const
    {
        RigidCoord<2, real> c = *this;
        c.center += getVCenter(a);
        c.orientation += getVOrientation(a);
        return c;
    }

    constexpr RigidCoord<2, real> operator -(const RigidCoord<2, real>& a) const
    {
        return RigidCoord<2, real>(this->center - a.getCenter(), this->orientation - a.orientation);
    }

    constexpr RigidCoord<2, real> operator +(const RigidCoord<2, real>& a) const
    {
        return RigidCoord<2, real>(this->center + a.getCenter(), this->orientation + a.orientation);
    }


    constexpr RigidCoord<2, real> operator-() const
    {
        return RigidCoord<2, real>(-this->center, this->orientation.inverse());
    }


    constexpr void operator +=(const RigidCoord<2, real>& a)
    {
        center += a.getCenter();
        orientation += a.getOrientation();
    }

    template<typename real2>
    constexpr void operator*=(real2 a)
    {
        center *= a;
        orientation *= (Real)a;
    }

    template<typename real2>
    constexpr void operator/=(real2 a)
    {
        center /= a;
        orientation /= (Real)a;
    }

    template<typename real2>
    constexpr RigidCoord<2, real> operator*(real2 a) const
    {
        RigidCoord<2, real> r = *this;
        r *= a;
        return r;
    }

    template<typename real2>
    constexpr RigidCoord<2, real> operator/(real2 a) const
    {
        RigidCoord<2, real> r = *this;
        r /= a;
        return r;
    }

    /// dot product, mostly used to compute residuals as sqrt(x*x)
    constexpr Real operator*(const RigidCoord<2, real>& a) const
    {
        return center[0] * a.center[0] + center[1] * a.center[1]
            + orientation * a.orientation;
    }

    /// Squared norm
    Real norm2() const
    {
        Real angle = fmod((Real)orientation, (Real)M_PI * 2); // fmod is constexpr in c++23
        return center * center + angle * angle;
    }

    /// Euclidean norm
    real norm() const
    {
        return helper::rsqrt(norm2());
    }

    constexpr Vec2& getCenter() { return center; }
    constexpr  Real& getOrientation() { return orientation; }
    constexpr const Vec2& getCenter() const { return center; }
    constexpr const Real& getOrientation() const { return orientation; }

    Vec2 rotate(const Vec2& v) const
    {
        Real s = sin(orientation);
        Real c = cos(orientation);
        return Vec2(c * v[0] - s * v[1],
            s * v[0] + c * v[1]);
    }

    Vec2 inverseRotate(const Vec2& v) const
    {
        Real s = sin(-orientation);
        Real c = cos(-orientation);
        return Vec2(c * v[0] - s * v[1],
            s * v[0] + c * v[1]);
    }

    constexpr Vec2 translate(const Vec2& v) const
    {
        return v + center;
    }

    constexpr RigidCoord<2, real> translate(const RigidCoord<2, real>& c) const
    {
        RigidCoord r;
        r.center = c.center + center;
        r.orientation = c.orientation();
        return r;
    }

    static constexpr RigidCoord<2, real> identity()
    {
        RigidCoord<2, real> c;
        return c;
    }

    /// Apply a transformation with respect to itself
    constexpr void multRight(const RigidCoord<2, real>& c)
    {
        center += /*orientation.*/rotate(c.getCenter());
        orientation = orientation + c.getOrientation();
    }

    /// compute the product with another frame on the right
    constexpr Vec2 mult(const Vec2& v) const
    {
        return /*orientation.*/rotate(v) + center;
    }

    /// compute the product with another frame on the right
    constexpr RigidCoord<2, real> mult(const RigidCoord<2, real>& c) const
    {
        RigidCoord<2, real> r;
        r.center = center + /*orientation.*/rotate(c.center);
        r.orientation = orientation + c.getOrientation();
        return r;
    }

    template<class Mat>
    void writeRotationMatrix(Mat& m) const
    {
        m[0][0] = (typename Mat::Real)cos(orientation); m[0][1] = (typename Mat::Real) - sin(orientation);
        m[1][0] = (typename Mat::Real)sin(orientation); m[1][1] = (typename Mat::Real) cos(orientation);
    }

    /// Set from the given matrix
    template<class Mat>
    void fromMatrix(const Mat& m)
    {
        center[0] = m[0][2];
        center[1] = m[1][2];
        orientation = atan2(m[1][0], m[0][0]);
    }

    /// Write to the given matrix
    template<class Mat>
    void toMatrix(Mat& m) const
    {
        m.identity();
        writeRotationMatrix(m);
        m[0][2] = center[0];
        m[1][2] = center[1];
    }


    /// create a homogeneous vector from a 2d vector
    template <typename V>
    static constexpr HomogeneousVec toHomogeneous(V v, real r = 1.) {
        return HomogeneousVec(v[0], v[1], r);
    }
    /// create a 2d vector from a homogeneous vector
    template <typename V>
    static constexpr Vec2 fromHomogeneous(V v) {
        return Vec2(v[0], v[1]);
    }


    /// Write the OpenGL transformation matrix
    constexpr void writeOpenGlMatrix(float m[16]) const
    {
        //orientation.writeOpenGlMatrix(m);
        m[0] = cos(orientation);
        m[1] = sin(orientation);
        m[2] = 0;
        m[3] = 0;
        m[4] = -sin(orientation);
        m[5] = cos(orientation);
        m[6] = 0;
        m[7] = 0;
        m[8] = 0;
        m[9] = 0;
        m[10] = 1;
        m[11] = 0;
        m[12] = (float)center[0];
        m[13] = (float)center[1];
        m[14] = (float)center[2];
        m[15] = 1;
    }

    /// compute the projection of a vector from the parent frame to the child
    constexpr Vec2 vectorToChild(const Vec2& v) const
    {
        return /*orientation.*/inverseRotate(v);
    }

    /// Apply the transform to a point, i.e. project a point from the child frame to the parent frame (translation and rotation)
    constexpr Vec2 projectPoint(const Vec2& v) const
    {
        return rotate(v) + center;
    }

    /// Apply the transform to a vector, i.e. project a vector from the child frame to the parent frame (rotation only, no translation added)
    constexpr Vec2 projectVector(const Vec2& v) const
    {
        return rotate(v);
    }

    /// Apply the inverse transform to a point, i.e. project a point from the parent frame to the child frame (translation and rotation)
    constexpr Vec2 unprojectPoint(const Vec2& v) const
    {
        return inverseRotate(v - center);
    }

    ///  Apply the inverse transform to a vector, i.e. project a vector from the parent frame to the child frame (rotation only, no translation)
    constexpr Vec2 unprojectVector(const Vec2& v) const
    {
        return inverseRotate(v);
    }

    /// obsolete. Use projectPoint.
    constexpr Vec2 pointToParent(const Vec2& v) const { return projectPoint(v); }
    /// obsolete. Use unprojectPoint.
    constexpr Vec2 pointToChild(const Vec2& v) const { return unprojectPoint(v); }


    /// write to an output stream
    inline friend std::ostream& operator << (std::ostream& out, const RigidCoord<2, real>& v)
    {
        out << v.center << " " << v.orientation;
        return out;
    }
    /// read from an input stream
    inline friend std::istream& operator >> (std::istream& in, RigidCoord<2, real>& v)
    {
        in >> v.center >> v.orientation;
        return in;
    }

    static constexpr Size max_size()
    {
        return 3;
    }

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    static constexpr sofa::Size total_size = 3;

    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
    static constexpr sofa::Size spatial_dimensions = 2;

    constexpr real* ptr() { return center.ptr(); }
    constexpr const real* ptr() const { return center.ptr(); }

    static constexpr Size size() { return 3; }

    /// Access to i-th element.
    constexpr real& operator[](Size i)
    {
        if (i < 2)
            return this->center(i);
        else
            return this->orientation;
    }

    /// Const access to i-th element.
    constexpr const real& operator[](Size i) const
    {
        if (i < 2)
            return this->center(i);
        else
            return this->orientation;
    }

    /// @name Tests operators
    /// @{

    constexpr bool operator==(const RigidCoord<2, real>& b) const
    {
        return center == b.center && orientation == b.orientation;
    }

    constexpr bool operator!=(const RigidCoord<2, real>& b) const
    {
        return center != b.center || orientation != b.orientation;
    }

    /// @}
};

} // namespace sofa::defaulttype
