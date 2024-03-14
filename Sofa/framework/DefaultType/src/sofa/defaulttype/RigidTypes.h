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

#include <sofa/defaulttype/RigidCoord.h>
#include <sofa/defaulttype/RigidDeriv.h>
#include <sofa/defaulttype/RigidMass.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixConstraint.h>

#include <sofa/type/Vec.h>
#include <sofa/type/Quat.h>
#include <sofa/type/vector.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/random.h>
#include <cmath>


namespace sofa::defaulttype
{

//=============================================================================
// 3D Rigids
//=============================================================================

/** Degrees of freedom of 3D rigid bodies. Orientations are modeled using quaternions.
 */
template<typename real>
class StdRigidTypes<3, real>
{
public:
    typedef real Real;
    typedef RigidCoord<3, real> Coord;
    typedef RigidDeriv<3, real> Deriv;
    typedef typename Coord::Vec3 Vec3;
    typedef typename Coord::Quat Quat;
    typedef type::Vec<3, Real> AngularVector;

    static constexpr sofa::Size spatial_dimensions = Coord::spatial_dimensions;
    static constexpr sofa::Size coord_total_size = Coord::total_size;
    static constexpr sofa::Size deriv_total_size = Deriv::total_size;

    typedef typename Coord::Pos CPos;
    typedef typename Coord::Rot CRot;
    static constexpr const CPos& getCPos(const Coord& c) { return c.getCenter(); }
    static constexpr void setCPos(Coord& c, const CPos& v) { c.getCenter() = v; }
    static constexpr const CRot& getCRot(const Coord& c) { return c.getOrientation(); }
    static constexpr void setCRot(Coord& c, const CRot& v) { c.getOrientation() = v; }

    typedef type::Vec<3, real> DPos;
    typedef type::Vec<3, real> DRot;
    static constexpr const DPos& getDPos(const Deriv& d) { return getVCenter(d); }
    static constexpr void setDPos(Deriv& d, const DPos& v) { getVCenter(d) = v; }
    static constexpr const DRot& getDRot(const Deriv& d) { return getVOrientation(d); }
    static constexpr void setDRot(Deriv& d, const DRot& v) { getVOrientation(d) = v; }

    typedef linearalgebra::CompressedRowSparseMatrixConstraint<Deriv> MatrixDeriv;

    typedef type::vector<Coord> VecCoord;
    typedef type::vector<Deriv> VecDeriv;
    typedef type::vector<Real> VecReal;

    template<typename T>
    static constexpr void set(Coord& c, T x, T y, T z)
    {
        c.getCenter()[0] = static_cast<Real>(x);
        c.getCenter()[1] = static_cast<Real>(y);
        c.getCenter()[2] = static_cast<Real>(z);
    }

    template<typename T>
    static constexpr void get(T& x, T& y, T& z, const Coord& c)
    {
        x = static_cast<T>(c.getCenter()[0]);
        y = static_cast<T>(c.getCenter()[1]);
        z = static_cast<T>(c.getCenter()[2]);
    }

    // set linear and angular velocities
    template<typename T>
    static constexpr void set(Deriv& c, T x, T y, T z, T rx, T ry, T rz)
    {
        c.getLinear()[0] = static_cast<Real>(x);
        c.getLinear()[1] = static_cast<Real>(y);
        c.getLinear()[2] = static_cast<Real>(z);
        c.getAngular()[0] = static_cast<Real>(rx);
        c.getAngular()[1] = static_cast<Real>(ry);
        c.getAngular()[2] = static_cast<Real>(rz);
    }

    template<typename T>
    static constexpr void add(Coord& c, T x, T y, T z)
    {
        c.getCenter()[0] += static_cast<Real>(x);
        c.getCenter()[1] += static_cast<Real>(y);
        c.getCenter()[2] += static_cast<Real>(z);
    }

    template<typename T>
    static constexpr void set(Deriv& c, T x, T y, T z)
    {
        getVCenter(c)[0] = static_cast<Real>(x);
        getVCenter(c)[1] = static_cast<Real>(y);
        getVCenter(c)[2] = static_cast<Real>(z);
    }

    template<typename T>
    static constexpr void get(T& x, T& y, T& z, const Deriv& c)
    {
        x = static_cast<T>(getVCenter(c)[0]);
        y = static_cast<T>(getVCenter(c)[1]);
        z = static_cast<T>(getVCenter(c)[2]);
    }

    template<typename T>
    static constexpr void add(Deriv& c, T x, T y, T z)
    {
        getVCenter(c)[0] += static_cast<Real>(x);
        getVCenter(c)[1] += static_cast<Real>(y);
        getVCenter(c)[2] += static_cast<Real>(z);
    }

    static constexpr const char* Name();

    /// Return a Deriv with random value. Each entry with magnitude smaller than the given value.
    static Deriv randomDeriv(Real minMagnitude, Real maxMagnitude)
    {
        Deriv result;
        set(result, Real(helper::drand(minMagnitude, maxMagnitude)), Real(helper::drand(minMagnitude, maxMagnitude)), Real(helper::drand(minMagnitude, maxMagnitude)),
            Real(helper::drand(minMagnitude, maxMagnitude)), Real(helper::drand(minMagnitude, maxMagnitude)), Real(helper::drand(minMagnitude, maxMagnitude)));
        return result;
    }

    static Deriv coordDifference(const Coord& c1, const Coord& c2)
    {
        type::Vec3 vCenter = c1.getCenter() - c2.getCenter();
        type::Quat<SReal> quat, quat1(c1.getOrientation()), quat2(c2.getOrientation());
        // Transformation between c2 and c1 frames
        quat = quat1 * quat2.inverse();
        quat.normalize();
        type::Vec3 axis(type::NOINIT);
        type::Quat<SReal>::value_type angle{};
        quat.quatToAxis(axis, angle);
        axis *= angle;
        return Deriv(vCenter, axis);
    }

    static Coord interpolate(const type::vector< Coord >& ancestors, const type::vector< Real >& coefs)
    {
        assert(ancestors.size() == coefs.size());

        Coord c;

        for (sofa::Size i = 0; i < ancestors.size(); i++)
        {
            // Position interpolation.
            c.getCenter() += ancestors[i].getCenter() * coefs[i];

            // Angle extraction from the orientation quaternion.
            type::Quat<Real> q = ancestors[i].getOrientation();
            Real angle = acos(q[3]) * 2;

            // Axis extraction from the orientation quaternion.
            type::Vec<3, Real> v(q[0], q[1], q[2]);
            Real norm = v.norm();
            if (norm > 0.0005)
            {
                v.normalize();

                // The scale factor is applied to the angle
                angle *= coefs[i];

                // Corresponding quaternion is computed, then added to the interpolated point orientation.
                q.axisToQuat(v, angle);
                q.normalize();

                c.getOrientation() += q;
            }
        }

        c.getOrientation().normalize();

        return c;
    }

    static Deriv interpolate(const type::vector< Deriv >& ancestors, const type::vector< Real >& coefs)
    {
        assert(ancestors.size() == coefs.size());

        Deriv d;

        for (sofa::Size i = 0; i < ancestors.size(); i++)
        {
            d += ancestors[i] * coefs[i];
        }

        return d;
    }

    /// inverse rigid transform
    static constexpr Coord inverse(const Coord& c)
    {
        CRot qinv = c.getOrientation().inverse();
        return Coord(-(qinv.rotate(c.getCenter())), qinv);
    }

    /// matrix product
    static constexpr Coord mult(const Coord& a, const Coord& b)
    {
        return a.mult(b);
    }

    /// double cross product: a * ( b * c )
    static constexpr Vec3 crosscross(const Vec3& a, const Vec3& b, const Vec3& c)
    {
        return cross(a, cross(b, c));
    }

    /// create a rotation from Euler angles. For homogeneity with 2D.
    static Quat rotationEuler(Real x, Real y, Real z) { return Quat::fromEuler(x, y, z); }

};

typedef StdRigidTypes<3,double> Rigid3dTypes;
typedef StdRigidTypes<3,float> Rigid3fTypes;

/// We now use template aliases so we do not break backward compatibility.
template<> constexpr const char* Rigid3dTypes::Name() { return "Rigid3d"; }
template<> constexpr const char* Rigid3fTypes::Name() { return "Rigid3f"; }

typedef StdRigidTypes<3,SReal> Rigid3Types;  ///< un-defined precision type
typedef StdRigidTypes<3,SReal> RigidTypes;   ///< alias (beurk)

//=============================================================================
// 2D Rigids
//=============================================================================
/** Degrees of freedom of 2D rigid bodies.
*/
template<typename real>
class StdRigidTypes<2, real>
{
public:
    typedef real Real;
    typedef type::Vec<2,real> Vec2;

    typedef RigidDeriv<2,Real> Deriv;
    typedef RigidCoord<2,Real> Coord;
    typedef Real AngularVector;

    static constexpr sofa::Size spatial_dimensions = Coord::spatial_dimensions;
    static constexpr sofa::Size coord_total_size = Coord::total_size;
    static constexpr sofa::Size deriv_total_size = Deriv::total_size;

    typedef typename Coord::Pos CPos;
    typedef typename Coord::Rot CRot;
    static constexpr const CPos& getCPos(const Coord& c) { return c.getCenter(); }
    static constexpr void setCPos(Coord& c, const CPos& v) { c.getCenter() = v; }
    static constexpr const CRot& getCRot(const Coord& c) { return c.getOrientation(); }
    static constexpr void setCRot(Coord& c, const CRot& v) { c.getOrientation() = v; }

    typedef type::Vec<2,real> DPos;
    typedef real DRot;
    static constexpr const DPos& getDPos(const Deriv& d) { return getVCenter(d); }
    static constexpr void setDPos(Deriv& d, const DPos& v) { getVCenter(d) = v; }
    static constexpr const DRot& getDRot(const Deriv& d) { return getVOrientation(d); }
    static constexpr void setDRot(Deriv& d, const DRot& v) { getVOrientation(d) = v; }

    static constexpr const char* Name();

    typedef type::vector<Coord> VecCoord;
    typedef type::vector<Deriv> VecDeriv;
    typedef type::vector<Real> VecReal;

    typedef linearalgebra::CompressedRowSparseMatrixConstraint<Deriv> MatrixDeriv;

    template<typename T>
    static constexpr void set(Coord& c, T x, T y, T)
    {
        c.getCenter()[0] = static_cast<Real>(x);
        c.getCenter()[1] = static_cast<Real>(y);
    }

    template<typename T>
    static constexpr void get(T& x, T& y, T& z, const Coord& c)
    {
        x = static_cast<T>(c.getCenter()[0]);
        y = static_cast<T>(c.getCenter()[1]);
        z = static_cast<T>(0);
    }

    template<typename T>
    static constexpr void add(Coord& c, T x, T y, T)
    {
        c.getCenter()[0] += static_cast<Real>(x);
        c.getCenter()[1] += static_cast<Real>(y);
    }

    template<typename T>
    static constexpr void set(Deriv& c, T x, T y, T)
    {
        c.getVCenter()[0] = static_cast<Real>(x);
        c.getVCenter()[1] = static_cast<Real>(y);
    }

    template<typename T>
    static constexpr void get(T& x, T& y, T& z, const Deriv& c)
    {
        x = static_cast<T>(c.getVCenter()[0]);
        y = static_cast<T>(c.getVCenter()[1]);
        z = static_cast<T>(0);
    }

    // Set linear and angular velocities, in 6D for uniformity with 3D
    template<typename T>
    static constexpr void set(Deriv& c, T x, T y, T, T vrot, T, T )
    {
        c.getVCenter()[0] = static_cast<Real>(x);
        c.getVCenter()[1] = static_cast<Real>(y);
        c.getVOrientation() = static_cast<Real>(vrot);
    }

    template<typename T>
    static constexpr void add(Deriv& c, T x, T y, T)
    {
        c.getVCenter()[0] += static_cast<Real>(x);
        c.getVCenter()[1] += static_cast<Real>(y);
    }

    /// Return a Deriv with random value. Each entry with magnitude smaller than the given value.
    static Deriv randomDeriv( Real minMagnitude, Real maxMagnitude )
    {
        Deriv result;
        set( result, Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)),
                     Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)));
        return result;
    }

    static Coord interpolate(const type::vector< Coord > & ancestors, const type::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        Coord c;

        for (sofa::Size i = 0; i < ancestors.size(); i++)
        {
            c += ancestors[i] * coefs[i];
        }

        return c;
    }

    static Deriv interpolate(const type::vector< Deriv > & ancestors, const type::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        Deriv d;

        for (sofa::Size i = 0; i < ancestors.size(); i++)
        {
            d += ancestors[i] * coefs[i];
        }

        return d;
    }

    /// specialized version of the double cross product: a * ( b * c ) for the variation of torque applied to the frame due to a small rotation with constant force.
    static constexpr Real crosscross ( const Vec2& f, const Real& dtheta, const Vec2& OP)
    {
        return dtheta * dot( f,OP );
    }

    /// specialized version of the double cross product: a * ( b * c ) for point acceleration
    static constexpr Vec2 crosscross ( const Real& omega, const Real& dtheta, const Vec2& OP)
    {
        return OP * omega * (-dtheta);
    }

    /// create a rotation from Euler angles (only the first is used). For homogeneity with 3D.
    static constexpr CRot rotationEuler( Real x, Real , Real ){ return CRot(x); }


};

template<> constexpr const char* Rigid2dTypes::Name() { return "Rigid2d"; }
template<> constexpr const char* Rigid2fTypes::Name() { return "Rigid2f"; }

/// \endcond



/** @name Helpers
 *  Helper Functions to more easily create tests and check the results.
 */
//@{

/** Velocity of a rigid body at a given point, based on its angular velocity and its linear velocity at another point.
  \param omega angular velocity
  \param v known linear velocity
  \param pv point where the linear velocity is known
  \param p point where we compute the velocity
  */
template <class Vec3>
static constexpr Vec3 rigidVelocity( const Vec3& omega, const Vec3& v, const Vec3& pv, const Vec3& p ) { return v + cross( omega, p-pv ); }

/// Apply the given translation and rotation to each entry of vector v
template<class V1, class Vec, class Rot>
static constexpr void displace( V1& v, Vec translation, Rot rotation )
{
    for(sofa::Size i=0; i<v.size(); i++)
        v[i] = translation + rotation.rotate(v[i]);
}

/// Apply the given translation and rotation to each entry of vector v
template<class V1, class Rot>
static constexpr void rotate( V1& v, Rot rotation )
{
    for(sofa::Size i=0; i<v.size(); i++)
        v[i] = rotation.rotate(v[i]);
}

/// Apply a rigid transformation (translation, Euler angles) to the given points and their associated velocities.
template<class V1, class V2>
static void rigidTransform ( V1& points, V2& velocities, SReal tx, SReal ty, SReal tz, SReal rx, SReal ry, SReal rz )
{
    typedef type::Vec3 Vec3;
    typedef type::Quat<SReal> Quat;
    Vec3 translation(tx,ty,tz);
    Quat rotation = Quat::createQuaterFromEuler(Vec3(rx,ry,rz));
    displace(points,translation,rotation);
    rotate(velocities,rotation);
}
//@}

} // namespace sofa::defaulttype
