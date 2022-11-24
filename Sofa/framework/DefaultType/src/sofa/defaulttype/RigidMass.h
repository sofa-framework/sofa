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
#include <cstdlib>
#include <cmath>


namespace sofa::defaulttype
{

template<typename real>
class RigidMass<3, real>
{
public:
    typedef real value_type;
    typedef real Real;
    typedef type::Mat<3,3,Real> Mat3x3;
    Real mass,volume;
    Mat3x3 inertiaMatrix;	      // Inertia matrix of the object
    Mat3x3 inertiaMassMatrix;    // Inertia matrix of the object * mass of the object
    Mat3x3 invInertiaMatrix;	  // inverse of inertiaMatrix
    Mat3x3 invInertiaMassMatrix; // inverse of inertiaMassMatrix
    RigidMass(Real m=1)
    {
        mass = m;
        volume = 1;
        inertiaMatrix.identity();
        recalc();
    }
    void operator=(Real m)
    {
        mass = m;
        recalc();
    }
    void operator+=(Real m)
    {
        mass += m;
        recalc();
    }
    void operator-=(Real m)
    {
        mass -= m;
        recalc();
    }
    // operator to cast to const Real
    constexpr operator const Real() const
    {
        return mass;
    }
    void recalc()
    {
        inertiaMassMatrix = inertiaMatrix * mass;
        const bool canInvert1 = invInertiaMatrix.invert(inertiaMatrix);
        const bool canInvert2 = invInertiaMassMatrix.invert(inertiaMassMatrix);
        assert(canInvert1);
        assert(canInvert2);
        SOFA_UNUSED(canInvert1);
        SOFA_UNUSED(canInvert2);
    }

    inline friend std::ostream& operator << (std::ostream& out, const RigidMass<3, real>& m )
    {
        out<<m.mass;
        out<<" "<<m.volume;
        out<<" "<<m.inertiaMatrix;
        return out;
    }
    inline friend std::istream& operator >> (std::istream& in, RigidMass<3, real>& m )
    {
        in>>m.mass;
        in>>m.volume;
        in>>m.inertiaMatrix;
        return in;
    }
    constexpr void operator *=(Real fact)
    {
        mass *= fact;
        inertiaMassMatrix *= fact;
        invInertiaMassMatrix /= fact;
    }
    constexpr void operator /=(Real fact)
    {
        mass /= fact;
        inertiaMassMatrix /= fact;
        invInertiaMassMatrix *= fact;
    }
};

template<typename real>
constexpr RigidDeriv<3,real> operator*(const RigidDeriv<3,real>& d, const RigidMass<3,real>& m)
{
    RigidDeriv<3,real> res;
    getVCenter(res) = getVCenter(d) * m.mass;
    getVOrientation(res) = m.inertiaMassMatrix * getVOrientation(d);
    return res;
}

template<typename real>
constexpr RigidDeriv<3,real> operator*(const RigidMass<3,real>& m, const RigidDeriv<3,real>& d)
{
    RigidDeriv<3,real> res;
    getVCenter(res) = getVCenter(d) * m.mass;
    getVOrientation(res) = m.inertiaMassMatrix * getVOrientation(d);
    return res;
}

template<typename real>
constexpr RigidDeriv<3,real> operator/(const RigidDeriv<3,real>& d, const RigidMass<3, real>& m)
{
    RigidDeriv<3,real> res;
    getVCenter(res) = getVCenter(d) / m.mass;
    getVOrientation(res) = m.invInertiaMassMatrix * getVOrientation(d);
    return res;
}

typedef RigidMass<3,double> Rigid3dMass;
typedef RigidMass<3,float> Rigid3fMass;
typedef RigidMass<3,SReal> Rigid3Mass;   ///< un-defined precision type

template<class real>
class RigidMass<2, real>
{
public:
    typedef real value_type;
    typedef real Real;
    Real mass,volume;
    Real inertiaMatrix;	      // Inertia matrix of the object
    Real inertiaMassMatrix;    // Inertia matrix of the object * mass of the object
    Real invInertiaMatrix;	  // inverse of inertiaMatrix
    Real invInertiaMassMatrix; // inverse of inertiaMassMatrix
    RigidMass(Real m=1)
    {
        mass = m;
        volume = 1;
        inertiaMatrix = 1;
        recalc();
    }
    void operator=(Real m)
    {
        mass = m;
        recalc();
    }
    void operator+=(Real m)
    {
        mass += m;
        recalc();
    }
    void operator-=(Real m)
    {
        mass -= m;
        recalc();
    }
    // operator to cast to const Real
    constexpr operator const Real() const
    {
        return mass;
    }
    /// Mass for a circle
    RigidMass(Real m, Real radius)
    {
        mass = m;
        volume = radius*radius*R_PI;
        inertiaMatrix = (radius*radius)/2;
        recalc();
    }
    /// Mass for a rectangle
    RigidMass(Real m, Real xwidth, Real ywidth)
    {
        mass = m;
        volume = xwidth*xwidth + ywidth*ywidth;
        inertiaMatrix = volume/12;
        recalc();
    }

    void recalc()
    {
        inertiaMassMatrix = inertiaMatrix * mass;
        if (inertiaMatrix == 0.)
        {
            throw std::runtime_error("Attempt to divide by zero");
        }
        invInertiaMatrix = 1. / inertiaMatrix;
        if (inertiaMassMatrix == 0.)
        {
            throw std::runtime_error("Attempt to divide by zero");
        }
        invInertiaMassMatrix = 1. / inertiaMassMatrix;
    }
    inline friend std::ostream& operator << (std::ostream& out, const RigidMass<2,Real>& m )
    {
        out<<m.mass;
        out<<" "<<m.volume;
        out<<" "<<m.inertiaMatrix;
        return out;
    }
    inline friend std::istream& operator >> (std::istream& in, RigidMass<2,Real>& m )
    {
        in>>m.mass;
        in>>m.volume;
        in>>m.inertiaMatrix;
        return in;
    }
    constexpr void operator *=(Real fact)
    {
        mass *= fact;
        inertiaMassMatrix *= fact;
        invInertiaMassMatrix /= fact;
    }

    constexpr void operator /=(Real fact)
    {
        mass /= fact;
        inertiaMassMatrix /= fact;
        invInertiaMassMatrix *= fact;
    }
};

template<typename real>
constexpr RigidDeriv<2,real> operator*(const RigidDeriv<2,real>& d, const RigidMass<2,real>& m)
{
    RigidDeriv<2,real> res;
    getVCenter(res) = getVCenter(d) * m.mass;
    getVOrientation(res) = m.inertiaMassMatrix * getVOrientation(d);
    return res;
}

template<typename real>
constexpr RigidDeriv<2,real> operator/(const RigidDeriv<2,real>& d, const RigidMass<2, real>& m)
{
    RigidDeriv<2,real> res;
    getVCenter(res) = getVCenter(d) / m.mass;
    getVOrientation(res) = m.invInertiaMassMatrix * getVOrientation(d);
    return res;
}

typedef RigidMass<2, double> Rigid2dMass;
typedef RigidMass<2, float> Rigid2fMass;
typedef RigidMass<2, SReal> Rigid2Mass;   ///< un-defined precision type

} // namespace sofa::defaulttype
