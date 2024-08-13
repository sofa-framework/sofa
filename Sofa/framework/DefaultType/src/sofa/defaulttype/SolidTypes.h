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
#ifndef SOFA_DEFAULTTYPE_SOLIDTYPES_H
#define SOFA_DEFAULTTYPE_SOLIDTYPES_H

#include <sofa/defaulttype/config.h>

#include <sofa/type/Vec.h>
#include <sofa/type/Quat.h>
#include <sofa/type/Mat.h>
#include <sofa/type/fixed_array.h>
#include <sofa/type/vector.h>
#include <iostream>
#include <map>
#include <sofa/type/Transform.h>



namespace sofa::defaulttype
{

/**
Base types for the ArticulatedSolid: position, orientation, velocity, angular velocity, etc.

@author François Faure, INRIA-UJF, 2006
*/
template< class R=float >
class SOFA_DEFAULTTYPE_API SolidTypes
{
public:
    typedef R Real;
    typedef type::Vec<3,Real> Vec3;
    typedef Vec3 Vec;  ///< For compatibility
    typedef type::Quat<Real> Rot;
    typedef type::Mat<3,3,Real> Mat3x3;
    typedef Mat3x3 Mat; ///< For compatibility
    typedef type::Mat<6,6,Real> Mat6x6;
    typedef Mat6x6 Mat66; ///< For compatibility
    typedef type::Vec<6,Real> Vec6;
    typedef Vec6 DOF; ///< For compatibility

    using SpatialVector = sofa::type::SpatialVector<R>;
    using Transform = sofa::type::Transform<R>;

    /**
     * \brief A twist aka a SpatialVector representing a velocity
     * This is pratically a SpatialVector (screw) with the additionnal semantics
     * that this screw represents a twist (velocity) and not a wrench (force and torque)
     * @author Anthony Truchet, CEA, 2006
     */
    class Twist : public SpatialVector
    {
    public:
        Twist(const Vec3& linear, const Vec3& angular)
            : SpatialVector(angular, linear) {}
    };

    /**
        * \brief A wrench aka a SpatialVector representing a force and a torque
     * This is pratically a SpatialVector (screw) with the additionnal semantics
     * that this screw represents a wrench (force and torque) and not a twist (velocity)
     * @author Anthony Truchet, CEA, 2006
     */
    class Wrench : public SpatialVector
    {
    public:
        Wrench(const Vec3& force, const Vec3& torque)
            : SpatialVector(force, torque) {}
    };


    class SOFA_DEFAULTTYPE_API RigidInertia
    {
    public:
        Real m;  ///< mass
        Vec h;   ///< position of the mass center in the local reference frame
        Mat I;  /// Inertia matrix around the mass center
        RigidInertia();
        RigidInertia( Real m, const Vec& h, const Mat& I );
        SpatialVector operator * (const SpatialVector& v ) const;
        RigidInertia operator * ( const Transform& t ) const;
        inline friend std::ostream& operator << (std::ostream& out, const RigidInertia& r )
        {
            out<<"I= "<<r.I<<std::endl;
            out<<"h= "<<r.h<<std::endl;
            out<<"m= "<<r.m<<std::endl;
            return out;
        }
    };

    class SOFA_DEFAULTTYPE_API ArticulatedInertia
    {
    public:
        Mat M;
        Mat H;
        Mat I;
        ArticulatedInertia();
        ArticulatedInertia( const Mat& M, const Mat& H, const Mat& I );
        SpatialVector operator * (const SpatialVector& v ) const;
        ArticulatedInertia operator * ( Real r ) const;
        ArticulatedInertia& operator = (const RigidInertia& Ri );
        ArticulatedInertia& operator += (const ArticulatedInertia& Ai );
        ArticulatedInertia operator + (const ArticulatedInertia& Ai ) const;
        ArticulatedInertia operator - (const ArticulatedInertia& Ai ) const;
        inline friend std::ostream& operator << (std::ostream& out, const ArticulatedInertia& r )
        {
            out<<"I= "<<r.I<<std::endl;
            out<<"H= "<<r.H<<std::endl;
            out<<"M= "<<r.M<<std::endl;
            return out;
        }
        /// Convert to a full 6x6 matrix
        void copyTo( Mat66& ) const;
    };

    typedef Transform Coord;
    typedef SpatialVector Deriv;
    typedef Coord VecCoord;
    typedef Deriv VecDeriv;
    typedef Real  VecReal;

    static Mat dyad( const Vec& u, const Vec& v );

    static Vec mult( const Mat& m, const Vec& v );

    static Vec multTrans( const Mat& m, const Vec& v );

    /// Cross product matrix of a vector
    static Mat crossM( const Vec& v );

    static ArticulatedInertia dyad ( const SpatialVector& u, const SpatialVector& v );

    static constexpr const char* Name()
    {
        return "Solid";
    }
};

#if !defined(SOFA_DEFAULTTYPE_SOLIDTYPES_CPP)
extern template class SOFA_DEFAULTTYPE_API SolidTypes<double>;
extern template class SOFA_DEFAULTTYPE_API SolidTypes<float>;
#endif

}

#endif


