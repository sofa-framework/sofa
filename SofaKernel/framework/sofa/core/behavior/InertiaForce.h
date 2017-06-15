/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_INTERTIAFORCE_H
#define SOFA_CORE_INTERTIAFORCE_H

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
#include <sofa/core/objectmodel/BaseContext.h>

namespace sofa
{
namespace core
{
namespace behavior
{

/** Return the inertia force applied to a body referenced in a moving coordinate system.
 *  \param sv spatial velocity (omega, vorigin) of the coordinate system
 *  \param a acceleration of the origin of the coordinate system
 *  \param m mass of the body
 *  \param x position of the body in the moving coordinate system
 *  \param v velocity of the body in the moving coordinate system
 *         This default implementation returns no inertia.
 */
template<class Coord, class Deriv, class Vec, class M, class SV>
Deriv inertiaForce( const SV& sv, const Vec& a, const M& m, const Coord& x, const Deriv& v )
{
    SOFA_UNUSED(sv);
    SOFA_UNUSED(a);
    SOFA_UNUSED(m);
    SOFA_UNUSED(x);
    SOFA_UNUSED(v);
    return Deriv();
    //const Deriv& omega=sv.getAngularVelocity();
    //return -( a + omega.cross( omega.cross(x) + v*2 ))*m;
}

/// Specialization of the inertia force for 3D particles
template <>
inline defaulttype::Vec<3, double> inertiaForce<
defaulttype::Vec<3, double>,
            defaulttype::Vec<3, double>,
            objectmodel::BaseContext::Vec3,
            double,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& sv,
                    const objectmodel::BaseContext::Vec3& a,
                    const double& m,
                    const defaulttype::Vec<3, double>& x,
                    const defaulttype::Vec<3, double>& v
            )
{
    const objectmodel::BaseContext::Vec3& omega=sv.getAngularVelocity();
    //std::cerr<<"inertiaForce, sv = "<<sv<<", omega ="<<omega<<", a = "<<a<<", m= "<<m<<", x= "<<x<<", v= "<<v<<std::endl;
    return -( a + omega.cross( omega.cross(x) + v*2 ))*m;
}

/// Specialization of the inertia force for 3D particles
template <>
inline defaulttype::Vec<3, float> inertiaForce<
defaulttype::Vec<3, float>,
            defaulttype::Vec<3, float>,
            objectmodel::BaseContext::Vec3,
            float,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& sv,
                    const objectmodel::BaseContext::Vec3& a,
                    const float& m,
                    const defaulttype::Vec<3, float>& x,
                    const defaulttype::Vec<3, float>& v
            )
{
    const objectmodel::BaseContext::Vec3& omega=sv.getAngularVelocity();
    //std::cerr<<"inertiaForce, sv = "<<sv<<", omega ="<<omega<<", a = "<<a<<", m= "<<m<<", x= "<<x<<", v= "<<v<<std::endl;
    return -( a + omega.cross( omega.cross(x) + v*2 ))*m;
}

/// Specialization of the inertia force for 2D particles
template <>
inline defaulttype::Vec<2, double> inertiaForce<
defaulttype::Vec<2, double>,
            defaulttype::Vec<2, double>,
            objectmodel::BaseContext::Vec3,
            double,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& sv,
                    const objectmodel::BaseContext::Vec3& a,
                    const double& m,
                    const defaulttype::Vec<2, double>& x,
                    const defaulttype::Vec<2, double>& v
            )
{
    double omega=(double)sv.getAngularVelocity()[2]; // z direction
    defaulttype::Vec<2, double> a2( (double)a[0], (double)a[1] );
    return -( a2 -( x*omega + v*2 )*omega )*m;
}

/// Specialization of the inertia force for 2D particles
template <>
inline defaulttype::Vec<2, float> inertiaForce<
defaulttype::Vec<2, float>,
            defaulttype::Vec<2, float>,
            objectmodel::BaseContext::Vec3,
            float,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& sv,
                    const objectmodel::BaseContext::Vec3& a,
                    const float& m,
                    const defaulttype::Vec<2, float>& x,
                    const defaulttype::Vec<2, float>& v
            )
{
    float omega=(float)sv.getAngularVelocity()[2]; // z direction
    defaulttype::Vec<2, float> a2( (float)a[0], (float)a[1] );
    return -( a2 -( x*omega + v*2 )*omega )*m;
}



/// Specialization of the inertia force for defaulttype::Rigid3dTypes
template <>
inline defaulttype::RigidDeriv<3, double> inertiaForce<
defaulttype::RigidCoord<3, double>,
            defaulttype::RigidDeriv<3, double>,
            objectmodel::BaseContext::Vec3,
            defaulttype::RigidMass<3, double>,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::RigidMass<3, double>& mass,
                    const defaulttype::RigidCoord<3, double>& x,
                    const defaulttype::RigidDeriv<3, double>& v
            )
{
    defaulttype::RigidDeriv<3, double>::Vec3 omega( vframe.lineVec[0], vframe.lineVec[1], vframe.lineVec[2] );
    defaulttype::RigidDeriv<3, double>::Vec3 origin = x.getCenter(), finertia, zero(0,0,0);

    finertia = -( aframe + omega.cross( omega.cross(origin) + v.getVCenter()*2 ))*mass.mass;
    return defaulttype::RigidDeriv<3, double>( finertia, zero );
    /// \todo replace zero by Jomega.cross(omega)
}

/// Specialization of the inertia force for defaulttype::Rigid3fTypes
template <>
inline defaulttype::RigidDeriv<3, float> inertiaForce<
defaulttype::RigidCoord<3, float>,
            defaulttype::RigidDeriv<3, float>,
            objectmodel::BaseContext::Vec3,
            defaulttype::RigidMass<3, float>,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::RigidMass<3, float>& mass,
                    const defaulttype::RigidCoord<3, float>& x,
                    const defaulttype::RigidDeriv<3, float>& v
            )
{
    defaulttype::RigidDeriv<3, float>::Vec3 omega( (float)vframe.lineVec[0], (float)vframe.lineVec[1], (float)vframe.lineVec[2] );
    defaulttype::RigidDeriv<3, float>::Vec3 origin = x.getCenter(), finertia, zero(0,0,0);

    finertia = -( aframe + omega.cross( omega.cross(origin) + v.getVCenter()*2 ))*mass.mass;
    return defaulttype::RigidDeriv<3, float>( finertia, zero );
    /// \todo replace zero by Jomega.cross(omega)
}


/// Specialization of the inertia force for defaulttype::LaparoscopicRigid3Types
template <>
inline defaulttype::LaparoscopicRigid3Types::Deriv inertiaForce<
defaulttype::LaparoscopicRigid3Types::Coord,
            defaulttype::LaparoscopicRigid3Types::Deriv,
            defaulttype::Vector3,
            defaulttype::Rigid3Mass,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const defaulttype::Vector3& aframe,
                    const defaulttype::Rigid3Mass& mass,
                    const defaulttype::LaparoscopicRigid3Types::Coord& x,
                    const defaulttype::LaparoscopicRigid3Types::Deriv& v )
{
    defaulttype::Vector3 omega( vframe.lineVec[0], vframe.lineVec[1], vframe.lineVec[2] );
    defaulttype::Vector3 origin, finertia, zero(0,0,0);
    origin[0] = x.getTranslation();

    finertia = -( aframe + omega.cross( omega.cross(origin) + defaulttype::Vector3(v.getVTranslation()*2,0,0) ))*mass.mass;
    return defaulttype::LaparoscopicRigid3Types::Deriv( finertia[0], zero );
    /// \todo replace zero by Jomega.cross(omega)
}

}
}
}

#endif // SOFA_CORE_INTERTIAFORCE_H
