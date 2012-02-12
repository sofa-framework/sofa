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
#ifndef SOFA_HELPER_UnitTest_H
#define SOFA_HELPER_UnitTest_H

#include <iostream>
#include <sstream>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/Quater.h>

namespace sofa
{

namespace helper
{

/** Base class for testing functions.
  To implement a new test, derive this class and implement the succeeds() method.
  Messages can be issued using the serr stream. They will be displayed or not, along with the name of the test, depending on the value of the verbose static member.
  */
struct UnitTest
{
    static bool verbose;  ///< Condition for printing test name, comments and results of the test.
    std::string name;     ///< Test name. Can be a long string explaining what the test checks.
    std::ostringstream  msg;  ///< Output stream used to issue messages during the tests. Displayed or discarded, depending on the value of variable verbose.

    /// The test name can be a long string explaining what the test checks.
    UnitTest( std::string testName );

    /// Runs the test and return true in case of failure. Optionally print begin and end messages, depending on the verbose variable
    bool fails();

    /// Perform the test and return true in case of success.
    virtual bool succeeds()=0;


    /** @name Helpers
     *  Helper Functions to more easily create tests and check the results.
     */
    //@{
    /// A very small value. Can be used to check if an error is small enough.
    virtual double epsilon() const { return 1.0e-10; }

    /** Velocity of a rigid body at a given point, based on its angular velocity and its linear velocity at another point.
      \param omega angular velocity
      \param v known linear velocity
      \param pv point where the linear velocity is known
      \param p point where we compute the velocity
      */
    template <class Vec3>
    static Vec3 rigidVelocity( const Vec3& omega, const Vec3& v, const Vec3& pv, const Vec3& p ) { return v + cross( omega, p-pv ); }

    /// Apply the given translation and rotation to each entry of vector v
    template<class V1, class Vec, class Rot>
    static void displace( V1& v, Vec translation, Rot rotation )
    {
        for(std::size_t i=0; i<v.size(); i++)
            v[i] = translation + rotation.rotate(v[i]);
    }

    /// Apply the given translation and rotation to each entry of vector v
    template<class V1, class Rot>
    static void rotate( V1& v, Rot rotation )
    {
        for(std::size_t i=0; i<v.size(); i++)
            v[i] = rotation.rotate(v[i]);
    }

    /// Apply a rigid transformation (translation, Euler angles) to the given points and their associated velocities.
    template<class V1, class V2>
    static void rigidTransform ( V1& points, V2& velocities, SReal tx, SReal ty, SReal tz, SReal rx, SReal ry, SReal rz )
    {
        typedef defaulttype::Vec<3,SReal> Vec3;
        typedef helper::Quater<SReal> Quat;
        Vec3 translation(tx,ty,tz);
        Quat rotation = Quat::createQuaterFromEuler(Vec3(rx,ry,rz));
        displace(points,translation,rotation);
        rotate(velocities,rotation);
    }


    //@}

};







} // namespace helper

} // namespace sofa

#endif
