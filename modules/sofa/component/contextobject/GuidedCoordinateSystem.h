/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
//
// C++ Interface: GuidedCoordinateSystem
//
// Description:
//
//
// Author: François Faure, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_COMPONENT_CONTEXTOBJECT_GUIDEDCOORDINATESYSTEM_H
#define SOFA_COMPONENT_CONTEXTOBJECT_GUIDEDCOORDINATESYSTEM_H

//#include <sofa/core/behavior/BaseConstraint.h>
#include <sofa/component/contextobject/CoordinateSystem.h>

namespace sofa
{

namespace component
{

namespace contextobject
{

/**
Make a CoordinateSystem turn around a given direction.
The rotation axis goes through the origin. It is defined by its direction.
The angular value is v(t) = wt + v0 where w is the velocity and v0 the initial value.

	@author François Faure
*/
class GuidedCoordinateSystem : public CoordinateSystem
{
public:
    SOFA_CLASS(GuidedCoordinateSystem, CoordinateSystem);
    typedef CoordinateSystem::Frame Frame;
    typedef CoordinateSystem::SpatialVector SpatialVector;
    typedef CoordinateSystem::Vec Vec;
    typedef CoordinateSystem::Rot Rot;

    GuidedCoordinateSystem();

    ~GuidedCoordinateSystem();

    // ContextObject
    virtual void apply();

    /// Initial transform (child wrt parent)
    //void setTransform( const Frame& frame );

    /// Initial transform (child wrt parent, vector in parent coordinates)
    //void setTransform( const Vec& origin, const Rot& orientation );

    /// Set the translation velocity, the direction of the rotation axis and the angular velocity. The vectors are given in parent coordinates. The axis is automatically normalized.
    //void setTransformRate( const Vec& linearVelocity, const Vec& rotationAxis, double omega );

    /// Linear velocity in parent coordinates
    //void setLinearVelocity( const Vec& linearVelocity );

    /// Angular velocity in parent coordinates
    //void setAngularVelocity( const Vec& linearVelocity );

    /// Set velocity with respect to parent, in parent coordinates
    void setVelocity( const SpatialVector& v );
    /// Velocity with respect to parent, in parent coordinates
    const SpatialVector& getVelocity() const;

protected:
    //Vec axis_;
    //double omega_;
    //Vec translationVelocity_;
    SpatialVector velocity_;
    Frame initialTransform_;

};


} // namespace contextobject

} // namespace component

} // namespace sofa

#endif


