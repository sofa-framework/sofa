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
// C++ Implementation: GuidedCoordinateSystem
//
// Description:
//
//
// Author: François Faure, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <sofa/component/contextobject/GuidedCoordinateSystem.h>
#include <sofa/core/visual/VisualParams.h>
#include <iostream>



namespace sofa
{

namespace component
{

namespace contextobject
{

GuidedCoordinateSystem::GuidedCoordinateSystem(/*CoordinateSystem* c*/)
    : CoordinateSystem()
/*        , axis_( Vec(1,0,0) )
        , omega_( 1 )
        , translationVelocity_( Vec(0,0,0) )
        , initialTransform_( this->getTransform() )*/
{}


GuidedCoordinateSystem::~GuidedCoordinateSystem()
{}

// void GuidedCoordinateSystem::setTransformRate(  const Vec& translationVelocity, const Vec& v, double omega )
// {
//     translationVelocity_ = translationVelocity;
//     double length = v.norm();
//     if( length < 1.0e-6 )
//     {
//         serr<<"GuidedCoordinateSystem:: axis too short"<<sendl;
//         return;
//     }
//     axis_ = v * 1/length;
//     omega_ = omega;
// }
//
// void GuidedCoordinateSystem::setTransform( const Frame& f )
// {
//     initialTransform_ = f;
//     CoordinateSystem::setTransform( initialTransform_ );
// }

// void GuidedCoordinateSystem::setTransform( const Vec& origin, const Rot& orientation )
// {
//     initialTransform_.set(origin, orientation);
//     CoordinateSystem::setTransform( initialTransform_ );
// }

void GuidedCoordinateSystem::setVelocity( const SpatialVector& v )
{
    velocity_ = v;
}

const GuidedCoordinateSystem::SpatialVector& GuidedCoordinateSystem::getVelocity() const
{
    return velocity_;
}



void GuidedCoordinateSystem::apply()
{
    //this->setTransform( Frame(velocity_ * getContext()->getDt()) * this->getTransform() );

    CoordinateSystem::apply();
}
// void GuidedCoordinateSystem::apply()
// {
//     double angle = getContext()->getTime() * omega_ ;
//     Vec translation = translationVelocity_ * getContext()->getTime();
//     //this->setRelativePosition( Frame::set(translation, Rot::set( axis_*angle )) * initialTransform_ );
//     //this->setTransform( Frame( initialTransform_.getOrigin() + translation, Rot::set ( axis_*angle ) * ));
//
//     SpatialVector velo;
//     velo.setAngularVelocity( axis_*omega_ );
//     velo.setLinearVelocity( translationVelocity_ );
//     //serr<<sendl<<"GuidedCoordinateSystem::projectVelocity(), velo= "<<velo<<sendl<<sendl;
//     this->setVelocity( velo );
//     //serr<<sendl<<"GuidedCoordinateSystem::apply(), new relative velocity= "<< this->getRelativeVelocity() <<sendl<<sendl;
//
//     CoordinateSystem::apply();
// }

}

}

}
