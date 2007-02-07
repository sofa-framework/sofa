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
#include <iostream>
using std::cerr;
using std::endl;

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
//         std::cerr<<"GuidedCoordinateSystem:: axis too short"<<std::endl;
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
//     //cerr<<endl<<"GuidedCoordinateSystem::projectVelocity(), velo= "<<velo<<endl<<endl;
//     this->setVelocity( velo );
//     //cerr<<endl<<"GuidedCoordinateSystem::apply(), new relative velocity= "<< this->getRelativeVelocity() <<endl<<endl;
//
//     CoordinateSystem::apply();
// }

}

}

}
