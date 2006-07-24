// Author: Fran�is Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution

#include "CoordinateSystem.h"
#include <Sofa/Components/Graph/GNode.h>
#include <iostream>
using std::cerr;
using std::endl;
#ifdef WIN32
# include <windows.h>
#endif
#include <GL/gl.h>

namespace Sofa
{

namespace Components
{
CoordinateSystem::CoordinateSystem()
    : relativePosition_( Frame::identity() )
    , relativeVelocity_( Vec(0,0,0), Vec(0,0,0) )
{}


const CoordinateSystem::Frame&  CoordinateSystem::getRelativePosition() const
{
    return relativePosition_;
}

CoordinateSystem* CoordinateSystem::setRelativePosition( const Frame& f )
{
    relativePosition_ = f;
    return this;
}

const CoordinateSystem::SpatialVector&  CoordinateSystem::getRelativeVelocity() const
{
    return relativeVelocity_;
}
CoordinateSystem* CoordinateSystem::setRelativeVelocity( const SpatialVector& f )
{
    relativeVelocity_ = f;
    return this;
}


void CoordinateSystem::apply()
{
//     cerr<<"CoordinateSystem, frame = "<<   getName() << endl;
    Abstract::BaseContext* context = getContext();

    // store parent position and velocity
    Frame parentToWorld = context->getLocalFrame();
    Vec parentLinearVelocity ( context->getSpatialVelocity().getLinearVelocity() );
    Vec parentAngularVelocity ( context->getSpatialVelocity().getAngularVelocity() );
    SpatialVector parentSpatialVelocity ( parentLinearVelocity, parentAngularVelocity );
    Vec parentLinearAcceleration ( context->getVelocityBasedLinearAcceleration() );

//     cerr<<"CoordinateSystem::apply(), thisToParent "<< *getX() <<endl;
//     cerr<<"CoordinateSystem::apply(), parentToWorld "<< parentToWorld <<endl;
//     cerr<<"CoordinateSystem::apply(), parentAngularVelocity= "<< parentAngularVelocity <<endl;
//     cerr<<"CoordinateSystem::apply(), parentLinearVelocity "<< parentLinearVelocity <<endl;

    // Project the relative velocity to the world.
    Vec vcenterProjected = parentToWorld.projectVector( getRelativeVelocity().getLinearVelocity() );
    const Vec& omega = getRelativeVelocity().getAngularVelocity();
    Vec omegaProjected = parentToWorld.projectVector(omega);
    omegaProjected += parentAngularVelocity;


    // Velocity induced by the rotation of the parent frame. Local origin is defined in parent frame.
    Vec originInParentProjected = parentToWorld.projectVector(getRelativePosition().getOriginInParent());
    Vec vinduced = parentAngularVelocity.cross( originInParentProjected );
    // Acceleration induced by the rotation of the parent frame. Local origin is defined in parent frame.
    Vec ainduced = parentAngularVelocity.cross( vinduced );


//     cerr<<"CoordinateSystem::apply(), vcenterProjected = "<<vcenterProjected<<endl;
//     cerr<<"CoordinateSystem::apply(), omegaProjected= "<< omegaProjected <<endl;
//     cerr<<"CoordinateSystem::apply(), vinduced= "<<vinduced<<endl;
//     cerr<<"CoordinateSystem::apply(), ainduced= "<<ainduced<<endl;


    // update context
    //context->setLinearAcceleration( parentOriginAcceleration + ainduced );
    //context->setLocalToWorld( context->getLocalToWorld() * (*getX()) );
    //context->setSpatialVelocity( parentSpatialVelocity + context->getLocalToWorld() * getVelocity() );
    Common::Vec3d newLinearAcceleration = parentLinearAcceleration + ainduced;
    cerr<<"CoordinateSystem::updateContext, component "<<this->getName()<<", parentToWorld = "<<parentToWorld<<endl;
    cerr<<"CoordinateSystem::updateContext, *getX() = "<< getRelativePosition() <<endl;
    Frame newLocalToWorld = parentToWorld * getRelativePosition();
    cerr<<"CoordinateSystem::updateContext, newLocalToWorld = "<< newLocalToWorld <<endl;
    SpatialVector newSpatialVelocity ( parentSpatialVelocity + parentToWorld * getRelativeVelocity() );
    //// Convert to required types
    //Common::Vec3d newTranslation = newLocalToWorld.getOriginInParent();
    //Common::Mat3x3d newMatrix = newLocalToWorld.getRotationMatrix();
    //Common::Quater<double> newQuat ( newLocalToWorld.getOrientation() );
    //Common::Vec3d newLinearVelocity ( newSpatialVelocity.lineVec );
    //Common::Vec3d newAngularVelocity ( newSpatialVelocity.freeVec );
    context->setVelocityBasedLinearAcceleration( newLinearAcceleration );
    context->setLocalFrame( newLocalToWorld );
    context->setSpatialVelocity( newSpatialVelocity );

//     cerr<<"CoordinateSystem::apply(), localToWorld= "<< context->getLocalToWorld() <<endl;
//     cerr<<"CoordinateSystem::apply(), spatial velocity= "<< context->getSpatialVelocity() <<endl;
//     cerr<<"CoordinateSystem::apply(), localLinearAcceleration= "<< context->getLinearAcceleration() <<endl;
    //
//     cerr<<"CoordinateSystem::apply(), end--------------------------"<<endl;


}


void CoordinateSystem::draw()
{

    glPushAttrib(GL_COLOR_BUFFER_BIT | GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);
    glBegin( GL_LINES );
    glColor3f( 1.f,0.f,0.f );
    glVertex3f( 0.f,0.f,0.f );
    glVertex3f( 1.f,0.f,0.f );
    glColor3f( 0.f,1.f,0.f );
    glVertex3f( 0.f,0.f,0.f );
    glVertex3f( 0.f,1.f,0.f );
    glColor3f( 0.f,0.f,1.f );
    glVertex3f( 0.f,0.f,0.f );
    glVertex3f( 0.f,0.f,1.f );
    glEnd();
    glPopAttrib();
}




} // namespace Components

} // namespace Sofa

