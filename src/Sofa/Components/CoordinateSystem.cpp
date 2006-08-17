// Author: Fran�is Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution

#include "CoordinateSystem.h"
#include <Sofa/Components/Common/ObjectFactory.h>
#include <Sofa/Components/Common/Vec.h>
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

CoordinateSystem* CoordinateSystem::setOrigin( const Vec& v )
{
    relativePosition_ = Frame::set
            ( v, this->getOrientation() );
    return this;
}

CoordinateSystem::Vec CoordinateSystem::getOrigin() const
{
    return this->getOrigin();
}

CoordinateSystem* CoordinateSystem::setOrientation( const Rot& r )
{
    relativePosition_ = Frame::set
            ( this->getOrigin(), r );
    return this;
}

CoordinateSystem::Rot CoordinateSystem::getOrientation( ) const
{
    return this->getOrientation();
}

CoordinateSystem* CoordinateSystem::set
( const Vec& v, const Rot& r )
{
    relativePosition_ = Frame::set
            ( v, r );
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
    //cerr<<"CoordinateSystem::apply(), frame = "<<   getName() <<", t="<<getContext()->getTime() << endl;
    Abstract::BaseContext* context = getContext();

    // store parent position and velocity
    Frame parentToWorld = context->getLocalFrame();
    SpatialVector parentSpatialVelocity = context->getSpatialVelocity();
    Vec parentLinearVelocity = parentSpatialVelocity.getLinearVelocity() ;
    Vec parentAngularVelocity = parentSpatialVelocity.getAngularVelocity() ;
    Vec parentLinearAcceleration = context->getVelocityBasedLinearAcceleration() ;


    // Velocity induced by the rotation of the parent frame. Local origin is defined in parent frame.
    Vec originInParentProjected = parentToWorld.projectVector(getRelativePosition().getOriginInParent());
    Vec vinduced = parentAngularVelocity.cross( originInParentProjected );
    // Acceleration induced by the rotation of the parent frame. Local origin is defined in parent frame.
    Vec ainduced = parentAngularVelocity.cross( vinduced );




    // update context
    Common::Vec3d newLinearAcceleration = parentLinearAcceleration + ainduced;
    Frame newLocalToWorld = parentToWorld * getRelativePosition();
    SpatialVector newSpatialVelocity ( parentSpatialVelocity + parentToWorld * getRelativeVelocity() );

    context->setVelocityBasedLinearAcceleration( newLinearAcceleration );
    context->setLocalFrame( newLocalToWorld );
    context->setSpatialVelocity( newSpatialVelocity );

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

using namespace Common;

void create(CoordinateSystem*& obj, ObjectDescription* arg)
{
    typedef Sofa::Components::CoordinateSystem::Frame Frame;
    typedef Sofa::Components::CoordinateSystem::Vec Vec;
    typedef Sofa::Components::CoordinateSystem::Rot Rot;
    //cout<<"create(CoordinateSystem*& obj, ObjectDescription*)"<< endl;
    obj = new CoordinateSystem;
    float x, y, z ;
    Vec vec;
    Vec rot;
    if (arg->getAttribute("origin"))
    {
        sscanf(arg->getAttribute("origin"),"%f%f%f",&x,&y,&z);
        vec = Vec(x,y,z);
    }
    if (arg->getAttribute("orientation"))
    {
        sscanf(arg->getAttribute("orientation"),"%f%f%f",&x,&y,&z);
        rot = Vec(x,y,z);
    }
    obj->setRelativePosition( Frame::set( vec, Rot::createFromRotationVector( rot ) ));
}

SOFA_DECL_CLASS(CoordinateSystem)

Creator< ObjectFactory, CoordinateSystem > CoordinateSystemClass("CoordinateSystem");

} // namespace Components

} // namespace Sofa

