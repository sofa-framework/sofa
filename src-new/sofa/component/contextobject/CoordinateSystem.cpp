// Author: Fran�is Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution

#include "CoordinateSystem.h"
#include <Sofa-old/Components/Common/ObjectFactory.h>
#include <Sofa-old/Components/Common/Vec.h>
#include <Sofa-old/Components/Graph/GNode.h>
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
    : positionInParent_( Frame::identity() )
    //, velocity_( Vec(0,0,0), Vec(0,0,0) )
{}


// CoordinateSystem::Frame  CoordinateSystem::getPositionInWorld() const
// {
//     return Abstract::BaseObject::getContext()->getPositionInWorld() * positionInParent_;
// }

const CoordinateSystem::Frame&  CoordinateSystem::getTransform() const
{
    return positionInParent_;
}

void CoordinateSystem::setTransform( const Frame& f )
{
    positionInParent_ = f;
}

// CoordinateSystem* CoordinateSystem::setOrigin( const Vec& v )
// {
//     positionInParent_ = Frame( v, this->getOrientation() );
//     return this;
// }
//
CoordinateSystem::Vec CoordinateSystem::getOrigin() const
{
//<<<<<<< .mine
    return positionInParent_.getOrigin();
    /*=======
        return this->relativePosition_.getOriginInParent();
    >>>>>>> .r414*/
}
//
// CoordinateSystem* CoordinateSystem::setOrientation( const Rot& r )
// {
//     positionInParent_ = Frame( this->getOrigin(), r );
//     return this;
// }
//
CoordinateSystem::Rot CoordinateSystem::getOrientation( ) const
{
//<<<<<<< .mine
    return positionInParent_.getOrientation();
    /*=======
        return this->relativePosition_.getOrientation();
    >>>>>>> .r414*/
}
//
// CoordinateSystem* CoordinateSystem::set( const Vec& v, const Rot& r )
// {
//     positionInParent_ = Frame( v, r );
//     return this;
// }

/*const CoordinateSystem::SpatialVector&  CoordinateSystem::getVelocity() const
{
    return velocity_;
}*/
/*CoordinateSystem::SpatialVector  CoordinateSystem::getVelocityInWorld() const
{
    return Abstract::BaseObject::getContext()->getVelocityInWorld() + getPositionInWorld() * velocity__;
}*/
// CoordinateSystem* CoordinateSystem::setVelocity( const SpatialVector& f )
// {
//     velocity_ = f;
//     return this;
// }


void CoordinateSystem::apply()
{
    //cerr<<"CoordinateSystem::apply(), frame = "<<   getName() <<", t="<<getContext()->getTime() << endl;
    Abstract::BaseContext* context = getContext();
    cerr<<"CoordinateSystem::apply, current position = "<<context->getPositionInWorld()<<endl;
    cerr<<"CoordinateSystem::apply, transform = "<<this->getTransform()<<endl;

    // store parent position and velocity
    Frame parentToWorld = context->getPositionInWorld();
    SpatialVector parentSpatialVelocity = context->getVelocityInWorld();
    Vec parentLinearVelocity = parentSpatialVelocity.getLinearVelocity() ;
    Vec parentAngularVelocity = parentSpatialVelocity.getAngularVelocity() ;
    Vec parentLinearAcceleration = context->getVelocityBasedLinearAccelerationInWorld() ;


    // Velocity induced by the rotation of the parent frame. Local origin is defined in parent frame.
    Vec originInParentProjected = parentToWorld.projectVector(getOrigin());
    Vec vinduced = parentAngularVelocity.cross( originInParentProjected );
    // Acceleration induced by the rotation of the parent frame. Local origin is defined in parent frame.
    Vec ainduced = parentAngularVelocity.cross( vinduced );




    // update context
    Common::Vec3d newLinearAcceleration = parentLinearAcceleration + ainduced;
    Frame newLocalToWorld = parentToWorld * getTransform();
    SpatialVector newSpatialVelocity ( parentSpatialVelocity /*+ newLocalToWorld * getVelocity()*/ );

    context->setVelocityBasedLinearAccelerationInWorld( newLinearAcceleration );
    context->setPositionInWorld( newLocalToWorld );
    context->setVelocityInWorld( newSpatialVelocity );
    cerr<<"CoordinateSystem::apply, new position = "<<context->getPositionInWorld()<<endl;

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
    obj->setTransform( Frame( vec, Rot::createFromRotationVector( rot ) ));
}

SOFA_DECL_CLASS(CoordinateSystem)

Creator< ObjectFactory, CoordinateSystem > CoordinateSystemClass("CoordinateSystem");

} // namespace Components

} // namespace Sofa

