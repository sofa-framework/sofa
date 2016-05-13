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
// Author: Fran�is Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution

#include <sofa/component/contextobject/CoordinateSystem.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/Node.h>
#include <iostream>
#include <sofa/helper/system/gl.h>




namespace sofa
{

namespace component
{

namespace contextobject
{

CoordinateSystem::CoordinateSystem()
    : origin(initData(&origin,defaulttype::Vec3f(),"origin", "Position of the local frame"))
    , orientation(initData(&orientation,defaulttype::Vec3f(),"orientation", "Orientation of the local frame"))
    , positionInParent_( Frame::identity() )
//, velocity_( Vec(0,0,0), Vec(0,0,0) )
{}


// CoordinateSystem::Frame  CoordinateSystem::getPositionInWorld() const
// {
//     return core::objectmodel::BaseObject::getContext()->getPositionInWorld() * positionInParent_;
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
    return positionInParent_.getOrigin();
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
    return positionInParent_.getOrientation();
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
    return core::objectmodel::BaseObject::getContext()->getVelocityInWorld() + getPositionInWorld() * velocity__;
}*/
// CoordinateSystem* CoordinateSystem::setVelocity( const SpatialVector& f )
// {
//     velocity_ = f;
//     return this;
// }


void CoordinateSystem::apply()
{
    //serr<<"CoordinateSystem::apply(), frame = "<<   getName() <<", t="<<getContext()->getTime() << endl;
    core::objectmodel::BaseContext* context = getContext();
    //serr<<"CoordinateSystem::apply, current position = "<<context->getPositionInWorld()<<sendl;
    //serr<<"CoordinateSystem::apply, transform = "<<this->getTransform()<<sendl;

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
    defaulttype::Vec3d newLinearAcceleration = parentLinearAcceleration + ainduced;
    Frame newLocalToWorld = parentToWorld * getTransform();
    SpatialVector newSpatialVelocity ( parentSpatialVelocity /*+ newLocalToWorld * getVelocity()*/ );

    context->setVelocityBasedLinearAccelerationInWorld( newLinearAcceleration );
    context->setPositionInWorld( newLocalToWorld );
    context->setVelocityInWorld( newSpatialVelocity );
    //serr<<"CoordinateSystem::apply, new position = "<<context->getPositionInWorld()<<sendl;

}



void CoordinateSystem::draw(const core::visual::VisualParams* vparams)
{
    /*
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
        */
}

using namespace sofa::defaulttype;



void CoordinateSystem::reinit()
{
    typedef CoordinateSystem::Frame Frame;
    typedef CoordinateSystem::Vec Vec;
    typedef CoordinateSystem::Rot Rot;
    setTransform( Frame( origin.getValue(), Rot::createFromRotationVector( orientation.getValue() ) ));
}


void CoordinateSystem::init()
{
    reinit();
}
SOFA_DECL_CLASS(CoordinateSystem)

int CoordinateSystemClass = core::RegisterObject("Translation and orientation of the local reference frame with respect to its parent")
        .add< CoordinateSystem >()
        ;

} // namespace contextobject

} // namespace component

} // namespace sofa

