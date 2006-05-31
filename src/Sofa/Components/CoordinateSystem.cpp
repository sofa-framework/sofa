// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution

#include "CoordinateSystem.h"
#include <Sofa/Components/Graph/GNode.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace Sofa
{

namespace Components
{



CoordinateSystem::CoordinateSystem()
//: Abstract::ContextObject(name,n->getContext())
    : Core::BasicMechanicalModel()
{
    //n->addObject(this);
    setX(VecId::position());
    setV(VecId::velocity());
    setF(VecId::force());
    setDx(VecId::dx());

    getX()->clear();
    *getV() = Velocity( Vec(0,0,0), Vec(0,0,0) );
}


const CoordinateSystem::Frame&  CoordinateSystem::getFrame() const
{
    return *getX();
}
CoordinateSystem* CoordinateSystem::setFrame( const Frame& f )
{
    *getX()=f;
    return this;
}
CoordinateSystem* CoordinateSystem::setFrame( const Vec& translation, const Rot& rotation )
{
    getX()->setTranslationRotation(translation,rotation);
    return this;
}

const CoordinateSystem::Velocity&  CoordinateSystem::getVelocity() const
{
    return *getV();
}

CoordinateSystem* CoordinateSystem::setVelocity( const Velocity& v )
{
    *getV() = v;
    return this;
}

CoordinateSystem* CoordinateSystem::setLinearVelocity( const Vec& linearVelocity )
{
    *getV()=Velocity(linearVelocity, getAngularVelocity());
    return this;
}

const CoordinateSystem::Vec& CoordinateSystem::getLinearVelocity() const
{
    return getV()->freeVec;
}

CoordinateSystem* CoordinateSystem::setAngularVelocity( const Vec& angularVelocity )
{
    *getV()=Velocity(  angularVelocity, getLinearVelocity());
    return this;
}

const CoordinateSystem::Vec& CoordinateSystem::getAngularVelocity() const
{
    return getV()->lineVec;
}

void CoordinateSystem::updateContext(Core::Context* context)
{
//     cerr<<"CoordinateSystem, frame = "<<   getName() << endl;

    // store parent position and velocity
    Frame parentToWorld ( Rot(context->getLocalToWorldRotationQuat()), Vec(context->getLocalToWorldTranslation()) );
    Vec parentLinearVelocity ( context->getLinearVelocity() );
    Vec parentAngularVelocity ( context->getAngularVelocity() );
    Velocity parentSpatialVelocity ( parentLinearVelocity, parentAngularVelocity );
    Vec parentLinearAcceleration ( context->getLinearAcceleration() );

//     cerr<<"CoordinateSystem::apply(), thisToParent "<< *getX() <<endl;
//     cerr<<"CoordinateSystem::apply(), parentToWorld "<< parentToWorld <<endl;
//     cerr<<"CoordinateSystem::apply(), parentAngularVelocity= "<< parentAngularVelocity <<endl;
//     cerr<<"CoordinateSystem::apply(), parentLinearVelocity "<< parentLinearVelocity <<endl;

    // Project the relative velocity to the world.
    Vec vcenterProjected = parentToWorld.projectVector( getLinearVelocity() );
    const Vec& omega = getAngularVelocity();
    Vec omegaProjected = parentToWorld.projectVector(omega);
    omegaProjected += parentAngularVelocity;


    // Velocity induced by the rotation of the parent frame. Local origin is defined in parent frame.
    Vec originInParentProjected = parentToWorld.projectVector(getX()->getOriginInParent());
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
    Frame newLocalToWorld = parentToWorld * (*getX());
    Velocity newSpatialVelocity ( parentSpatialVelocity + parentToWorld * getVelocity() );
    // Convert to required types
    Common::Vec3d newTranslation = newLocalToWorld.getOriginInParent();
    Common::Mat3x3d newMatrix = newLocalToWorld.getRotationMatrix();
    Common::Quater<double> newQuat ( newLocalToWorld.getOrientation() );
    Common::Vec3d newLinearVelocity ( newSpatialVelocity.lineVec );
    Common::Vec3d newAngularVelocity ( newSpatialVelocity.freeVec );
    context->setLinearAcceleration( newLinearAcceleration );
    context->setLocalToWorld( newTranslation, newQuat.ptr(), newMatrix.ptr() );
    context->setLinearVelocity( newLinearVelocity );
    context->setAngularVelocity( newAngularVelocity );

//     cerr<<"CoordinateSystem::apply(), localToWorld= "<< context->getLocalToWorld() <<endl;
//     cerr<<"CoordinateSystem::apply(), spatial velocity= "<< context->getSpatialVelocity() <<endl;
//     cerr<<"CoordinateSystem::apply(), localLinearAcceleration= "<< context->getLinearAcceleration() <<endl;
//
//     cerr<<"CoordinateSystem::apply(), end--------------------------"<<endl;


}

CoordinateSystem::Frame* CoordinateSystem::getCoord(unsigned int index)
{
    if (index>=coords_.size())
        coords_.resize(index+1);
    if (coords_[index]==NULL)
        coords_[index] = new Frame;
    return coords_[index];
}
CoordinateSystem::Velocity* CoordinateSystem::getDeriv(unsigned int index)
{
    if (index>=derivs_.size())
        derivs_.resize(index+1);
    if (derivs_[index]==NULL)
        derivs_[index] = new Velocity;
    return derivs_[index];
}


void CoordinateSystem::setX( VecId i )
{
    x_=getCoord(i.index);
}
CoordinateSystem::Frame* CoordinateSystem::getX()
{
    return x_;
}
const CoordinateSystem::Frame* CoordinateSystem::getX() const
{
    return x_;
}

void CoordinateSystem::setV( VecId i )
{
    v_=getDeriv(i.index);
}
CoordinateSystem::Velocity* CoordinateSystem::getV()
{
    return v_;
}
const CoordinateSystem::Velocity* CoordinateSystem::getV() const
{
    return v_;
}

void CoordinateSystem::setF( VecId i )
{
    f_=getDeriv(i.index);
}
CoordinateSystem::Velocity* CoordinateSystem::getF()
{
    return f_;
}
const CoordinateSystem::Velocity* CoordinateSystem::getF() const
{
    return f_;
}

void CoordinateSystem::setDx( VecId i )
{
    dx_=getDeriv(i.index);
}
CoordinateSystem::Velocity* CoordinateSystem::getDx()
{
    return dx_;
}
const CoordinateSystem::Velocity* CoordinateSystem::getDx() const
{
    return dx_;
}

void CoordinateSystem::resetForce()
{
    *getF() = Velocity( Vec(0,0,0),Vec(0,0,0) );
}

/*void CoordinateSystem::printX( std::ostream& out )
{
        out<< getX();
}
void CoordinateSystem::printV( std::ostream& out )
{
        out<< getV();
}
void CoordinateSystem::printF( std::ostream& out )
{
        out<< getF();
}
void CoordinateSystem::printDx( std::ostream& out )
{
        out<< getDx();
}*/

void CoordinateSystem::printDOF( VecId v, std::ostream& out)
{
    if( v.type==V_COORD )
        out<<*getCoord(v.index);
    else if( v.type==V_DERIV )
        out<<*getDeriv(v.index);
    else
        out<<"CoordinateSystem::printDOF, unknown v.type = "<<v.type<<endl;
}


double CoordinateSystem::vDot(VecId a, VecId b)
{
    double r = 0.0;
    if (a.type == V_COORD && b.type == V_COORD)
    {
        assert(0);
    }
    else if (a.type == V_DERIV && b.type == V_DERIV)
    {
        const Velocity& va = *getDeriv(a.index);
        const Velocity& vb = *getDeriv(b.index);
        r += va * vb;
    }
    else
    {
        std::cerr << "Invalid dot operation ("<<a<<','<<b<<")\n";
    }
    return r;
}


void CoordinateSystem::vOp(VecId v, VecId a, VecId b, double f)
{
    if(v.isNull())
    {
        // ERROR
        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
        return;
    }
    if (a.isNull())
    {
        if (b.isNull())
        {
            // v = 0
            if (v.type == V_COORD)
            {
                Frame& vv = *getCoord(v.index);
                vv = Frame::identity();
            }
            else
            {
                Velocity& vv = *getDeriv(v.index);
                vv = Velocity( Vec(0,0,0), Vec(0,0,0) );
            }
        }
        else
        {
            if (b.type != v.type)
            {
                // ERROR
                std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                return;
            }
            if (v == b)
            {
                // v *= f
                if (v.type == V_COORD)
                {
                    *getCoord(v.index) *= f;
                }
                else
                {
                    *getDeriv(v.index) *= f;
                }
            }
            else
            {
                // v = b*f
                if (v.type == V_COORD)
                {
                    Frame& vv = *getCoord(v.index);
                    Frame& vb = *getCoord(b.index);
                    vv = vb * f;
                }
                else
                {
                    Velocity& vv = *getDeriv(v.index);
                    Velocity& vb = *getDeriv(b.index);
                    vv = vb * f;
                }
            }
        }
    }
    else
    {
        if (a.type != v.type)
        {
            // ERROR
            std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
            return;
        }
        if (b.isNull())   // v = a
        {
            if (v.type == V_COORD)
            {
                Frame& vv = *getCoord(v.index);
                Frame& va = *getCoord(a.index);
                vv = va;
            }
            else
            {
                Velocity& vv = *getDeriv(v.index);
                Velocity& va = *getDeriv(a.index);
                vv = va;
            }
        }
        else
        {
            if (v == a)
            {
                if (f==1.0)
                {
                    // v += b
                    if (v.type == V_COORD)
                    {
                        Frame& vv = *getCoord(v.index);
                        if (b.type == V_COORD)
                        {
                            Frame& vb = *getCoord(b.index);
                            vv += vb;
                        }
                        else
                        {
                            Velocity& vb = *getDeriv(b.index);
                            vv += vb;
                        }
                    }
                    else if (b.type == V_DERIV)
                    {
                        Velocity& vv = *getDeriv(v.index);
                        Velocity& vb = *getDeriv(b.index);
                        vv += vb;
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                        return;
                    }
                }
                else
                {
                    // v += b*f
                    if (v.type == V_COORD)
                    {
                        Frame& vv = *getCoord(v.index);
                        if (b.type == V_COORD)
                        {
                            Frame& vb = *getCoord(b.index);
                            vv += vb*f;
                        }
                        else
                        {
                            Velocity& vb = *getDeriv(b.index);
                            vv += vb*f;
                        }
                    }
                    else if (b.type == V_DERIV)
                    {
                        Velocity& vv = *getDeriv(v.index);
                        Velocity& vb = *getDeriv(b.index);
                        vv += vb*f;
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                        return;
                    }
                }
            }
            else
            {
                if (f==1.0)
                {
                    // v = a+b
                    if (v.type == V_COORD)
                    {
                        Frame& vv = *getCoord(v.index);
                        Frame& va = *getCoord(a.index);
                        if (b.type == V_COORD)
                        {
                            Frame& vb = *getCoord(b.index);
                            vv = va;
                            vv += vb;
                        }
                        else
                        {
                            Velocity& vb = *getDeriv(b.index);
                            vv = va;
                            vv += vb;
                        }
                    }
                    else if (b.type == V_DERIV)
                    {
                        Velocity& vv = *getDeriv(v.index);
                        Velocity& va = *getDeriv(a.index);
                        Velocity& vb = *getDeriv(b.index);
                        vv = va + vb;
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                        return;
                    }
                }
                else
                {
                    // v = a+b*f
                    if (v.type == V_COORD)
                    {
                        Frame& vv = *getCoord(v.index);
                        Frame& va = *getCoord(a.index);
                        if (b.type == V_COORD)
                        {
                            Frame& vb = *getCoord(b.index);
                            vv = va;
                            vv += vb*f;
                        }
                        else
                        {
                            Velocity& vb = *getDeriv(b.index);
                            vv = va;
                            vv += vb*f;
                        }
                    }
                    else if (b.type == V_DERIV)
                    {
                        Velocity& vv = *getDeriv(v.index);
                        Velocity& va = *getDeriv(a.index);
                        Velocity& vb = *getDeriv(b.index);
                        vv = va + vb*f;
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                        return;
                    }
                }
            }
        }
    }
}


} // namespace Components

} // namespace Sofa

