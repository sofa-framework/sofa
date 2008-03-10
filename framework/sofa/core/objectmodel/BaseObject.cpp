/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace core
{

namespace objectmodel
{

BaseObject::BaseObject()
    : Base()
    , f_listening(initData( &f_listening, false, "listening", "if true, handle the events, otherwise ignore the events"))
    , f_printLog(initData( &f_printLog, false, "printLog", "if true, print logs at run-time"))
    , context_(NULL)
/*        , m_isListening(false)
        , m_printLog(false)*/
{}

BaseObject::~BaseObject()
{}

void BaseObject::setContext(BaseContext* n)
{
    context_ = n;
}

const BaseContext* BaseObject::getContext() const
{
    //return (context_==NULL)?BaseContext::getDefault():context_;
    return context_;
}

BaseContext* BaseObject::getContext()
{
    return (context_==NULL)?BaseContext::getDefault():context_;
    //return context_;
}

void BaseObject::init()
{ }

void BaseObject::bwdInit()
{ }

/// Update method called when variables used in precomputation are modified.
void BaseObject::reinit()
{ std::cout<<"WARNING: for this object, the reinit method does nothing.\n\n";}

/// Save the initial state for later uses in reset()
void BaseObject::storeResetState()
{ }

/// Reset to initial state
void BaseObject::reset()
{ }

void BaseObject::writeState( std::ostream& )
{ }

/// Called just before deleting this object
/// Any object in the tree bellow this object that are to be removed will be removed only after this call,
/// so any references this object holds should still be valid.
void BaseObject::cleanup()
{ }

/// Handle an event
void BaseObject::handleEvent( Event* e )
{
    cerr<<"BaseObject "<<getName()<<" ("<<getTypeName()<<") gets an event"<<endl;
    if( KeypressedEvent* ke = dynamic_cast<KeypressedEvent*>( e ) )
    {
        cerr<<"BaseObject "<<getName()<<" gets a key event: "<<ke->getKey()<<endl;
    }
}

// void BaseObject::setListening( bool b )
// {
//     m_isListening = b;
// }
//
// bool BaseObject::isListening() const
// {
//     return m_isListening;
// }
//
// BaseObject* BaseObject::setPrintLog( bool b )
// {
//     m_printLog = b;
//     return this;
// }
//
// bool BaseObject::printLog() const
// {
//     return m_printLog;
// }

double BaseObject::getTime() const
{
    return getContext()->getTime();
}


} // namespace objectmodel

} // namespace core

} // namespace sofa

