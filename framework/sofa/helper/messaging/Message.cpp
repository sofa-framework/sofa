/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
* This component is open-source                                               *
*                                                                             *
* Authors: Damien Marchal                                                     *
*          Bruno Carrez                                                       *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*****************************************************************************
* User of this library should read the documentation
* in the TextMessaging.h file.
******************************************************************************/
#include <sstream>
using std::ostringstream ;

#include <iostream>
using std::endl ;

#include <string>
using std::string ;

#include <sofa/core/objectmodel/Base.h>
using sofa::core::objectmodel::Base ;

//#include "Messagehandler.h"
#include "Message.h"

namespace sofa
{

namespace helper
{

namespace messaging
{

Message::Message(){}
Message::Message(string mclass, string type,  Base* sender, string source, int lineno){
    m_sender = sender ;
    m_source = source ;
    m_lineno = lineno ;
    m_message = "undefined" ;
    m_class   = mclass;
    m_type    = type ;
    m_id      = -1 ;

    if(sender!=nullptr){
        ostringstream s;
        s << "[" << sender->getName() << "(" << sender->getClassName() <<   ")]: ";
        m_sendername = s.str();
    }else{
        m_sendername = "undefined" ;
    }
}

Message::Message(string mclass, string type,  const string& sendername, string source, int lineno){
    m_sender = nullptr ;
    m_sendername = sendername ;
    m_source = source ;
    m_lineno = lineno ;
    m_message = "undefined" ;
    m_class   = mclass;
    m_type    = type ;
    m_id      = -1 ;
}

Message& Message::operator<=(const std::ostream& s)
{
    ostringstream tmp ;
    tmp << s.rdbuf() ;

    m_message = tmp.str() ;
    return *this ;
}

const string& Message::source() const {
    return m_source;
}

int Message::lineno() const  {
    return m_lineno;
}

const string& Message::message() const  {
    return m_message;
}

Base*  Message::sender() const  {
    return m_sender;
}

const std::string&  Message::sendername() const  {
    return m_sendername;
}

const string& Message::context() const  {
    return m_class;
}

const string& Message::type() const {
    return m_type;
}

int Message::id() const  {
    return m_id;
}

void Message::setId(int id){
    m_id = id;
}

std::ostream& operator<< (std::ostream& s, const Message& m){
    s << "[" << m.sendername() << "]: " << endl ;
    s << "         Message id: " << m.id() << endl ;
    s << "       Message type: " << m.type() << endl ;
    s << "    Message content: " << m.message() << endl ;
    s << "    source code loc: " << m.source() << ":" << m.lineno() << endl ;
    return s;
}

Message Message::empty = Message() ;

} // messaging
} // helper
} // sofa
