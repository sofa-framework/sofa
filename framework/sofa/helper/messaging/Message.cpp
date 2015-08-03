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

//#include "Messagehandler.h"
#include "Message.h"

namespace sofa
{

namespace helper
{

namespace messaging
{

Message::Message(const string& mclass, const string& type, const string& message,
                 const string& sender, const FileInfo& fileInfo):
    m_sender(sender),
    m_fileInfo(fileInfo),
    m_message(message),
    m_class(mclass),
    m_type(type),
    m_id(-1)
{
}

const FileInfo& Message::fileInfo() const {
    return m_fileInfo;
}

const string& Message::message() const  {
    return m_message;
}

const std::string&  Message::sender() const  {
    return m_sender;
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
    s << "[" << m.sender() << "]: " << endl ;
    s << "         Message id: " << m.id() << endl ;
    s << "       Message type: " << m.type() << endl ;
    s << "    Message content: " << m.message() << endl ;
    s << "    source code loc: " << m.fileInfo().filename << ":" << m.fileInfo().line << endl ;
    return s;
}

} // messaging
} // helper
} // sofa
