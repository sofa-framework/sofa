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
*****************************************************************************/
#ifndef MESSAGE_H
#define MESSAGE_H
#include <iostream>
#include <string>
#include <sofa/helper/helper.h>

namespace sofa {
namespace core {
namespace objectmodel{
class Base ;
}
}
}

namespace sofa
{

namespace helper
{

namespace messaging
{


using sofa::core::objectmodel::Base ;
using std::ostream ;
using std::string ;

class SOFA_HELPER_API Message
{
public:
    // todo(damien): I don't like the two version one with the base an other with a sendername...
    // this is weird to me. But I see no other approach than using a templated class
    // to capture the nature of the sender and a BaseMessage...
    Message(string mclass, string type,  const std::string& sender, string source, int lineno) ;
    Message(string mclass, string type,  Base* sender, string source, int lineno) ;
    Message() ;

    Message& operator<=(const std::ostream& s) ;

    const string& source() const ;
    int           lineno() const ;
    const string& message() const ;
    Base*         sender() const ;
    const string& context() const ;
    const string& type() const ;
    const string& sendername() const ;
    int           id() const ;
    void          setId(int id) ;

    static Message empty ;

private:
    Base*   m_sender;
    string  m_sendername;
    string  m_source;
    int     m_lineno;
    string  m_message;
    string  m_class;
    string  m_type;
    int     m_id;
};

ostream& operator<< (ostream&, const Message&) ;

} // messaging
} // helper
} // sofa


#endif // MESSAGE_H
