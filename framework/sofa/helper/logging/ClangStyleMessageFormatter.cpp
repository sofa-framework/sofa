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
* Authors: Bruno Carrez                                                       *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*****************************************************************************
* User of this library should read the documentation
* in the messaging.h file.
******************************************************************************/

#include "ClangStyleMessageFormatter.h"
#include "Message.h"
using std::ostringstream ;

#include <string>
using std::string ;

#include <iostream>
using std::endl ;
using std::cout ;
using std::cerr ;

#include <sofa/helper/fixed_array.h>
using sofa::helper::fixed_array ;

namespace sofa
{

namespace helper
{

namespace logging
{

namespace unique {
    ClangStyleMessageFormatter clangstyleformatter;
}

// This string conversion is internal to the way clang
// format the info/warning/error. The typing are important.
// In the current state there is no reason to have this function in the public
// headers.
// Don't transform this into a static array. The lazy initialization is here
// to not rely in the implicit static initialization mechanisme.
string getTypeString(const Message::Type t){
    static bool isInited=false;
    static fixed_array<string,Message::TypeCount> messageTypeStrings;

    if(!isInited){
        messageTypeStrings[Message::Info]    = "info";
        messageTypeStrings[Message::Warning] = "warning";
        messageTypeStrings[Message::Error]   = "error";
        messageTypeStrings[Message::Fatal]   = "fatal";
        messageTypeStrings[Message::TEmpty]  = "empty";
        isInited = true ;
    }
    return messageTypeStrings[t];
}

void ClangStyleMessageFormatter::formatMessage(const Message& m,std::ostream& out)
{
    if(m.sender()!="")
        out << m.fileInfo().filename << ":" << m.fileInfo().line << ":1: " << getTypeString(m.type()) << ": " << m.message().rdbuf() << std::endl ;
    else
        out << m.fileInfo().filename << ":" << m.fileInfo().line << ":1: " << getTypeString(m.type()) << ": ["<< m.sender() <<"] " << m.message().rdbuf() << std::endl ;
    out << " message id: " << m.id() << std::endl ;
}


} // logging
} // helper
} // sofa
