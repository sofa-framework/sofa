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

#include <sofa/helper/fixed_array.h>

namespace sofa
{

namespace helper
{

namespace logging
{

helper::fixed_array<std::string,Message::TypeCount> s_messageTypeStrings;
ClangStyleMessageFormatter ClangStyleMessageFormatter::s_instance;


ClangStyleMessageFormatter::ClangStyleMessageFormatter()
{
    s_messageTypeStrings[Message::Info]       = "info";
    s_messageTypeStrings[Message::Deprecated] = "deprecated";
    s_messageTypeStrings[Message::Warning]    = "warning";
    s_messageTypeStrings[Message::Error]      = "error";
    s_messageTypeStrings[Message::Fatal]      = "fatal";
    s_messageTypeStrings[Message::TEmpty]     = "empty";
}


void ClangStyleMessageFormatter::formatMessage(const Message& m,std::ostream& out)
{
    if(m.sender()!="")
        out << m.fileInfo().filename << ":" << m.fileInfo().line << ":1: " << s_messageTypeStrings[m.type()] << ": " << m.message().str() << std::endl ;
    else
        out << m.fileInfo().filename << ":" << m.fileInfo().line << ":1: " << s_messageTypeStrings[m.type()] << ": ["<< m.sender() <<"] " << m.message().str() << std::endl ;
//    out << " message id: " << m.id() << std::endl ;
}


} // logging
} // helper
} // sofa
