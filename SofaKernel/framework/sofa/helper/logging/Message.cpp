/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*****************************************************************************
* User of this library should read the documentation
* in the TextMessaging.h file.
******************************************************************************/
#define SOFA_MESSAGE_CPP
#include "Message.h"

using std::endl ;
using std::string ;


namespace sofa
{

namespace helper
{

namespace logging
{

Message::TypeSet Message::AnyTypes = {Type::Info,Type::Advice,Type::Deprecated,
                                      Type::Warning,Type::Error,Type::Fatal};

Message Message::emptyMsg(CEmpty, TEmpty, ComponentInfo::SPtr(), EmptyFileInfo) ;

Message::Message(Class mclass, Type type,
                 const ComponentInfo::SPtr& componentInfo,
                 const FileInfo::SPtr& fileInfo) :
    m_componentinfo(componentInfo),
    m_fileInfo(fileInfo),
    m_class(mclass),
    m_type(type),
    m_id(-1)
{
}

Message::Message( const Message& msg )
    : m_componentinfo(msg.componentInfo())
, m_fileInfo(msg.fileInfo())
    , m_class(msg.context())
    , m_type(msg.type())
{
    m_stream << msg.message().str();
}

Message& Message::operator=( const Message& msg )
{
    m_fileInfo = msg.fileInfo();
    m_componentinfo = msg.componentInfo();
    m_class = msg.context();
    m_type = msg.type();
    m_stream << msg.message().str();
    return *this;
}

const SOFA_HELPER_API std::string toString(const Message::Type type)
{
    switch (type) {
    case Message::Advice:
        return "Advice";
    case Message::Deprecated:
        return "Deprecated";
    case Message::Info:
        return "Info";
    case Message::Warning:
        return "Warning";
    case Message::Error:
        return "Error";
    case Message::Fatal:
        return "Fatal";
    default:
        break;
    }
    return "Unknown type of message";
}

std::ostream& operator<< (std::ostream& s, const Message& m){
    s << "[" << m.sender() << "]: " << endl ;
    s << "    Message type   : " << toString(m.type()) << endl ;
    s << "    Message content: " << m.message().str() << endl ;

    if(m.fileInfo())
        s << "    source code loc: " << m.fileInfo()->filename << ":" << m.fileInfo()->line << endl ;

    if(m.componentInfo())
        s << "      component: " << m.componentInfo() ;

    return s;
}

bool Message::empty() const
{
    // getting the size without creating a copy like m_stream.str().size()
    std::streambuf* buf = m_stream.rdbuf();

    // the current position to restore it after
    std::stringstream::pos_type cur = buf->pubseekoff(0, std::ios_base::cur);

    // go to the end
    std::stringstream::pos_type end = buf->pubseekoff(0, std::ios_base::end);

    // restore initial position
    buf->pubseekpos(cur, m_stream.out);

    return end <= 0;
}

template<>

SOFA_HELPER_API Message& Message::operator<<(const FileInfo::SPtr &fi)
{
    m_fileInfo = fi;
    return *this;
}


} // logging
} // helper
} // sofa
